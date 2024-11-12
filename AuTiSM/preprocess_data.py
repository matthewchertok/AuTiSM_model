# read data from GSD files, compute pairwise x, y, and z distances as 3 channel images, and apply 2D convolution

import gsd.hoomd
import os
import freud
import numpy as np
import pickle
from multiprocessing import Pool
import time
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

parser = argparse.ArgumentParser(description="Process GSD files to compute pairwise distances and apply 2D convolution.")
parser.add_argument("--window_length", type=int, default=50, help="Length of the window for splitting frames into sublists.")
args = parser.parse_args()
window_length = args.window_length


def process_file(
    file, last_frames_frac, reversible_bonding=True, n_backbone=100
):
    if reversible_bonding:
        data_dir = "../../data/production_reversible_bonds"
    else:
        data_dir = "../../data/production_irreversible_bonds"
    filepath = os.path.join(data_dir, file)

    with gsd.hoomd.open(filepath) as traj:
        if len(traj) == 0:
            print(f"{file} has empty trajectory")
            return {file: []}  # return empty list if the trajectory is empty

        # simulation box will be the same for all frames
        box = freud.box.Box.from_box(traj[0].configuration.box)
        positions_all_frames = np.array(
            [
                box.unwrap(frame.particles.position, frame.particles.image)
                for frame in traj
            ]
        )

        # Keep only the last fraction of frames after the bonds have formed
        positions_all_frames = positions_all_frames[
            -int(last_frames_frac * len(positions_all_frames)) :
        ]

        # Split the remaining frames into non-overlapping sublists
        # this will create multiple, partly independent, trajectories per run
        # allowing for more independent samples
        n_sublists = len(positions_all_frames) // window_length
        sublist_length = len(positions_all_frames) // n_sublists
        positions_all_frames = np.array_split(
            positions_all_frames[: sublist_length * n_sublists], n_sublists
        )
        # Get the indices corresponding to each array split
        frame_indices = [
            (i * sublist_length, (i + 1) * sublist_length)
            for i in range(n_sublists)
        ]
        

    results = []

    for i, sublist in enumerate(positions_all_frames):
        frames = frame_indices[i]
        # Extract backbone positions for all frames in the sublist
        backbone_positions_all_frames = sublist[:, :n_backbone, :]

        # Compute the radius of gyration for each frame
        center_of_mass_positions = np.mean(backbone_positions_all_frames, axis=1)
        ri_minus_rcom = (
            backbone_positions_all_frames - center_of_mass_positions[:, np.newaxis, :]
        )
        summands = (
            np.einsum("...ij,...ik->...jk", ri_minus_rcom, ri_minus_rcom) / n_backbone
        )
        eigenvalues = np.linalg.eigh(summands)[0]
        rgs = np.sqrt(np.sum(eigenvalues, axis=1))

        # Compute relative shape anisotropy for each frame
        numerator = 3 / 2 * np.sum(eigenvalues**2, axis=1)
        denominator = np.sum(eigenvalues, axis=1) ** 2
        kappa_squared = numerator / denominator - 1 / 2

        # compute acylindricity for each frame
        c = eigenvalues[:, 1] - eigenvalues[:, 0]

        # compute asphericity for each frame
        b = eigenvalues[:, 2] - (eigenvalues[:, 0] + eigenvalues[:, 1]) / 2

        f = file.split("f=")[1].split(",")[0]
        beta = file.split("beta=")[1].split(",")[0]
        kappa = file.split("kappa=")[1].split(",")[0]
        plate_speed = file.split("plate_speed=constant=")[1].split("_")[0]
        shear_rate = 2 * float(plate_speed) / 100
        combo = file.split("combo")[1].split("_")[0]
        chain = file.split("chain")[1].split("_")[0]
        combo_chain_id = combo + chain

        # Append the results for this sublist
        results.append(
            {
                "rg": rgs,
                "relative_shape_anisotropy": kappa_squared,
                "acylindricity": c,
                "asphericity": b,
                "frames": frames,
                "relative_positions" : ri_minus_rcom,
                "relative_positions_reversed" : ri_minus_rcom[::-1],  # include reversed positions for data augmentation
                "f": float(f),
                "beta": float(beta),
                "kappa": float(kappa),
                "shear_rate": float(shear_rate),
                "combo_chain_id": combo_chain_id
            }
        )

    # Return a dictionary with the results for this file
    return {file: results}


def main(reversible_bonding=True):
    if reversible_bonding:
        data_dir = "../../data/production_reversible_bonds"
    else:
        data_dir = "../../data/production_irreversible_bonds"
    files = sorted(os.listdir(data_dir))
    n_cores = os.cpu_count()
    print(f"Processing {len(files)} files using {n_cores} cores")

    # Process files in parallel and collect results
    with Pool(n_cores) as pool:
        results = pool.starmap(
            process_file,
            [(file, 0.10, reversible_bonding) for file in files],
        )

    # Combine all results into a single dictionary
    combined_results = {}

    # this works because each result is a dictionary with a single key
    for result in results:
        for filename, results_list in result.items():
            if results_list:  # Exclude entries where results_list is an empty list
                combined_results[filename] = results_list

    # Dump the combined dictionary to a single pickle file
    filename = "rg_and_anisotropy_production.pkl"
    with open(filename, "wb") as f:
        pickle.dump(combined_results, f)


if __name__ == "__main__":
    start = time.time()
    main(reversible_bonding=False)
    end = time.time()
    print(f"Processing took {end - start} seconds")
