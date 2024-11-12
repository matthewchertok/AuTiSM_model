# Select random input and generate reconstruction
import matplotlib.animation as animation
from train_autism_model import VideoVAE
import torch
import pickle
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import plotly.express as px

latent_dim = 3

# Change directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("rg_and_anisotropy_production_dataframe.pkl", "rb") as f:
    data = pickle.load(f)

inputs = np.array(data["pdist_evecs"].tolist(), dtype=np.float32)
inputs = inputs.transpose(
    0, 4, 1, 2, 3
)  # Reshape from (N,D,H,W,C) to (N,C,D,H,W) for PyTorch Conv3d
inputs = torch.tensor(inputs, dtype=torch.float32)

# Get dimensions from the input data
input_depth = inputs.size(2)  # Depth/time dimension
input_height = inputs.size(3)  # Height dimension
input_width = inputs.size(4)  # Width dimension

print(
    f"Input dimensions: Depth={input_depth}, Height={input_height}, Width={input_width}"
)

# Training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VideoVAE(
    input_channels=3,
    latent_dim=latent_dim,
    input_depth=5,
    input_height=100,
    input_width=100,
).to(device)

model.load_state_dict(
    torch.load(
        f"outputs/weights/trained_vae_model_latentdim_{latent_dim}.pth",
        map_location=device,
    )
)
model.eval()


# Get the first sample from the dataset
sample = inputs[0].unsqueeze(0).to(device)  # Add batch dimension and move to device

# Generate reconstruction
with torch.no_grad():
    reconstruction, _, _ = model(sample)

# Convert tensors to numpy arrays and transpose to (D,H,W,C) format
original = sample[0].cpu().numpy()
reconstructed = reconstruction[0].cpu().numpy()

print(f"Original min: {original.min()}, max: {original.max()}")
print(f"Reconstructed min: {reconstructed.min()}, max: {reconstructed.max()}")


def create_animation(video_data, output_filename):
    # Transpose from (C,D,H,W) to (D,H,W,C) for matplotlib
    frames = video_data.transpose(1, 2, 3, 0)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis("off")
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)

    writer = FFMpegWriter(fps=5, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(output_filename, writer=writer)
    plt.close(fig)


# Create animations from original and reconstructed sequences
create_animation(original, "original_sequence.mp4")
create_animation(reconstructed, "reconstructed_sequence.mp4")


# Randomly select 1000 samples
num_samples = 1000
random_indices = np.random.choice(len(inputs), num_samples, replace=False)
selected_inputs = inputs[random_indices]
# Create DataLoader for the selected inputs
batch_size = 100
dataset = torch.utils.data.TensorDataset(selected_inputs)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Get latent representations
model.eval()
with torch.no_grad():
    latent_vectors = []
    for batch, in dataloader:  # Note the comma after batch (unpacking single-item tuple)
        batch = batch.to(device)
        _, mu, _ = model(batch)
        latent_vectors.append(mu.cpu().numpy())
    latent_vectors = np.concatenate(latent_vectors, axis=0)

# Get corresponding parameters
params = {
    "f": data["f"].iloc[random_indices],
    "beta": data["beta"].iloc[random_indices],
    "kappa": data["kappa"].iloc[random_indices],
    "shear_rate": data["shear_rate"].iloc[random_indices],
}

# Create scatter plots using plotly
for param_name, param_values in params.items():
    fig = px.scatter_3d(
        x=latent_vectors[:, 0],
        y=latent_vectors[:, 1],
        z=latent_vectors[:, 2],
        color=param_values,
        title=f"Latent Space Colored by {param_name}",
        labels={
            "x": "Latent Dim 1",
            "y": "Latent Dim 2",
            "z": "Latent Dim 3",
            "color": param_name,
        },
    )
    fig.write_html(f"latent_space_{param_name}.html")
