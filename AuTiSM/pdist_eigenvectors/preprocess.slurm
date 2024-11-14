#!/bin/bash
#SBATCH --job-name=preprocess_data_long_window
#SBATCH --output=slurm_outs/preprocess_%j.out
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=100        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=05:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=mc7345@princeton.edu
module purge
module load anaconda3/2024.6
conda activate '/home/mc7345/anaconda3/envs/matthew'

python preprocess_data.py --window_length 50
python dict_to_dataframe.py --save_name 'inputs_window_size_50_frames'