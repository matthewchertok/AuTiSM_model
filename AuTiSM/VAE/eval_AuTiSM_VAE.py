import os
import torch
import torch.nn as nn
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from AuTiSM_VAE import AuTiSM_Model
from torch.utils.data import Dataset, DataLoader
import pickle
import matplotlib.pyplot as plt

# Check if CUDA is available and set it as default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the absolute path of the current script and change to its directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


# Assuming 'data_list' is your dataset containing tensors of shape (M, N, 3)
class ParticleDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].to(device)


"end class"

# Load data from pickle file
with open("inputs_window_size_50_frames.pkl", "rb") as f:
    data_list = pickle.load(f)

# Convert DataFrame columns to lists and combine
data_list = (
    data_list["relative_positions"].tolist()
    + data_list["relative_positions_reversed"].tolist()
)
data_list = [torch.tensor(arr) for arr in data_list]
# Initialize dataset and dataloader
dataset = ParticleDataset(data_list)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize the model
N = dataset[0].size(1)  # Number of particles
M = dataset[0].size(0)  # Number of frames

model = AuTiSM_Model(N=N, M=M).to(device)
# Load model weights
weights_path = os.path.join(script_dir, "AuTiSM_VAE.pth")
model.load_state_dict(torch.load(weights_path, map_location=device))

# Evaluate model on a random sample
model.eval()  # Set model to evaluation mode


criterion = nn.MSELoss()

with torch.no_grad():
    # Get random sample
    idx = torch.randint(0, len(dataset), (1,)).item()
    sample = dataset[idx].unsqueeze(0).float()

    # Get reconstruction
    reconstructed, _, _ = model(sample)

    # Calculate reconstruction error
    reconstruction_error = criterion(reconstructed, sample)
"end with"

# Print error and create visualization outside the with block
print(f"Reconstruction error for sample {idx}: {reconstruction_error.item():.4f}")

# Create movies for both original and reconstructed
sample = sample.cpu().numpy()  # move to the CPU for plotting
reconstructed = reconstructed.cpu().numpy()
data = [(sample, "original"), (reconstructed, "reconstructed")]

for tensor, name in data:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter([], [], [])
    ax.set_xlim(tensor[0, :, :, 0].min(), tensor[0, :, :, 0].max())
    ax.set_ylim(tensor[0, :, :, 1].min(), tensor[0, :, :, 1].max())
    ax.set_zlim(tensor[0, :, :, 2].min(), tensor[0, :, :, 2].max())
    ax.set_title(name.capitalize())
    ax.grid(True)

    def animate(frame):
        scat._offsets3d = (tensor[0, frame, :, 0], tensor[0, frame, :, 1], tensor[0, frame, :, 2])
        return scat,

    anim = FuncAnimation(fig, animate, frames=M, interval=100)
    writer = FFMpegWriter(fps=10)
    anim.save(f"{name}_particles.mp4", writer=writer)
    plt.close()
