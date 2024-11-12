import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from AuTiSM_VAE import AuTiSM_Model

# Change to the script's directory using absolute path
script_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(script_dir)

# Check if CUDA is available and set it as default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    print("CUDA is not available. GPU acceleration required.")
    exit()


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
model = AuTiSM_Model(N=N, M=M, latent_dim_A=256, latent_dim_B=8).to(device)

# Define loss function and optimizer
optimizer = torch.optim.Adam(model.parameters())

def loss_function(reconstructed_x, x, mu, logvar):
    reconstruction_loss = nn.MSELoss()(reconstructed_x, x)
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + KLD

# Training loop
num_epochs = 100
losses = []

print("Training model...")
for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0

    for batch in dataloader:
        # Forward pass
        reconstructed_x, mu, logvar = model(batch.float())
        loss = loss_function(reconstructed_x, batch.float(), mu, logvar)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Plot loss vs epoch
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.grid(True)
plt.savefig("AuTiSM_VAE.png")
plt.close()

# Save model weights
torch.save(model.state_dict(), "AuTiSM_VAE.pth")
print("Model training complete. Weights saved to 'AuTiSM_VAE.pth'.")
