import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from ConvAuTiSM import AuTiSM_Model
import numpy as np

latent_dim_A = 256
latent_dim_B = 256

# Change to the script's directory using absolute path
script_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(script_dir)

# Check if CUDA is available and set it as default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    print("CUDA is not available. GPU acceleration required.")
    exit()


# Load data from pickle file
with open("inputs_window_size_50_frames.pkl", "rb") as f:
    df = pickle.load(f)

data_list = np.array(df["pdist"].tolist(), dtype=np.float32)
data_list = data_list.transpose(
    0, 4, 1, 2, 3
)  # Reshape from (N,D,H,W,C) to (N,C,D,H,W)
data_list = torch.tensor(data_list, dtype=torch.float32)
# Initialize dataset and dataloader
dataset = TensorDataset(data_list)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize the model
N = data_list[0].size(2)  # Number of particles
M = data_list[0].size(1)  # Number of frames
model = AuTiSM_Model(N=N, M=M, latent_dim_A=latent_dim_A, latent_dim_B=latent_dim_B).to(
    device
)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
num_epochs = 100
losses = []

print("Training model...")
for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0

    for batch in dataloader:
        batch = batch[0].to(
            device
        )  # Get first element since TensorDataset returns a tuple
        # Forward pass
        outputs, latent_B = model(batch.float())
        loss = criterion(outputs, batch.float())

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
plt.savefig(f"training_loss_latentA={latent_dim_A}_latentB={latent_dim_B}.png")
plt.close()

# Save model weights
filename = f"AuTiSM_latentA={latent_dim_A}_latentB={latent_dim_B}.pth"
torch.save(model.state_dict(), filename)
print("Model training complete. Model weights saved as", filename)
