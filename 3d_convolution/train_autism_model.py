import torch
import os
import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

latent_dim = 3


class VideoVAE(nn.Module):
    def __init__(
        self,
        input_channels=3,
        latent_dim=3,
        input_depth=5,
        input_height=100,
        input_width=100,
    ):
        super(VideoVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), ceil_mode=False),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), ceil_mode=False),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), ceil_mode=False),
        )

        # Compute the size of the flattened feature maps
        sample_input = torch.zeros(
            1, input_channels, input_depth, input_height, input_width
        )
        sample_output = self.encoder(sample_input)
        self.flattened_size = sample_output.view(1, -1).size(1)
        self.output_shape = sample_output.size()[1:]  # (C, D, H, W)

        # FC layers for mean and variance
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_var = nn.Linear(self.flattened_size, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                64,
                32,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                output_padding=(0, 1, 1),
            ),
            nn.ReLU(),
            nn.ConvTranspose3d(
                32,
                16,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                output_padding=(0, 1, 1),
            ),
            nn.ReLU(),
            nn.ConvTranspose3d(
                16,
                3,
                kernel_size=(1, 7, 7),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                output_padding=(0, 1, 1),
            ),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, *self.output_shape)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

if __name__ == "__main__":
    start = time.time()

    # Loss function
    def vae_loss(recon_x, x, mu, log_var):
        reconstruction_loss = F.mse_loss(recon_x, x, reduction="sum")
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return reconstruction_loss + kld_loss


    "end vae_loss"

    with open("rg_and_anisotropy_production_dataframe.pkl", "rb") as f:
        data = pickle.load(f)

    inputs = np.array(data["pdist_evecs"].tolist(), dtype=np.float32)
    inputs = inputs.transpose(
        0, 4, 1, 2, 3
    )  # Reshape from (N,D,H,W,C) to (N,C,D,H,W) for PyTorch Conv3d
    inputs = torch.tensor(inputs, dtype=torch.float32)
    dataset = TensorDataset(inputs)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Get dimensions from the input data
    input_depth = inputs.size(2)  # Depth/time dimension
    input_height = inputs.size(3)  # Height dimension
    input_width = inputs.size(4)  # Width dimension

    print(
        f"Input dimensions: Depth={input_depth}, Height={input_height}, Width={input_width}"
    )

    # Training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model with matching dimensions
    model = VideoVAE(
        input_channels=3,
        latent_dim=latent_dim,
        input_depth=input_depth,
        input_height=input_height,
        input_width=input_width,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 100

    dataset_size = inputs.size(0)
    batch_size = dataloader.batch_size

    print("Starting training...")

    # Training loop
    # Store losses for plotting
    losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            batch = batch[0].to(
                device
            )  # Get first element since TensorDataset returns a tuple

            optimizer.zero_grad()
            recon_batch, mu, log_var = model(batch)
            loss = vae_loss(recon_batch, batch, mu, log_var)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        losses.append(avg_loss)

    print("Training complete!")
    # Create directories if they don't exist
    os.makedirs("outputs/weights", exist_ok=True)
    os.makedirs("outputs/losses", exist_ok=True)

    # Save the trained model
    torch.save(model.state_dict(), f"outputs/weights/trained_vae_model_latentdim_{latent_dim}.pth")

    # Save the losses for future reference
    with open(f"outputs/losses/training_losses_latentdim_{latent_dim}.pkl", "wb") as f:
        pickle.dump(losses, f)

    # After training, plot the loss curve
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"loss_curve_latentdim_{latent_dim}.png")
    plt.close()

    end = time.time()
    elapsed_time = end - start
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training took {int(hours):02}:{int(minutes):02}:{seconds:.2f} (hh:mm:ss)")

