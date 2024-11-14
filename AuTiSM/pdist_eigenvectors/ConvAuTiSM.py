import torch
import torch.nn as nn
import math

class PositionEncoder(nn.Module):
    """A neural network module for encoding position images into a latent representation.

    This encoder uses convolutional layers to process images and produce
    a fixed-size latent representation.

    Args:
        latent_dim_A (int, optional): The dimensionality of the latent space A. Defaults to 64.

    Returns:
        torch.Tensor: The latent representation of shape (batch_size, latent_dim_A)
                      containing the encoded position image information.
    """

    def __init__(self, latent_dim_A=64):
        super(PositionEncoder, self).__init__()
        self.latent_dim_A = latent_dim_A
        # Convolutional layers with appropriate activation functions
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # Output: (batch_size, 16, H, W)
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample by factor of 2 -> (H/2, W/2)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Output: (batch_size, 32, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample by factor of 2 -> (H/4, W/4)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output: (batch_size, 64, H/4, W/4)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Output: (batch_size, 64, 1, 1)
        )
        self.fc = nn.Linear(64, latent_dim_A)

    def forward(self, x):
        # x shape: (batch_size, channels=3, height=H, width=W)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 64)
        latent_A = self.fc(x)  # Map to (batch_size, latent_dim_A)
        return latent_A


class PositionDecoder(nn.Module):
    """A decoder module that converts latent representation to position images.

    This module uses convolutional layers to decode latent vectors into images.
    The strides and paddings are computed dynamically to handle input images of any size.

    Args:
        latent_dim_A (int, optional): The dimensionality of the latent space A. Defaults to 64.
        output_shape (tuple): The shape of the output images (channels, height, width).

    Returns:
        torch.Tensor: Decoded images. Shape: (batch_size, channels, height, width)
    """

    def __init__(self, latent_dim_A=64, output_shape=(3, None, None)):
        super(PositionDecoder, self).__init__()
        self.latent_dim_A = latent_dim_A
        channels, height, width = output_shape
        self.channels = channels
        self.height = height
        self.width = width

        # Calculate initial spatial dimensions after upsampling
        # Since we used MaxPool2d(2) three times in the encoder, we need to upsample by 2^3 = 8
        self.init_height = math.ceil(self.height / 8)
        self.init_width = math.ceil(self.width / 8)

        # Linear layer to expand the latent vector to the initial feature map size
        self.fc = nn.Linear(latent_dim_A, 64 * self.init_height * self.init_width)

        # Convolutional layers with appropriate activation functions and dynamic upsampling
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Output: (batch_size, 64, init_H, init_W)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Upsample -> (H/4, W/4)
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # Output: (batch_size, 32, H/4, W/4)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Upsample -> (H/2, W/2)
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # Output: (batch_size, 16, H/2, W/2)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Upsample -> (H, W)
            nn.Conv2d(16, channels, kernel_size=3, padding=1),  # Output: (batch_size, channels, H, W)
            nn.Sigmoid()  # output is normalized distance matrix, so constrain to [0, 1]
        )

    def forward(self, latent_A):
        # latent_A shape: (batch_size, latent_dim_A)
        batch_size = latent_A.size(0)
        x = self.fc(latent_A)  # Map to (batch_size, 64 * init_H * init_W)
        x = x.view(batch_size, 64, self.init_height, self.init_width)  # Reshape to feature map
        x = self.decoder(x)
        # Crop to the original size in case of rounding errors
        x = x[:, :, :self.height, :self.width]
        return x


class TemporalEncoder(nn.Module):
    """Temporal encoder that processes sequences of latent vectors into a single latent representation.

    This module takes a sequence of latent vectors (latent space A) that were generated from
    position encodings over multiple frames, and encodes them into a single latent vector
    (latent space B) using a GRU network.

    Args:
        input_size (int, optional): Dimensionality of input latent vectors (latent space A).
            Defaults to 64.
        latent_dim_B (int, optional): Dimensionality of output latent vector (latent space B).
            Defaults to 8.

    Returns:
        torch.Tensor: Single latent vector in space B encoding the temporal information.
            Shape: (batch_size, latent_dim_B)
    """

    def __init__(self, input_size=64, latent_dim_B=8):
        super(TemporalEncoder, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=latent_dim_B, batch_first=True
        )

    def forward(self, latent_A_seq):
        _, h_n = self.gru(latent_A_seq)
        latent_B = h_n.squeeze(0)
        return latent_B


class TemporalDecoder(nn.Module):
    """
    Temporal decoder that reconstructs latent sequences from a latent vector.

    Takes a latent vector created by TemporalEncoder and uses it to reconstruct a sequence
    of latent vectors in space A.

    Args:
        latent_dim_B (int, optional): Dimension of the GRU hidden state. Defaults to 8.
        output_size (int, optional): Dimension of the output vectors at each timestep.
            Should match latent_dim_A. Defaults to 64.
        seq_length (int, optional): Length of the sequence to generate. Must be specified.

    Returns:
        torch.Tensor: Reconstructed sequence of latent vectors A with shape
            [batch_size, seq_length, output_size]
    """

    def __init__(self, latent_dim_B=8, output_size=64, seq_length=None):
        super(TemporalDecoder, self).__init__()
        self.seq_length = seq_length
        self.gru = nn.GRU(
            input_size=output_size, hidden_size=latent_dim_B, batch_first=True
        )
        self.fc = nn.Linear(latent_dim_B, output_size)

    def forward(self, latent_B):
        batch_size = latent_B.size(0)
        h_0 = latent_B.unsqueeze(0)  # Initial hidden state
        inputs = torch.zeros(batch_size, self.seq_length, self.fc.out_features).to(latent_B.device)
        output, _ = self.gru(inputs, h_0)
        output = self.fc(output)
        return output


class AuTiSM_Model(nn.Module):
    """
    AuTiSM (Autoencoded Time Series Mapping) Model for processing sequential particle data.

    This model implements a hierarchical autoencoder architecture that processes particle position data
    across multiple frames.

    Architecture:
        - Frame-level encoding/decoding: Processes individual frames of particle positions
        - Sequence-level encoding/decoding: Handles temporal relationships between frames

    Args:
        N (int): Spatial dimension of the images (assumed square images of size N x N)
        M (int): Number of frames in the sequence
        latent_dim_A (int, optional): Dimensionality of latent space A. Defaults to 64.
        latent_dim_B (int, optional): Dimensionality of latent space B. Defaults to 8.

    Returns:
        Tensor of same shape as input, containing reconstructed particle positions
        across all frames in the sequence
    """

    def __init__(self, N, M, latent_dim_A=64, latent_dim_B=8):
        super(AuTiSM_Model, self).__init__()
        self.N = N  # Spatial dimension of the images
        self.M = M  # Number of frames
        self.frame_encoder = PositionEncoder(latent_dim_A=latent_dim_A)
        self.frame_decoder = PositionDecoder(latent_dim_A=latent_dim_A, output_shape=(3, N, N))
        self.sequence_encoder = TemporalEncoder(input_size=latent_dim_A, latent_dim_B=latent_dim_B)
        self.sequence_decoder = TemporalDecoder(
            latent_dim_B=latent_dim_B, output_size=latent_dim_A, seq_length=M
        )

    def forward(self, x):
        # x shape: (batch_size, channels=3, M, height=N, width=N)
        # Reshape x to combine batch and M dimensions
        batch_size = x.size(0)
        reshaped_x = x.permute(0, 2, 1, 3, 4).reshape(-1, 3, self.N, self.N)
        
        # Process all frames in parallel
        latent_A_all = self.frame_encoder(reshaped_x)  # Shape: (batch_size*M, latent_dim_A)
        latent_A_seq = latent_A_all.view(batch_size, self.M, -1)  # Shape: (batch_size, M, latent_dim_A)
        
        # Process sequence
        latent_B = self.sequence_encoder(latent_A_seq)  # Shape: (batch_size, latent_dim_B)
        reconstructed_latent_A_seq = self.sequence_decoder(latent_B)  # Shape: (batch_size, M, latent_dim_A)
        
        # Decode all frames in parallel
        reshaped_latent_A = reconstructed_latent_A_seq.reshape(-1, reconstructed_latent_A_seq.size(-1))
        reconstructed_frames_all = self.frame_decoder(reshaped_latent_A)  # Shape: (batch_size*M, 3, N, N)
        
        # Reshape back to original dimensions
        reconstructed_x = reconstructed_frames_all.view(batch_size, self.M, 3, self.N, self.N).permute(0, 2, 1, 3, 4)
        
        return reconstructed_x, latent_B