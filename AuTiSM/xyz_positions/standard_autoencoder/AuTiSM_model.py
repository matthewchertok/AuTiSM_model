import torch
import torch.nn as nn


class BondVectorEncoder(nn.Module):
    """A neural network module for encoding bond vector sequences into a latent representation.

    This encoder uses a linear layer to map input bond vectors to a higher-dimensional latent space,
    followed by a GRU (Gated Recurrent Unit) to process the sequence and produce
    a fixed-size latent representation.

    Args:
        input_size (int, optional): The number of expected features in the input x. Defaults to 3.
        latent_dim_A (int, optional): The dimensionality of the latent space A. Defaults to 64.
        hidden_size (int, optional): The dimensionality of the GRU input size. Defaults to 128.

    Attributes:
        fc_in (nn.Linear): Linear layer mapping input_size to hidden_size.
        gru (nn.GRU): The GRU layer used for sequence processing.

    Returns:
        torch.Tensor: The latent representation of shape (batch_size, latent_dim_A)
                      containing the encoded position sequence information.
    """

    def __init__(self, input_size=3, latent_dim_A=64, hidden_size=128):
        super(BondVectorEncoder, self).__init__()
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size, hidden_size=latent_dim_A, batch_first=True
        )

    def forward(self, x):
        x = self.fc_in(x)
        _, h_n = self.gru(x)
        latent_A = h_n.squeeze(0)
        return latent_A


class BondVectorDecoder(nn.Module):
    """A decoder module that converts latent representation to bond vector sequences.

    This module uses a GRU (Gated Recurrent Unit) to decode latent vectors into
    sequences of 3D positions. It consists of a GRU layer followed by a fully
    connected layer to map to the output space. The output is passed through
    tanh to ensure values are in range (-1, 1).

    Args:
        latent_dim_A (int, optional): The size of the GRU hidden state. Defaults to 64.
        output_size (int, optional): The dimensionality of the output position vectors. Defaults to 3.
        seq_length (int, optional): The length of the output sequence. Defaults to None.

    Forward Args:
        latent_A (torch.Tensor): The latent vector to be decoded. Shape: (batch_size, latent_dim_A)

    Returns:
        torch.Tensor: Decoded sequence of positions in range (-1, 1). Shape: (batch_size, seq_length, output_size)
    """

    def __init__(self, latent_dim_A=64, output_size=3, seq_length=None):
        super(BondVectorDecoder, self).__init__()
        self.seq_length = seq_length
        self.gru = nn.GRU(
            input_size=output_size, hidden_size=latent_dim_A, batch_first=True
        )
        self.fc = nn.Linear(latent_dim_A, output_size)

    def forward(self, latent_A):
        batch_size = latent_A.size(0)
        h_0 = latent_A.unsqueeze(0)
        inputs = torch.zeros(batch_size, self.seq_length, 3).to(latent_A.device)
        output, _ = self.gru(inputs, h_0.contiguous())
        output = self.fc(output)
        output = torch.tanh(output)  # Constrain output to (-1, 1)
        return output


class TemporalEncoder(nn.Module):
    """Temporal encoder that processes sequences of latent vectors into a single latent representation.

    This module takes a sequence of latent vectors (latent space A) that were generated from
    position encodings over multiple frames, maps them to a higher-dimensional space using a
    linear layer, and encodes them into a single latent vector (latent space B) using a GRU network.
    The temporal relationships between the sequential latent vectors are preserved in the final encoding.

    Args:
        input_size (int, optional): Dimensionality of input latent vectors (latent space A).
            Defaults to 64.
        hidden_size (int, optional): Dimensionality of the hidden representation after the linear layer.
            Defaults to 128.
        latent_dim_B (int, optional): Dimensionality of output latent vector (latent space B).
            Defaults to 8.

    Parameters:
        fc_in (nn.Linear): Linear layer mapping input_size to hidden_size.
        gru (nn.GRU): Gated Recurrent Unit layer that processes the sequence.

    Input:
        latent_A_seq (torch.Tensor): Sequence of latent vectors from space A.
            Shape: (batch_size, sequence_length, input_size)

    Returns:
        torch.Tensor: Single latent vector in space B encoding the temporal information.
            Shape: (batch_size, latent_dim_B)
    """

    def __init__(self, input_size=64, hidden_size=128, latent_dim_B=8):
        super(TemporalEncoder, self).__init__()
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size, hidden_size=latent_dim_B, batch_first=True
        )

    def forward(self, latent_A_seq):
        x = self.fc_in(latent_A_seq)
        _, h_n = self.gru(x)
        latent_B = h_n.squeeze(0)
        return latent_B


class TemporalDecoder(nn.Module):
    """
    Temporal decoder that reconstructs latent sequences from a latent vector.

    Takes a latent vector created by TemporalEncoder and uses it to reconstruct a sequence
    of latent vectors in space A. The decoder uses a GRU (Gated Recurrent Unit) architecture
    to progressively generate latent vectors conditioned on the input latent representation.

    Args:
        latent_dim_B (int, optional): Dimension of the GRU hidden state. Defaults to 8.
        output_size (int, optional): Dimension of the output vectors at each timestep.
            Should match latent_dim_A. Defaults to 64.
        seq_length (int, optional): Length of the sequence to generate. Must be specified.
            Defaults to None.

    Attributes:
        seq_length (int): Stored sequence length for generation
        gru (nn.GRU): GRU layer that processes the sequence
        fc (nn.Linear): Final linear layer to project to output dimension

    Forward Args:
        latent_B (torch.Tensor): Latent vector encoding of shape [batch_size, latent_dim_B]
            created by the temporal encoder.

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
        h_0 = latent_B.unsqueeze(0)
        inputs = torch.zeros(batch_size, self.seq_length, self.fc.out_features).to(
            latent_B.device
        )
        output, _ = self.gru(inputs, h_0)
        output = self.fc(output)
        return output


class AuTiSM_Model(nn.Module):
    """
    AuTiSM (Autoencoded Time Series Mapping) Model for processing sequential particle data.

    This model implements a hierarchical autoencoder architecture that processes particle position data
    across multiple frames. AuTiSM stands for Autoencoded Time Series Mapping, designed to learn
    compact representations of particle dynamics over time.

    Architecture:
        - Frame-level encoding/decoding: Processes individual frames of particle positions
        - Sequence-level encoding/decoding: Handles temporal relationships between frames

    Args:
        N (int): Number of particles in each frame
        M (int): Number of frames in the sequence
        latent_dim_A (int, optional): Dimensionality of latent space A. Defaults to 64.
        latent_dim_B (int, optional): Dimensionality of latent space B. Defaults to 8.

    Input Shape:
        x: Tensor of shape (batch_size, M, N, feature_dim) containing particle positions
        where:
            - batch_size: Number of sequences in the batch
            - M: Number of frames per sequence
            - N: Number of particles per frame
            - feature_dim: Dimensionality of particle features

    Returns:
        Tensor of same shape as input, containing reconstructed particle positions
        across all frames in the sequence

    Components:
        - frame_encoder: Encodes individual frames to latent space A
        - frame_decoder: Decodes latent representations back to particle positions
        - sequence_encoder: Encodes temporal sequence of frame latents into latent space B
        - sequence_decoder: Decodes temporal latent back to frame latents
    """

    def __init__(self, N, M, latent_dim_A=64, latent_dim_B=8):
        super(AuTiSM_Model, self).__init__()
        self.N = N  # Number of particles
        self.M = M  # Number of frames
        self.frame_encoder = BondVectorEncoder(latent_dim_A=latent_dim_A)
        self.frame_decoder = BondVectorDecoder(latent_dim_A=latent_dim_A, seq_length=N)
        self.sequence_encoder = TemporalEncoder(
            input_size=latent_dim_A, latent_dim_B=latent_dim_B
        )
        self.sequence_decoder = TemporalDecoder(
            latent_dim_B=latent_dim_B, output_size=latent_dim_A, seq_length=M
        )

    def forward(self, x):
        batch_size = x.size(0)
        # Reshape x to combine batch and frame dimensions for parallel processing
        x_reshaped = x.view(-1, self.N, 3)  # Shape: (batch_size * M, N, 3)

        # Encode all frames simultaneously
        latent_A = self.frame_encoder(x_reshaped)  # Shape: (batch_size * M, latent_dim_A)

        # Reshape back to (batch_size, M, latent_dim_A)
        latent_A_seq = latent_A.view(batch_size, self.M, -1)

        # Encode the sequence of latent vectors
        latent_B = self.sequence_encoder(latent_A_seq)  # Shape: (batch_size, latent_dim_B)

        # Decode the sequence from the latent vector
        reconstructed_latent_A_seq = self.sequence_decoder(latent_B)  # Shape: (batch_size, M, latent_dim_A)

        # Reshape to (batch_size * M, latent_dim_A) for frame decoding
        reconstructed_latent_A = reconstructed_latent_A_seq.view(-1, reconstructed_latent_A_seq.size(-1))

        # Decode all frames simultaneously
        reconstructed_frames = self.frame_decoder(reconstructed_latent_A)  # Shape: (batch_size * M, N, 3)

        # Reshape back to (batch_size, M, N, 3)
        reconstructed_x = reconstructed_frames.view(batch_size, self.M, self.N, 3)

        return reconstructed_x, latent_B
