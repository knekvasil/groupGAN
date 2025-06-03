# project_models/discriminator

"""
This module defines the Discriminator model for a Generative Adversarial Network (GAN)
or a similar setup where a model needs to classify sequences (like text) as real or fake.

The Discriminator takes a sequence of token IDs as input, embeds them, flattens the
embeddings, and passes them through fully connected layers with a Sigmoid output
to produce a single probability score indicating the likelihood of the input
sequence being "real".
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    A simple feed-forward Discriminator model for classifying sequences.

    The model embeds input token sequences, flattens the embeddings, and uses
    dense layers to output a score for the input being real (no sigmoid for WGAN-GP).

    Attributes:
        sequence_length (int): The fixed length of the input sequences.
        embedding (nn.Embedding): Embedding layer for token IDs.
        fc1 (nn.Linear): First fully connected layer.
        relu (nn.ReLU): ReLU activation function.
        fc2 (nn.Linear): Second fully connected layer (output layer).
        # Sigmoid removed for WGAN-GP
    """

    def __init__(
        self, vocab_size, embedding_dim=128, hidden_dim=128, sequence_length=20
    ):
        """
        Initializes the Discriminator model.

        Args:
            vocab_size (int): The size of the vocabulary (number of unique tokens).
            embedding_dim (int): The dimension of the token embeddings. Defaults to 128.
            hidden_dim (int): The dimension of the hidden layer in the feed-forward
                              network. Defaults to 128.
            sequence_length (int): The expected length of the input sequences.
                                   This is used to calculate the input size for
                                   the first fully connected layer after flattening
                                   the embeddings. Defaults to 20.
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.embedding_dim = (
            embedding_dim  # Store for potential use in forward_from_embedding
        )
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # The input size for fc1 is based on the flattened embeddings
        self.fc1 = nn.Linear(embedding_dim * sequence_length, hidden_dim)
        self.relu = nn.ReLU()
        # Final layer outputs a single score, NOT a probability (no sigmoid here for WGAN-GP)
        self.fc2 = nn.Linear(hidden_dim, 1)
        # self.sigmoid = nn.Sigmoid() # REMOVED for WGAN-GP

    def forward(self, x: torch.Tensor):
        """
        Performs the forward pass of the Discriminator.
        This method expects discrete token IDs (torch.long).

        Args:
            x (torch.Tensor): Input tensor containing sequences of token IDs.
                              Expected shape is (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output tensor containing the score of each input
                          sequence being real. Shape is (batch_size, 1).
        """
        # Ensure input is long type for embedding lookup
        if x.dtype != torch.long:
            x = x.long()

        embedded = self.embedding(
            x
        )  # Shape: (batch_size, sequence_length, embedding_dim)

        # Pass the embedded sequence to the internal forward_from_embedding method
        # This allows re-using the main logic for both direct calls and GP calls
        return self.forward_from_embedding(embedded)

    def forward_from_embedding(self, embedded: torch.Tensor):
        """
        Forward pass starting from pre-computed embeddings (e.g., for WGAN-GP).
        This method expects continuous embeddings (torch.float).

        Args:
            embedded (torch.Tensor): Embedded input of shape (batch_size, sequence_length, embedding_dim)
        Returns:
            torch.Tensor: Output scores (batch_size, 1)
        """
        # Ensure input is float type for calculations
        if embedded.dtype != torch.float:
            embedded = embedded.float()

        # Flatten embeddings
        # Ensure the view operation matches the expected input dimension for fc1
        # It should be (batch_size, sequence_length * embedding_dim)
        flat = embedded.view(embedded.size(0), -1)

        out = self.relu(self.fc1(flat))
        # No sigmoid here, output raw scores for WGAN-GP
        out = self.fc2(out)
        return out

