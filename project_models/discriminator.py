# project_models/discriminator.py
"""
This module defines the Discriminator model for a Generative Adversarial Network (GAN)
or a similar setup where a model needs to classify sequences (like text) as real or fake.

The Discriminator takes a sequence of token IDs as input, embeds them, flattens the
embeddings, and passes them through fully connected layers with a Sigmoid output
to produce a single probability score indicating the likelihood of the input
sequence being "real".
"""

import torch.nn as nn


class Discriminator(nn.Module):
    """
    A simple feed-forward Discriminator model for classifying sequences.

    The model embeds input token sequences, flattens the embeddings, and uses
    dense layers followed by a Sigmoid activation to output a probability
    score for the input being real.

    Attributes:
        sequence_length (int): The fixed length of the input sequences.
        embedding (nn.Embedding): Embedding layer for token IDs.
        fc1 (nn.Linear): First fully connected layer.
        relu (nn.ReLU): ReLU activation function.
        fc2 (nn.Linear): Second fully connected layer (output layer).
        sigmoid (nn.Sigmoid): Sigmoid activation to output a probability.
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
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim * sequence_length, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Performs the forward pass of the Discriminator.

        Args:
            x (torch.Tensor): Input tensor containing sequences of token IDs.
                              Expected shape is (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output tensor containing the probability of each input
                          sequence being real. Shape is (batch_size, 1).
        """
        embedded = self.embedding(
            x
        )  # Shape: (batch_size, sequence_length, embedding_dim)
        # Flatten the embedded sequences
        flat = embedded.view(
            x.size(0), -1
        )  # Shape: (batch_size, sequence_length * embedding_dim)
        out = self.relu(self.fc1(flat))  # Shape: (batch_size, hidden_dim)
        out = self.sigmoid(self.fc2(out))  # Shape: (batch_size, 1)
        return out
