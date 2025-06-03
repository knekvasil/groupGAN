# project_models/generator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer


class Generator(nn.Module):
    """
    GRU-based Generator model for generating sequences of token IDs (titles).
    Takes a random noise vector as input and outputs probabilities over the
    vocabulary for each position in the sequence.
    """

    def __init__(
        self,
        latent_dim: int,  # Dimension of the input noise vector (e.g., 100)
        vocab_size: int,  # Number of tokens in the vocabulary (must match tokenizer)
        sequence_length: int,  # Fixed length of the output sequence (must match discriminator input)
        hidden_dim: int = 256,  # Dimension of the GRU hidden state (can be adjusted)
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim

        # 1. Project the input noise vector to the size of the GRU's hidden state
        # This projected vector can serve as the initial hidden state of the GRU.
        self.fc_project = nn.Linear(latent_dim, hidden_dim)

        # 2. GRU layer
        # The input at each step will be a learned tensor (repeated across the sequence)
        # The initial hidden state comes from the projected noise.
        # batch_first=True means input/output shape is (batch_size, seq_length, feature_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,  # Size of the input tensor at each time step
            hidden_size=hidden_dim,  # Size of the hidden state
            num_layers=1,  # Single layer GRU for simplicity
            batch_first=True,
        )

        # 3. Output layer
        # Projects the GRU's output hidden state at each time step to the vocabulary size
        # to get logits for each token position.
        self.fc_output = nn.Linear(hidden_dim, vocab_size)

        # 4. Learned input tensor for the GRU
        # Instead of feeding token embeddings (which requires sampling during forward, complex),
        # we use a constant learned tensor as input at each time step. The sequence is
        # driven by the initial hidden state (from noise) and the GRU's recurrence.
        self.learned_input = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(
        self,
        z: torch.Tensor,
        temperature: float = 1.0,  # Controls sampling sharpness (higher = more random)
        hard: bool = False,  # If True, returns one-hot vectors (straight-through)
    ) -> torch.Tensor:
        """
        Forward pass with Gumbel-Softmax sampling for differentiable training.

        Args:
            z: Input noise (batch_size, latent_dim)
            temperature: Softmax temperature (0.1 to 1.0 typical)
            hard: If True, use straight-through estimator (one-hot vectors)

        Returns:
            - During training: Sampled token probabilities with Gumbel-Softmax
            (batch_size, seq_len, vocab_size)
            - During eval: Softmax probabilities (batch_size, seq_len, vocab_size)
        """
        batch_size = z.size(0)

        # 1. Project noise to initial hidden state
        initial_hidden = self.fc_project(z).unsqueeze(0)  # (1, batch_size, hidden_dim)

        # 2. Prepare GRU input sequence (learned tensor repeated)
        rnn_input_seq = self.learned_input.repeat(batch_size, self.sequence_length, 1)

        # 3. Forward through GRU
        gru_output_seq, _ = self.gru(rnn_input_seq, initial_hidden)

        # 4. Project to vocabulary logits
        vocab_logits = self.fc_output(
            gru_output_seq
        )  # (batch_size, seq_len, vocab_size)

        # 5. Gumbel-Softmax sampling (training) or softmax (eval)
        if self.training:
            # Differentiable sampling for training
            # Using hard=False for soft probabilities during training for D's input
            # If you want hard one-hot vectors for D, you'd use hard=True here.
            # For WGAN-GP, soft outputs are often preferred for GP calculation.
            samples = F.gumbel_softmax(
                vocab_logits,
                tau=temperature,
                hard=hard,  # Use the 'hard' parameter passed to the method
                dim=-1,
            )
            return samples
        else:
            # Standard softmax for inference
            return torch.softmax(vocab_logits, dim=-1)

    # Helper method to generate actual token IDs from probabilities (for sampling/evaluation)
    def generate(self, z: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Generates a sequence of token IDs from the noise vector.
        Uses greedy sampling (argmax) on the output probabilities.
        This method is for inference/evaluation where discrete tokens are needed.

        Args:
            z: Input noise vector (batch_size, latent_dim)
            temperature: Temperature for Gumbel-Softmax if used, though argmax ignores it.
                         Kept for consistency if one wanted to sample via multinomial.

        Returns:
            Sequence of token IDs (batch_size, sequence_length)
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            # Get probabilities over the vocabulary using the forward pass
            # We want soft probabilities first, then argmax for discrete tokens
            vocab_probs = self.forward(
                z,
                temperature=temperature,
                hard=False,  # Always get soft probs for argmax
            )  # Shape: (batch_size, sequence_length, vocab_size)

            # Sample token IDs from the probabilities using greedy sampling (most probable token)
            generated_tokens = torch.argmax(
                vocab_probs, dim=2
            )  # Shape: (batch_size, sequence_length)

        return generated_tokens
