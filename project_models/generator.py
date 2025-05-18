# project_models/generator.py

import torch
import torch.nn as nn
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
        # embedding_dim: int = 128 # Optional: if Generator used embeddings internally (less common in simple GAN generators)
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generates a sequence of probabilities over the vocabulary from a noise vector.

        Args:
            z: Input noise vector (batch_size, latent_dim)

        Returns:
            Probabilities over the vocabulary for each step (batch_size, sequence_length, vocab_size)
        """
        batch_size = z.size(0)

        # 1. Project noise to create the initial hidden state for the GRU
        # Shape needed: (num_layers * directions, batch_size, hidden_size)
        # For single-layer, unidirectional GRU: (1, batch_size, hidden_dim)
        initial_hidden = self.fc_project(z).unsqueeze(0)  # (1, batch_size, hidden_dim)

        # 2. Prepare the input sequence for the GRU
        # Repeat the learned input tensor for the entire sequence length and batch size
        rnn_input_seq = self.learned_input.repeat(batch_size, self.sequence_length, 1)
        # Shape: (batch_size, sequence_length, hidden_dim)

        # 3. Pass the input sequence and initial hidden state through the GRU
        # gru_output_seq shape: (batch_size, sequence_length, hidden_dim)
        # final_hidden shape: (1, batch_size, hidden_dim) - we don't need this here
        gru_output_seq, _ = self.gru(rnn_input_seq, initial_hidden)

        # 4. Project the GRU output at each time step to vocabulary logits
        # Shape: (batch_size, sequence_length, vocab_size)
        vocab_logits = self.fc_output(gru_output_seq)

        # 5. Apply Softmax to get probabilities over the vocabulary
        # The Discriminator typically works with probabilities or sampled tokens.
        # Often, in GAN training, you return logits and use BCEWithLogitsLoss
        # in the training loop for stability, but conceptual output is probabilities.
        # Let's return probabilities here for clarity, but remember logits might be needed for training loss.
        vocab_probs = torch.softmax(vocab_logits, dim=2)

        # The Generator's forward pass typically returns these probabilities or logits.
        # Sampling (e.g., argmax or multinomial) to get discrete token IDs
        # happens OUTSIDE the forward method in the training loop or generation function.
        return vocab_probs

    # Helper method to generate actual token IDs from probabilities (for sampling/evaluation)
    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generates a sequence of token IDs from the noise vector.
        Uses greedy sampling (argmax) on the output probabilities.

        Args:
            z: Input noise vector (batch_size, latent_dim)

        Returns:
            Sequence of token IDs (batch_size, sequence_length)
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            # Get probabilities over the vocabulary
            vocab_probs = self.forward(
                z
            )  # Shape: (batch_size, sequence_length, vocab_size)

            # Sample token IDs from the probabilities using greedy sampling (most probable token)
            generated_tokens = torch.argmax(
                vocab_probs, dim=2
            )  # Shape: (batch_size, sequence_length)

        return generated_tokens


# FOR TESTING ONLY
if __name__ == "__main__":
    # --- Define Path to Tokenizer ---
    TOKENIZER_PATH = "../project_data/hf_tokenizer.json"
    # --------------------------------

    # --- Load the Tokenizer ---
    try:
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        # Get vocab size and sequence length from the tokenizer for model instantiation
        # Ensure this matches SEQUENCE_LENGTH used in tokenize_titles.py
        vocab_size = tokenizer.get_vocab_size()
        print(f"Tokenizer loaded. Vocab size: {vocab_size}")

    except FileNotFoundError:
        print(f"Error: Tokenizer not found at {TOKENIZER_PATH}. Cannot decode tokens.")
        tokenizer = None  # Set to None to handle case where tokenizer isn't found
        vocab_size = 10000  # Use a default vocab size for model instantiation
        print("Using default vocab size for model instantiation.")
    except Exception as e:
        print(f"An error occurred loading tokenizer: {e}. Cannot decode tokens.")
        tokenizer = None
        vocab_size = 10000
        print("Using default vocab size for model instantiation.")

    # --- Model and Generation Parameters ---
    latent_dim = 100  # Example
    sequence_length = 20  # Example - MUST match tokenization length
    batch_size = 32

    # Create a generator instance (use the loaded vocab_size)
    # Ensure sequence_length matches as well
    generator = Generator(
        latent_dim=latent_dim, vocab_size=vocab_size, sequence_length=sequence_length
    )

    # Create a random noise vector
    noise = torch.randn(batch_size, latent_dim)

    # Generate actual token IDs (for feeding to Discriminator or inspection)
    generated_tokens = generator.generate(noise)
    print(f"Generated tokens shape: {generated_tokens.shape}")

    # --- Decode Generated Tokens Back to Text ---
    print("\nExample generated text snippet (first sample):")
    if tokenizer is not None:  # Only decode if tokenizer was loaded successfully
        # Get the padding token ID
        # Check if '[PAD]' token exists in vocab before getting ID
        pad_token = "[PAD]"
        pad_token_id = (
            tokenizer.token_to_id(pad_token)
            if pad_token in tokenizer.get_vocab()
            else None
        )

        # Iterate through the token IDs of the first generated sequence
        token_strings = []
        # generated_tokens[0] is the first sequence tensor (shape [sequence_length])
        for token_id in generated_tokens[0].tolist():  # Convert tensor to Python list
            # Convert token_id to integer if it's not already
            token_id_int = int(token_id)

            # Skip the padding token if it exists and we know its ID
            if pad_token_id is not None and token_id_int == pad_token_id:
                continue  # Skip padding tokens

            # Get the token string from the tokenizer's vocabulary
            token_string = tokenizer.id_to_token(token_id_int)

            # Handle potential None if token_id is somehow invalid
            if token_string is not None:
                token_strings.append(token_string)
            else:
                token_strings.append("[INVALID_TOKEN]")  # Placeholder for invalid IDs

        # Join the token strings with spaces
        print(" ".join(token_strings))

    else:
        print("Tokenizer not loaded. Cannot decode token IDs to text.")

    # --- End Decode ---
