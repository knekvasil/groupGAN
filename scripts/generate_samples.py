# scripts/generate_samples.py

import torch
from pathlib import Path
import sys
import os
from tokenizers import Tokenizer

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from project_models.generator import Generator

# --- Configuration (Must match training config) ---
TOKENIZER_PATH = str(
    Path(__file__).parent.parent / "project_data" / "hf_tokenizer.json"
)
SEQUENCE_LENGTH = 20  # Must match the sequence_length used during training
LATENT_DIM = 100  # Must match the latent_dim used during training
NUM_SAMPLES_TO_GENERATE = 10  # How many text samples you want to generate

# Path to the trained generator checkpoint
GENERATOR_CHECKPOINT_PATH = "checkpoints/generator_final.pt"
# You can also load 'checkpoints/generator_best.pt' if you prefer the best performing one
# GENERATOR_CHECKPOINT_PATH = "checkpoints/generator_best.pt"


def generate_samples():
    """
    Loads the trained Generator model, generates noise, and decodes generated tokens
    into human-readable text.
    """
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load the Tokenizer ---
    print(f"Loading tokenizer from: {TOKENIZER_PATH}")
    try:
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        vocab_size = tokenizer.get_vocab_size()
        print(f"Tokenizer loaded. Vocab size: {vocab_size}")
    except FileNotFoundError:
        print(f"Error: Tokenizer not found at {TOKENIZER_PATH}. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred loading tokenizer: {e}. Exiting.")
        sys.exit(1)

    # --- Initialize the Generator model ---
    # Ensure these parameters match those used when the model was trained
    generator = Generator(
        latent_dim=LATENT_DIM, vocab_size=vocab_size, sequence_length=SEQUENCE_LENGTH
    ).to(device)
    print("Generator model initialized.")

    # --- Load the trained weights ---
    if not Path(GENERATOR_CHECKPOINT_PATH).exists():
        print(
            f"Error: Generator checkpoint not found at {GENERATOR_CHECKPOINT_PATH}. Exiting."
        )
        sys.exit(1)
    try:
        generator.load_state_dict(
            torch.load(GENERATOR_CHECKPOINT_PATH, map_location=device)
        )
        generator.eval()  # Set generator to evaluation mode
        print(f"Generator weights loaded from {GENERATOR_CHECKPOINT_PATH}.")
    except Exception as e:
        print(f"Error loading generator state dict: {e}. Exiting.")
        sys.exit(1)

    # --- Generate Samples ---
    print(f"\nGenerating {NUM_SAMPLES_TO_GENERATE} text samples...")
    # Create random noise vectors
    noise = torch.randn(NUM_SAMPLES_TO_GENERATE, LATENT_DIM, device=device)

    # Generate token IDs from the noise
    # The .generate() method handles the argmax for discrete tokens
    with torch.no_grad():  # Ensure no gradients are computed during generation
        generated_token_ids = generator.generate(noise)

    # --- Decode Generated Tokens Back to Text ---
    print("\n--- Generated Text Samples ---")

    # Get the padding token ID to remove it from decoded strings
    pad_token = "[PAD]"
    pad_token_id = (
        tokenizer.token_to_id(pad_token) if pad_token in tokenizer.get_vocab() else None
    )

    for i, token_ids_tensor in enumerate(generated_token_ids):
        # Convert tensor to Python list
        token_ids_list = token_ids_tensor.tolist()

        # Filter out padding tokens
        if pad_token_id is not None:
            filtered_token_ids = [
                token_id for token_id in token_ids_list if token_id != pad_token_id
            ]
        else:
            filtered_token_ids = token_ids_list

        # Decode the filtered token IDs
        # The tokenizer.decode() method can usually handle lists of IDs directly
        generated_text = tokenizer.decode(filtered_token_ids, skip_special_tokens=True)

        print(f"Sample {i+1}: {generated_text}")
    print("----------------------------")


if __name__ == "__main__":
    generate_samples()
