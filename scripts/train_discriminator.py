# scripts/train_discriminator.py
"""
This script trains the Discriminator model defined in `project_models/discriminator.py`.

It loads the pre-processed and tokenized data (training and validation sets)
from numpy files, initializes the Discriminator model, defines the loss
function and optimizer, and runs the training loop. After training, it saves
the trained model's state dictionary.

The script requires the output of the `tokenize_titles.py` script.
"""

import sys
from pathlib import Path

# Add project root to Python path
# This allows importing modules from the project_models and utils directories
# assuming the script is run from the project root or the 'scripts' directory.
project_root = str(
    Path(__file__).resolve().parents[1]
)  # Goes up two levels from scripts/
if project_root not in sys.path:
    sys.path.append(project_root)


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tokenizers import Tokenizer  # Used only to get vocab size here, not for tokenizing
from project_models import Discriminator

# --- Config ---
# Directory containing the processed training and validation data files (.npy)
DATA_DIR = "../project_data/processed/"
# Path to the trained Hugging Face tokenizer file (.json)
# Used here only to load and get the vocabulary size for the model.
TOKENIZER_PATH = "../project_data/hf_tokenizer.json"
# Path where the trained model's state dictionary will be saved
MODEL_PATH = "../project_models/discriminator.pth"
# Expected sequence length (should match the tokenization script)
SEQUENCE_LENGTH = 20
# Training batch size
BATCH_SIZE = 64
# Number of training epochs
EPOCHS = 5
# Learning rate for the optimizer
LR = 1e-3


def load_data():
    """
    Loads training and validation data from numpy files and creates DataLoaders.

    It expects `X_train.npy`, `y_train.npy`, `X_val.npy`, and `y_val.npy`
    to be present in the directory specified by `DATA_DIR`.

    The data is converted into PyTorch Tensors and wrapped in TensorDataset
    and DataLoader objects. Labels `y` are unsqueezed to have shape (Batch, 1)
    which is suitable for `nn.BCELoss`.

    Returns:
        A tuple containing the training DataLoader and the validation DataLoader
        (train_loader, val_loader).

    Raises:
        FileNotFoundError: If any of the required data files are not found.
    """
    print(f"Loading data from {DATA_DIR}...")
    try:
        X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
        y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
        X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
        y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
        print("Data files loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        print(f"Please ensure data files exist in {DATA_DIR}.")
        raise e  # Re-raise the exception

    # Convert to torch tensors
    # X tensors are token IDs, need to be long
    # y tensors are labels (0 or 1), need to be float32 for BCELoss and shape (Batch, 1)
    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(
        1
    )  # Add a dimension for BCELoss
    X_val = torch.tensor(X_val, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(
        1
    )  # Add a dimension for BCELoss
    print("Data converted to PyTorch tensors.")

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )  # No shuffle for validation
    print(f"DataLoaders created with batch size {BATCH_SIZE}.")

    return train_loader, val_loader


def train(model, train_loader, val_loader, device):
    """
    Trains the discriminator model and evaluates it on the validation set each epoch.

    Args:
        model: The PyTorch model to train (an instance of Discriminator).
        train_loader: DataLoader for the training data.
        val_loader: DataLoader for the validation data.
        device: The device to run training on ('cuda' or 'cpu').
    """
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # BCELoss is used for binary classification with Sigmoid output
    criterion = nn.BCELoss()

    print(f"Starting training for {EPOCHS} epochs on device: {device}")
    for epoch in range(EPOCHS):
        model.train()  # Set model to training mode
        total_loss = 0

        # Training loop
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(X_batch)  # Forward pass
            loss = criterion(outputs, y_batch)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0
        correct = 0
        with torch.no_grad():  # Disable gradient calculation for validation
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                # Calculate accuracy
                # Thresholding probabilities at 0.5 to get predictions
                preds = (outputs > 0.5).float()
                correct += (preds == y_batch).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.4f}"
        )


def main():
    """
    Main function to set up and start the discriminator training process.

    It ensures the model save directory exists, loads the vocabulary size
    from the tokenizer file, loads the data, initializes the Discriminator
    model, determines the device (GPU or CPU), starts the training process,
    and finally saves the trained model's state dictionary.
    """
    # Ensure directory for saving the model exists
    model_dir = os.path.dirname(MODEL_PATH)
    if model_dir:  # Check if model_dir is not an empty string
        os.makedirs(model_dir, exist_ok=True)
        print(f"Ensured model save directory '{model_dir}' exists.")

    print("Loading tokenizer to get vocab size...")
    try:
        # Load the tokenizer just to get the vocabulary size needed for the embedding layer.
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        vocab_size = tokenizer.get_vocab_size()
        print(f"Tokenizer loaded. Vocab size: {vocab_size}")
    except FileNotFoundError:
        print(f"Error: Tokenizer file not found at {TOKENIZER_PATH}")
        print("Please run tokenize_titles.py first.")
        sys.exit(1)  # Exit if tokenizer is not found
    except Exception as e:
        print(f"An error occurred loading the tokenizer: {e}")
        sys.exit(1)

    # Load data using the dedicated function
    try:
        train_loader, val_loader = load_data()
    except FileNotFoundError:
        sys.exit(1)  # Exit if data loading failed due to missing files

    print("Initializing model...")
    # Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the Discriminator model with the vocabulary size
    model = Discriminator(vocab_size=vocab_size, sequence_length=SEQUENCE_LENGTH)
    model.to(device)  # Move model to the selected device
    print(f"Model initialized and moved to {device}.")
    print(f"Model architecture:\n{model}")

    print("Training discriminator...")
    # Start the training process
    train(model, train_loader, val_loader, device)

    print(f"Saving model state dictionary to {MODEL_PATH}")
    # Save only the model's learnable parameters
    torch.save(model.state_dict(), MODEL_PATH)
    print("Training complete and model saved.")


if __name__ == "__main__":
    main()
