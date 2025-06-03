# scripts/train_gan.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # For one_hot if needed
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
from pathlib import Path
import sys
import os
import numpy as np
from tokenizers import Tokenizer
import torch.autograd as autograd

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from project_models.generator import Generator
from project_models.discriminator import Discriminator

# --- Config ---
DATA_DIR = str(Path(__file__).parent.parent / "project_data" / "processed")
TOKENIZER_PATH = str(
    Path(__file__).parent.parent / "project_data" / "hf_tokenizer.json"
)
SEQUENCE_LENGTH = 20
BATCH_SIZE = 64
NUM_EPOCHS = 50
LATENT_DIM = 100
LEARNING_RATE_G = 0.0002  # Generator learning rate
LEARNING_RATE_D = 0.00005  # Discriminator learning rate (often lower for stability)
BETA1 = 0.5  # Common beta1 for Adam in GANs
CHECKPOINT_INTERVAL = 5
GP_LAMBDA = 10.0  # Gradient Penalty coefficient
D_CRITIC_STEPS = 5  # Number of discriminator updates per generator update


def load_data():
    """
    Loads training and validation data from numpy files and creates DataLoaders.
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
        raise e

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    print("Data converted to PyTorch tensors.")

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"DataLoaders created with batch size {BATCH_SIZE}.")

    return train_loader, val_loader


def gradient_penalty(
    discriminator: Discriminator,
    real_tokens: torch.Tensor,
    fake_tokens: torch.Tensor,
    device: torch.device,
):
    """
    Calculates the gradient penalty for WGAN-GP.
    This function now correctly handles discrete token IDs by operating on their embeddings.

    Args:
        discriminator: The Discriminator (critic) model.
        real_tokens (torch.Tensor): Batch of real token IDs (batch_size, sequence_length).
        fake_tokens (torch.Tensor): Batch of fake token IDs (batch_size, sequence_length).
        device (torch.device): The device (cuda/cpu) to perform computations on.

    Returns:
        torch.Tensor: The calculated gradient penalty.
    """
    batch_size = real_tokens.size(0)

    # Get embeddings from the discriminator's embedding layer
    # Ensure real_tokens and fake_tokens are long type for embedding lookup
    real_emb = discriminator.embedding(real_tokens.long())
    fake_emb = discriminator.embedding(fake_tokens.long())

    # Create interpolated samples in the embedding space
    epsilon = torch.rand(batch_size, 1, 1, device=device)
    # Unsqueeze epsilon to match (batch_size, sequence_length, embedding_dim) for broadcasting
    epsilon = epsilon.expand_as(real_emb)

    interpolated_embeddings = epsilon * real_emb + (1 - epsilon) * fake_emb

    # Set requires_grad_ to True for interpolated_embeddings to compute gradients
    interpolated_embeddings.requires_grad_(True)

    # Pass interpolated embeddings through the rest of the discriminator (after embedding layer)
    d_interpolated = discriminator.forward_from_embedding(interpolated_embeddings)

    # Calculate gradients of discriminator output with respect to interpolated embeddings
    # `grad_outputs` should be ones_like to sum gradients for each sample
    gradients = autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated_embeddings,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,  # Needed for computing higher-order gradients (for generator training)
        retain_graph=True,  # Needed if this graph is used again (e.g., for D's main loss)
        only_inputs=True,
    )[0]

    # Flatten gradients to compute norm per sample
    gradients = gradients.view(batch_size, -1)

    # Calculate the gradient penalty: (||grad||_2 - 1)^2
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def train_gan(
    generator: Generator,
    discriminator: Discriminator,
    train_loader: DataLoader,
    num_epochs: int,
    latent_dim: int,
    device: torch.device,
    lr_g: float = LEARNING_RATE_G,  # Use specific learning rates for G and D
    lr_d: float = LEARNING_RATE_D,
    beta1: float = BETA1,
    use_wandb: bool = True,
    gp_lambda: float = GP_LAMBDA,
    early_stopping_patience: int = 10,
    d_critic_steps: int = D_CRITIC_STEPS,
):
    """
    Train the GAN model using WGAN-GP.

    Args:
        generator: The Generator model
        discriminator: The Discriminator model
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        latent_dim: Dimension of the latent space
        device: Device to train on (cuda/cpu)
        lr_g: Learning rate for the Generator
        lr_d: Learning rate for the Discriminator
        beta1: Beta1 parameter for Adam optimizer
        use_wandb: Whether to use Weights & Biases for logging
        gp_lambda: Gradient Penalty coefficient
        early_stopping_patience: Number of epochs to wait for improvement before stopping
        d_critic_steps: Number of discriminator updates per generator update
    """
    g_optimizer = optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, 0.999))

    # For WGAN-GP, track the Discriminator's core Wasserstein loss for convergence
    # We want this to decrease and ideally go towards zero (or a small negative value)
    best_wasserstein_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        d_losses_epoch, g_losses_epoch = [], []
        wasserstein_distances_epoch = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (real_tokens, _) in enumerate(progress_bar):
            real_tokens = real_tokens.to(device)
            batch_size = real_tokens.size(0)

            # === Train Discriminator ===
            # Update D multiple times for each G update (d_critic_steps)
            for _ in range(d_critic_steps):
                noise = torch.randn(batch_size, latent_dim, device=device)

                # Generator outputs hard tokens via .generate() for discriminator input
                # Detach to prevent gradients flowing to Generator during D training
                fake_tokens_d = generator.generate(noise).detach()

                d_optimizer.zero_grad()

                # Discriminator takes discrete token IDs (torch.long)
                d_real = discriminator(real_tokens)
                d_fake = discriminator(fake_tokens_d)

                # Calculate Gradient Penalty on embeddings
                gp = gradient_penalty(discriminator, real_tokens, fake_tokens_d, device)

                # WGAN-GP Discriminator Loss
                wasserstein_distance = d_fake.mean() - d_real.mean()
                d_loss = wasserstein_distance + gp_lambda * gp
                d_loss.backward()
                d_optimizer.step()

                # Record for epoch average
                d_losses_epoch.append(d_loss.item())
                wasserstein_distances_epoch.append(wasserstein_distance.item())

            # === Train Generator ===
            # Generator is updated once after d_critic_steps discriminator updates
            noise = torch.randn(batch_size, latent_dim, device=device)

            # Generator's forward pass returns soft probabilities via Gumbel-Softmax
            # We use hard=False for soft outputs for the generator's internal training
            # However, the discriminator expects hard tokens, so we sample using argmax
            # from the soft output. This is a common practice for text GANs.
            # Alternatively, if D was designed to take soft inputs, you'd pass `fake_soft_probs_g` directly.
            # Given current D, we convert to hard tokens.
            fake_soft_probs_g = generator(noise, temperature=0.5, hard=False)
            fake_tokens_g = torch.argmax(
                fake_soft_probs_g, dim=-1
            )  # Convert to hard tokens for D

            g_optimizer.zero_grad()
            d_fake_for_g = discriminator(fake_tokens_g)

            # WGAN-GP Generator Loss: Generator tries to maximize D's output for fake
            g_loss = -d_fake_for_g.mean()
            g_loss.backward()
            g_optimizer.step()

            g_losses_epoch.append(g_loss.item())

            progress_bar.set_postfix(
                {
                    "D_loss": f"{d_losses_epoch[-1]:.4f}",
                    "G_loss": f"{g_losses_epoch[-1]:.4f}",
                    "Wasserstein_D": f"{wasserstein_distances_epoch[-1]:.4f}",
                }
            )

            if use_wandb and batch_idx % 100 == 0:
                wandb.log(
                    {
                        "d_loss": d_losses_epoch[-1],
                        "g_loss": g_losses_epoch[-1],
                        "epoch": epoch,
                        "batch": batch_idx,
                        "wasserstein_distance": wasserstein_distances_epoch[-1],
                    }
                )

        avg_g_loss = sum(g_losses_epoch) / len(g_losses_epoch)
        avg_d_loss = sum(d_losses_epoch) / len(d_losses_epoch)
        avg_wasserstein_distance = sum(wasserstein_distances_epoch) / len(
            wasserstein_distances_epoch
        )

        print(
            f"\nEpoch {epoch+1}: G_loss = {avg_g_loss:.4f}, D_loss = {avg_d_loss:.4f}, Wasserstein_D = {avg_wasserstein_distance:.4f}"
        )

        # === Early Stopping Logic (Corrected for WGAN-GP) ===
        # Monitor the average core Wasserstein distance (d_fake.mean() - d_real.mean())
        # We want this value to be as small as possible (ideally converging to zero or small negative)
        if avg_wasserstein_distance < best_wasserstein_loss:
            best_wasserstein_loss = avg_wasserstein_distance
            best_epoch = epoch
            patience_counter = 0
            torch.save(generator.state_dict(), save_dir / "generator_best.pt")
            torch.save(discriminator.state_dict(), save_dir / "discriminator_best.pt")
            print("Saved new best models based on Wasserstein Distance.")
        else:
            patience_counter += 1
            print(
                f"No improvement in Wasserstein Distance. Patience {patience_counter}/{early_stopping_patience}"
            )

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

        # Optional periodic saving
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            torch.save(
                generator.state_dict(), save_dir / f"generator_epoch_{epoch+1}.pt"
            )
            torch.save(
                discriminator.state_dict(),
                save_dir / f"discriminator_epoch_{epoch+1}.pt",
            )


def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer to get vocab size
    print("Loading tokenizer to get vocab size...")
    try:
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        vocab_size = tokenizer.get_vocab_size()
        print(f"Tokenizer loaded. Vocab size: {vocab_size}")
    except FileNotFoundError:
        print(f"Error: Tokenizer file not found at {TOKENIZER_PATH}")
        print("Please ensure the tokenizer file exists in the project_data directory.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred loading the tokenizer: {e}")
        sys.exit(1)

    # Load data
    try:
        train_loader, val_loader = load_data()
    except FileNotFoundError:
        sys.exit(1)

    # Initialize models
    # Discriminator needs embedding_dim for its internal calculations
    # Assuming default embedding_dim=128 from Discriminator's __init__
    generator = Generator(
        latent_dim=LATENT_DIM, vocab_size=vocab_size, sequence_length=SEQUENCE_LENGTH
    ).to(device)

    discriminator = Discriminator(
        vocab_size=vocab_size,
        sequence_length=SEQUENCE_LENGTH,
        embedding_dim=128,  # Pass embedding_dim explicitly
    ).to(device)

    # Initialize wandb (uncomment if you want to use it)
    """
    wandb.init(project="group-gan", config={
        "architecture": "WGAN-GP",
        "dataset": "titles",
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "latent_dim": LATENT_DIM,
        "learning_rate_g": LEARNING_RATE_G,
        "learning_rate_d": LEARNING_RATE_D,
        "beta1": BETA1,
        "gp_lambda": GP_LAMBDA,
        "d_critic_steps": D_CRITIC_STEPS,
        "early_stopping_patience": 20
    })
    """

    # Train the GAN
    train_gan(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        num_epochs=NUM_EPOCHS,
        latent_dim=LATENT_DIM,
        device=device,
        lr_g=LEARNING_RATE_G,
        lr_d=LEARNING_RATE_D,
        beta1=BETA1,
        use_wandb=False,  # Set to True if wandb.init() is uncommented
        gp_lambda=GP_LAMBDA,
        early_stopping_patience=20,
        d_critic_steps=D_CRITIC_STEPS,
    )

    # Save final models
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    torch.save(generator.state_dict(), save_dir / "generator_final.pt")
    torch.save(discriminator.state_dict(), save_dir / "discriminator_final.pt")

    # wandb.finish() # Uncomment if wandb.init() is uncommented


if __name__ == "__main__":
    main()
