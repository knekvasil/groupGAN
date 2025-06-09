# scripts/train_gan.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
from pathlib import Path
import sys
import os
import numpy as np
from tokenizers import Tokenizer
import torch.autograd as autograd

sys.path.append(str(Path(__file__).parent.parent))
from project_models.generator import Generator
from project_models.discriminator import Discriminator

# --- Config ---
DATA_DIR = str(Path(__file__).parent.parent / "project_data" / "processed")
TOKENIZER_PATH = str(
    Path(__file__).parent.parent / "project_data" / "hf_tokenizer.json"
)
SEQUENCE_LENGTH = 10
BATCH_SIZE = 64
NUM_EPOCHS = 100
LATENT_DIM = 100
LEARNING_RATE = 0.0002
BETA1 = 0.5
CHECKPOINT_INTERVAL = 5
USE_GRADIENT_PENALTY = False  # Disabled unless WGAN-GP is used


def load_data():
    print(f"Loading data from {DATA_DIR}...")
    try:
        X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
        y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
        X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
        y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        raise

    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def gradient_penalty(discriminator, real_tokens, fake_tokens, device):
    batch_size = real_tokens.size(0)

    real_emb = discriminator.embedding(real_tokens)
    fake_emb = discriminator.embedding(fake_tokens)

    epsilon = torch.rand(batch_size, 1, 1, device=device)
    interpolated = epsilon * real_emb + (1 - epsilon) * fake_emb
    interpolated = interpolated.detach().requires_grad_(True)

    d_interpolated = discriminator.forward_from_embedding(interpolated)

    ones = torch.ones_like(d_interpolated, device=device)

    gradients = autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def train_gan(
    generator: Generator,
    discriminator: Discriminator,
    train_loader: DataLoader,
    num_epochs: int,
    latent_dim: int,
    device: torch.device,
    lr: float = 0.0001,
    beta1: float = 0.5,
    use_wandb: bool = True,
    gp_lambda: float = 10.0,
    early_stopping_patience: int = 10,
):
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.9))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.9))

    best_g_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        d_losses, g_losses = [], []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (real_tokens, _) in enumerate(progress_bar):
            real_tokens = real_tokens.to(device)
            batch_size = real_tokens.size(0)

            # === Train Discriminator ===
            for _ in range(5):  # multiple D updates per G update
                noise = torch.randn(batch_size, latent_dim, device=device)
                fake_tokens = generator.generate(noise).detach()

                d_optimizer.zero_grad()
                d_real = discriminator(real_tokens)
                d_fake = discriminator(fake_tokens)

                gp = gradient_penalty(discriminator, real_tokens, fake_tokens, device)
                d_loss = d_fake.mean() - d_real.mean() + gp_lambda * gp
                d_loss.backward()
                d_optimizer.step()

            # === Train Generator ===
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_probs = generator(noise, temperature=0.5, hard=True)
            fake_tokens = fake_probs.argmax(dim=-1)
            g_optimizer.zero_grad()

            d_fake = discriminator(fake_tokens)
            g_loss = -d_fake.mean()
            g_loss.backward()
            g_optimizer.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

            progress_bar.set_postfix(
                {"D_loss": f"{d_loss.item():.4f}", "G_loss": f"{g_loss.item():.4f}"}
            )

            if use_wandb and batch_idx % 100 == 0:
                wandb.log(
                    {
                        "d_loss": d_loss.item(),
                        "g_loss": g_loss.item(),
                        "epoch": epoch,
                        "batch": batch_idx,
                    }
                )

        avg_g_loss = sum(g_losses) / len(g_losses)
        avg_d_loss = sum(d_losses) / len(d_losses)

        print(
            f"\nEpoch {epoch+1}: G_loss = {avg_g_loss:.4f}, D_loss = {avg_d_loss:.4f}"
        )

        # Save best generator
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(generator.state_dict(), save_dir / "generator_best.pt")
            torch.save(discriminator.state_dict(), save_dir / "discriminator_best.pt")
            print("Saved new best models.")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience {patience_counter}/{early_stopping_patience}"
            )

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

        # Optional periodic saving
        if (epoch + 1) % 5 == 0:
            torch.save(
                generator.state_dict(), save_dir / f"generator_epoch_{epoch+1}.pt"
            )
            torch.save(
                discriminator.state_dict(),
                save_dir / f"discriminator_epoch_{epoch+1}.pt",
            )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        vocab_size = tokenizer.get_vocab_size()
    except Exception as e:
        print(f"Tokenizer loading failed: {e}")
        sys.exit(1)

    try:
        train_loader, val_loader = load_data()
    except FileNotFoundError:
        sys.exit(1)

    generator = Generator(
        latent_dim=LATENT_DIM, vocab_size=vocab_size, sequence_length=SEQUENCE_LENGTH
    ).to(device)

    discriminator = Discriminator(
        vocab_size=vocab_size, sequence_length=SEQUENCE_LENGTH
    ).to(device)

    wandb.init(
        project="group-gan",
        config={
            "architecture": "GAN",
            "dataset": "titles",
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "latent_dim": LATENT_DIM,
            "learning_rate": LEARNING_RATE,
        },
    )

    train_gan(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        num_epochs=NUM_EPOCHS,
        latent_dim=LATENT_DIM,
        device=device,
        lr=LEARNING_RATE,
        beta1=BETA1,
        use_wandb=True,
        gp_lambda=10.0,
        early_stopping_patience=10,
    )

    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    torch.save(generator.state_dict(), save_dir / "generator_final.pt")
    torch.save(discriminator.state_dict(), save_dir / "discriminator_final.pt")
    wandb.finish()


if __name__ == "__main__":
    main()
