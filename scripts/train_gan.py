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

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from project_models.generator import Generator
from project_models.discriminator import Discriminator

# --- Config ---
DATA_DIR = str(Path(__file__).parent.parent / "project_data" / "processed")
TOKENIZER_PATH = str(Path(__file__).parent.parent / "project_data" / "hf_tokenizer.json")
SEQUENCE_LENGTH = 20
BATCH_SIZE = 64
NUM_EPOCHS = 100
LATENT_DIM = 100
LEARNING_RATE = 0.0002
BETA1 = 0.5

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

def gradient_penalty(discriminator, real_data, fake_data, device):
    batch_size = real_data.size(0)
    epsilon = torch.rand(batch_size, 1, 1, device=device)
    epsilon = epsilon.expand_as(real_data)

    interpolated = (epsilon * real_data + (1 - epsilon) * fake_data).requires_grad_(True)
    interpolated_output = discriminator(interpolated)

    gradients = torch.autograd.grad(
        outputs=interpolated_output,
        inputs=interpolated,
        grad_outputs=torch.ones_like(interpolated_output),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
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
    lr: float = 0.0002,
    beta1: float = 0.5,
    use_wandb: bool = True,
):
    """
    Train the GAN model.
    
    Args:
        generator: The Generator model
        discriminator: The Discriminator model
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        latent_dim: Dimension of the latent space
        device: Device to train on (cuda/cpu)
        lr: Learning rate
        beta1: Beta1 parameter for Adam optimizer
        use_wandb: Whether to use Weights & Biases for logging
    """
    # Initialize optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Labels for real and fake data
    real_label = 1.0
    fake_label = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        # Initialize metrics
        d_losses = []
        g_losses = []
        d_real_acc = []
        d_fake_acc = []
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (real_tokens, _) in enumerate(progress_bar):
            batch_size = real_tokens.size(0)
            real_tokens = real_tokens.to(device)
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            lambda_gp = 10
            d_optimizer.zero_grad()
            
            # Train with real data
            label_real = torch.full((batch_size, 1), real_label, device=device)
            output_real = discriminator(real_tokens)
            d_loss_real = criterion(output_real, label_real)
            
            # Train with fake data
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_data = generator.generate(noise)
            label_fake = torch.full((batch_size, 1), fake_label, device=device)
            output_fake = discriminator(fake_data.detach())
            d_loss_fake = criterion(output_fake, label_fake)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # ---------------------
            # Train Generator
            # ---------------------
            g_optimizer.zero_grad()

            # Generate new fake data with Gumbel-Softmax
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_data_probs = generator(noise, temperature=0.5, hard=True)  # Differentiable sampling
            fake_tokens = fake_data_probs.argmax(dim=-1)  # Convert to token IDs
            output_fake = discriminator(fake_tokens)

            # Generator wants to fool discriminator
            g_loss = criterion(output_fake, label_real)
            g_loss.backward()
            g_optimizer.step()
                    
            # Calculate metrics
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            d_real_acc.append((output_real > 0.5).float().mean().item())
            d_fake_acc.append((output_fake < 0.5).float().mean().item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}',
                'G_loss': f'{g_loss.item():.4f}',
                'D_real_acc': f'{(output_real > 0.5).float().mean().item():.4f}',
                'D_fake_acc': f'{(output_fake < 0.5).float().mean().item():.4f}'
            })
            
            # Log to wandb
            if use_wandb and batch_idx % 100 == 0:
                wandb.log({
                    'd_loss': d_loss.item(),
                    'g_loss': g_loss.item(),
                    'd_real_acc': (output_real > 0.5).float().mean().item(),
                    'd_fake_acc': (output_fake < 0.5).float().mean().item(),
                    'epoch': epoch,
                    'batch': batch_idx
                })
        
        # Print epoch statistics
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'D_loss: {sum(d_losses)/len(d_losses):.4f}')
        print(f'G_loss: {sum(g_losses)/len(g_losses):.4f}')
        print(f'D_real_acc: {sum(d_real_acc)/len(d_real_acc):.4f}')
        print(f'D_fake_acc: {sum(d_fake_acc)/len(d_fake_acc):.4f}')
        
        # Save models periodically
        if (epoch + 1) % 5 == 0:
            save_dir = Path('checkpoints')
            save_dir.mkdir(exist_ok=True)
            torch.save(generator.state_dict(), save_dir / f'generator_epoch_{epoch+1}.pt')
            torch.save(discriminator.state_dict(), save_dir / f'discriminator_epoch_{epoch+1}.pt')

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize wandb
    """
    # TODO: Add wandb logging once all errors are fixed

    wandb.init(project="group-gan", config={
        "architecture": "GAN",
        "dataset": "titles",
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "latent_dim": LATENT_DIM,
        "learning_rate": LEARNING_RATE
    })
     """   
    # Load tokenizer to get vocab size
    print("Loading tokenizer to get vocab size...")
    try:
        # Convert to absolute path to ensure we can find the file
        tokenizer_path = Path(__file__).parent.parent / "project_data" / "hf_tokenizer.json"
        print(f"Looking for tokenizer at: {tokenizer_path}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        vocab_size = tokenizer.get_vocab_size()
        print(f"Tokenizer loaded. Vocab size: {vocab_size}")
    except FileNotFoundError:
        print(f"Error: Tokenizer file not found at {tokenizer_path}")
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
    generator = Generator(
        latent_dim=LATENT_DIM,
        vocab_size=vocab_size,
        sequence_length=SEQUENCE_LENGTH
    ).to(device)
    
    discriminator = Discriminator(
        vocab_size=vocab_size,
        sequence_length=SEQUENCE_LENGTH
    ).to(device)
    
    # Train the GAN
    train_gan(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        num_epochs=NUM_EPOCHS,
        latent_dim=LATENT_DIM,
        device=device,
        lr=LEARNING_RATE,
        beta1=BETA1,
        use_wandb=False # True
    )
    
    # Save final models
    save_dir = Path('checkpoints')
    save_dir.mkdir(exist_ok=True)
    torch.save(generator.state_dict(), save_dir / 'generator_final.pt')
    torch.save(discriminator.state_dict(), save_dir / 'discriminator_final.pt')
    
    wandb.finish()

if __name__ == '__main__':
    main()
