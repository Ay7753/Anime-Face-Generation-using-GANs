"""
gan_anime_faces.py
Assignment: Anime Face Generation using GANs
Framework: PyTorch
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt

# --- 1. Configuration & Hyperparameters ---
DATA_PATH = "./data/Anime_Face_Dataset" # Update this to your Kaggle data path
IMAGE_SIZE = 64
BATCH_SIZE = 32
NZ = 100        # Size of latent vector (input to generator)
NC = 3          # Number of channels (RGB)
NGF = 64        # Generator feature map size
NDF = 64        # Discriminator feature map size
NUM_EPOCHS = 5
LR = 0.0002
BETA1 = 0.5     # Beta1 hyperparam for Adam optimizers
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Data Preparation ---
def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Check if dataset exists
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please download from Kaggle.")

    dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# --- 3. Model Architecture ---

# Weight initialization recommended by the DCGAN paper
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is noise Z, going into a convolution
            nn.ConvTranspose2d(NZ, NGF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(True),
            # State size: (NGF*8) x 4 x 4
            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),
            # State size: (NGF*4) x 8 x 8
            nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),
            # State size: (NGF*2) x 16 x 16
            nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
            # State size: (NGF) x 32 x 32
            nn.ConvTranspose2d(NGF, NC, 4, 2, 1, bias=False),
            nn.Tanh()
            # Final size: (NC) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input size: (NC) x 64 x 64
            nn.Conv2d(NC, NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (NDF) x 32 x 32
            nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (NDF*2) x 16 x 16
            nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (NDF*4) x 8 x 8
            nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (NDF*8) x 4 x 4
            nn.Conv2d(NDF * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# --- 4. Training Pipeline ---

def train():
    dataloader = get_dataloader()

    print("Total images:", len(dataloader.dataset))
    print("Using device:", DEVICE)

    # Initialize models
    netG = Generator().to(DEVICE)
    netG.apply(weights_init)
    
    netD = Discriminator().to(DEVICE)
    netD.apply(weights_init)

    # Loss and Optimizers
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, NZ, 1, 1, device=DEVICE)
    
    optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

    #  ADD THIS (Loss tracking)
    G_losses = []
    D_losses = []

    print("Starting Training Loop...")
    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(dataloader, 0):

            # (1) Train Discriminator
            netD.zero_grad()
            real_cpu = data[0].to(DEVICE)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), 1.0, dtype=torch.float, device=DEVICE)

            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(b_size, NZ, 1, 1, device=DEVICE)
            fake = netG(noise)
            label.fill_(0.0)

            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()

            optimizerD.step()

            # (2) Train Generator
            netG.zero_grad()
            label.fill_(1.0)

            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()

            optimizerG.step()

            #  STORE LOSSES
            D_losses.append(errD_real.item() + errD_fake.item())
            G_losses.append(errG.item())

            # Print progress
            if i % 10 == 0:
                print(f'[{epoch}/{NUM_EPOCHS}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD_real.item()+errD_fake.item():.4f} '
                      f'Loss_G: {errG.item():.4f}')

        #  Save images (improved grid)
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            utils.save_image(fake, f"output_epoch_{epoch}.png", normalize=True, nrow=8)

    # ✅ PLOT LOSS CURVE (VERY IMPORTANT)
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.show()

    # Save models
    torch.save(netG.state_dict(), "generator.pth")
    torch.save(netD.state_dict(), "discriminator.pth")


import os

if __name__ == "__main__":
    train()