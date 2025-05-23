import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from generator import Generator
from discriminator import Discriminator

class RainyImageDataset(Dataset):
    def __init__(self, rainy_dir, clean_dir, transform=None):
        self.rainy_dir = rainy_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.rainy_images = sorted(os.listdir(rainy_dir))
        self.clean_images = sorted(os.listdir(clean_dir))
        
    def __len__(self):
        return len(self.rainy_images)
    
    def __getitem__(self, idx):
        rainy_img = Image.open(os.path.join(self.rainy_dir, self.rainy_images[idx])).convert('RGB')
        clean_img = Image.open(os.path.join(self.clean_dir, self.clean_images[idx])).convert('RGB')
        
        if self.transform:
            rainy_img = self.transform(rainy_img)
            clean_img = self.transform(clean_img)
            
        return rainy_img, clean_img

class IDCGANTrainer:
    def __init__(self, 
                 manual_seed=999,
                 workers=4,
                 batch_size=32,
                 image_size=64,
                 nc=3,
                 nz=100,
                 ngf=64,
                 ndf=64,
                 num_epochs=100,
                 lr=0.0002,
                 beta1=0.5,
                 ngpu=1,
                 rainy_dir=None,
                 clean_dir=None,
                 output_dir="content/TRAIN"):
        """
        Initialize the ID-CGAN trainer with configuration parameters
        
        Args:
            manual_seed (int): Random seed for reproducibility
            workers (int): Number of workers for dataloader
            batch_size (int): Batch size during training
            image_size (int): Size of training images
            nc (int): Number of channels in training images
            nz (int): Size of latent vector
            ngf (int): Size of feature maps in generator
            ndf (int): Size of feature maps in discriminator
            num_epochs (int): Number of training epochs
            lr (float): Learning rate for optimizers
            beta1 (float): Beta1 hyperparameter for Adam optimizers
            ngpu (int): Number of GPUs to use
            rainy_dir (str): Directory containing rainy images
            clean_dir (str): Directory containing clean images
            output_dir (str): Output directory for processed data
        """
        # Set random seed
        self.manual_seed = manual_seed
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        torch.use_deterministic_algorithms(True)
        
        # Training parameters
        self.workers = workers
        self.batch_size = batch_size
        self.image_size = image_size
        self.nc = nc
        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf
        self.num_epochs = num_epochs
        self.lr = lr
        self.beta1 = beta1
        self.ngpu = ngpu
        
        # Directories
        self.rainy_dir = rainy_dir or "../../content/rainy_images"
        self.clean_dir = clean_dir or "../../content/clean_images"
        self.output_dir = output_dir
        
        # Initialize device
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        
        # Initialize models and optimizers
        self.netG = None
        self.netD = None
        self.optimizerG = None
        self.optimizerD = None
        
        # Training history
        self.G_losses = []
        self.D_losses = []
        self.img_list = []
        
    def initialize_models(self):
        """Initialize generator and discriminator models"""
        # Create the Discriminator
        self.netD = Discriminator(self.ngpu, self.ndf, self.nc, image_size=self.image_size).to(self.device)
        
        # Handle multi-GPU if desired
        if (self.device.type == 'cuda') and (self.ngpu > 1):
            self.netD = nn.DataParallel(self.netD, list(range(self.ngpu)))
            
        # Apply weights initialization
        self.netD.apply(self._weights_init)
        
        # Create the generator
        self.netG = Generator(self.ngpu, self.nz, self.ngf, self.nc, image_size=self.image_size).to(self.device)
        
        # Handle multi-GPU if desired
        if (self.device.type == 'cuda') and (self.ngpu > 1):
            self.netG = nn.DataParallel(self.netG, list(range(self.ngpu)))
            
        # Apply weights initialization
        self.netG.apply(self._weights_init)
        
        # Initialize optimizers
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        
    def _weights_init(self, m):
        """Initialize network weights"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
            
    def create_dataloader(self):
        """Create and return the dataloader"""
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        dataset = RainyImageDataset(
            rainy_dir=self.rainy_dir,
            clean_dir=self.clean_dir,
            transform=transform
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True
        )
        
    def train(self):
        """Main training loop"""
        # Initialize models if not already done
        if self.netG is None or self.netD is None:
            self.initialize_models()
            
        # Create dataloader
        dataloader = self.create_dataloader()
        
        # Initialize BCELoss
        criterion = nn.BCELoss()
        
        # Enable automatic mixed precision training
        scaler = torch.amp.GradScaler()
        
        print("Starting Training Loop...")
        iters = 0
        
        # For each epoch
        for epoch in range(self.num_epochs):
            # For each batch in the dataloader
            for i, (rainy_images, clean_images) in enumerate(dataloader, 0):
                # Format batch
                rainy_images = rainy_images.to(self.device)
                clean_images = clean_images.to(self.device)
                b_size = rainy_images.size(0)
                
                # Generate random noise
                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                self.netD.zero_grad()
                
                # Train with real pairs
                label_real = torch.full((b_size,), 1.0, device=self.device)
                output_real = self.netD(rainy_images, clean_images).view(-1)
                errD_real = criterion(output_real, label_real)
                
                # Train with fake pairs
                label_fake = torch.full((b_size,), 0.0, device=self.device)
                fake_images = self.netG(rainy_images, noise)
                output_fake = self.netD(rainy_images, fake_images.detach()).view(-1)
                errD_fake = criterion(output_fake, label_fake)
                
                # Calculate discriminator loss
                errD = errD_real + errD_fake
                
                # Update discriminator
                scaler.scale(errD).backward()
                scaler.step(self.optimizerD)
                scaler.update()
                
                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                
                # Generate new fake images
                fake_images = self.netG(rainy_images, noise)
                output = self.netD(rainy_images, fake_images).view(-1)
                
                # Calculate generator loss
                errG = criterion(output, label_real)
                
                # Update generator
                scaler.scale(errG).backward()
                scaler.step(self.optimizerG)
                scaler.update()
                
                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                          % (epoch, self.num_epochs, i, len(dataloader),
                             errD.item(), errG.item()))
                
                # Save Losses for plotting later
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())
                
                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == self.num_epochs-1) and (i == len(dataloader)-1)):
                    with torch.no_grad():
                        fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)
                        fake = self.netG(rainy_images[:64], fixed_noise).detach().cpu()
                    self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                
                iters += 1
                
    def plot_training_results(self):
        """Plot the training results"""
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses, label="G")
        plt.plot(self.D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        
        # Plot the fake images from the last epoch
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Generated Clean Images")
        plt.imshow(np.transpose(self.img_list[-1],(1,2,0)))
        plt.show()
        
    def save_results(self):
        """Save the generated images and model"""
        # Create directory for saving images if it doesn't exist
        os.makedirs('../../generated_images/IDCGAN', exist_ok=True)
        
        # Save the final grid of fake images
        plt.figure(figsize=(15,15))
        plt.axis("off")
        plt.title("Final Generated Clean Images Grid")
        grid_img = np.transpose(self.img_list[-1],(1,2,0))
        plt.imshow(grid_img)
        plt.savefig('../../generated_images/IDCGAN/final_grid.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Save the trained model
        torch.save({
            'generator_state_dict': self.netG.state_dict(),
            'discriminator_state_dict': self.netD.state_dict(),
            'optimizerG_state_dict': self.optimizerG.state_dict(),
            'optimizerD_state_dict': self.optimizerD.state_dict(),
        }, '../../models/IDCGAN/gan_model.pth')

if __name__ == '__main__':
    # Create trainer instance
    trainer = IDCGANTrainer()
    
    # Train the model
    trainer.train()
    
    # Plot and save results
    trainer.plot_training_results()
    trainer.save_results() 