# System Dependencies
from data_loader import DataLoader
from discriminator import Discriminator
from generator import Generator

# Dependencies 
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
import matplotlib.animation as animation
from IPython.display import HTML

class WGANTrainer:
    def __init__(self, 
                 manual_seed=999,
                 workers=4,
                 batch_size=256,
                 image_size=64,
                 nc=3,
                 nz=100,
                 ngf=64,
                 ndf=64,
                 num_epochs=5,
                 lr=0.0002,
                 beta1=0.5,
                 ngpu=1,
                 input_dirs=None,
                 output_dir="content/TRAIN"):
        """
        Initialize the WGAN trainer with configuration parameters
        
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
            input_dirs (list): List of input directories for dataset
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
        self.input_dirs = input_dirs or [
            "../../content/OCT/train/DRUSEN",
            "../../content/OCT/train/CNV",
            "../../content/OCT/train/NORMAL"
        ]
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
        self.optimizerD = optim.RMSprop(self.netD.parameters(), lr=self.lr)
        self.optimizerG = optim.RMSprop(self.netG.parameters(), lr=self.lr)
        
    def _weights_init(self, m):
        """Initialize network weights"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
            
    def preprocess_data(self):
        """Preprocess the input datasets"""
        data_loader = DataLoader()
        for input_dir in self.input_dirs:
            data_loader.process_dataset(input_dir, self.output_dir)
            
    def create_dataloader(self):
        """Create and return the dataloader"""
        dataset = dset.ImageFolder(
            root=self.output_dir,
            transform=transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        
        return torch.utils.data.DataLoader(
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
        
        # Create batch of latent vectors for visualization
        fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)
        
        # Number of critic iterations per generator iteration
        n_critic = 5
        
        # Clip parameter for WGAN
        clip_value = 0.01
        
        # Enable automatic mixed precision training
        scaler = torch.amp.GradScaler('cuda')
        
        print("Starting Training Loop...")
        iters = 0
        
        # For each epoch
        for epoch in range(self.num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):
                # Train discriminator
                for _ in range(n_critic):
                    self.netD.zero_grad(set_to_none=True)
                    
                    # Format batch
                    real_cpu = data[0].to(self.device, non_blocking=True)
                    b_size = real_cpu.size(0)
                    
                    # Train with real
                    with torch.amp.autocast('cuda'):
                        errD_real = -torch.mean(self.netD(real_cpu))
                    
                    scaler.scale(errD_real).backward()
                    
                    # Train with fake
                    noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                    
                    with torch.amp.autocast('cuda'):
                        fake = self.netG(noise)
                        errD_fake = torch.mean(self.netD(fake.detach()))
                        errD = errD_real + errD_fake
                    
                    scaler.scale(errD_fake).backward()
                    
                    scaler.step(self.optimizerD)
                    scaler.update()
                    
                    # Clip weights of discriminator
                    for p in self.netD.parameters():
                        p.data.clamp_(-clip_value, clip_value)
                
                # Train generator
                self.netG.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda'):
                    fake = self.netG(noise)
                    errG = -torch.mean(self.netD(fake))
                
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
                        fake = self.netG(fixed_noise).detach().cpu()
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
        plt.title("Fake Images")
        plt.imshow(np.transpose(self.img_list[-1],(1,2,0)))
        plt.show()
        
    def save_results(self):
        """Save the generated images and model"""
        # Create directory for saving images if it doesn't exist
        os.makedirs('../../generated_images/WGAN', exist_ok=True)
        
        # Save the final grid of fake images
        plt.figure(figsize=(15,15))
        plt.axis("off")
        plt.title("Final Fake Images Grid")
        grid_img = np.transpose(self.img_list[-1],(1,2,0))
        plt.imshow(grid_img)
        plt.savefig('../../generated_images/WGAN/final_grid.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Generate and save individual fake images
        print("Generating and saving individual fake images...")
        with torch.no_grad():
            # Generate a batch of fake images
            noise = torch.randn(64, self.nz, 1, 1, device=self.device)
            fake_images = self.netG(noise).detach().cpu()
            
            # Save individual images
            for idx in range(fake_images.size(0)):
                vutils.save_image(fake_images[idx], 
                                f'../../generated_images/WGAN/fake_image_{idx+1}.png',
                                normalize=True)
        
        # Save the trained model
        torch.save({
            'generator_state_dict': self.netG.state_dict(),
            'discriminator_state_dict': self.netD.state_dict(),
            'optimizerG_state_dict': self.optimizerG.state_dict(),
            'optimizerD_state_dict': self.optimizerD.state_dict(),
        }, '../../models/WGAN/gan_model.pth')

if __name__ == '__main__':
    # Create trainer instance
    trainer = WGANTrainer()
    
    # Preprocess data
    trainer.preprocess_data()
    
    # Train the model
    trainer.train()
    
    # Plot and save results
    trainer.plot_training_results()
    trainer.save_results()
