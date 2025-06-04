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
import time
from tqdm import tqdm
import pandas as pd

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
        """Main training loop with time tracking and progress bars"""
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
        
        # Track overall training time
        self.total_start_time = time.time()
        self.epoch_times = []
        
        # For each epoch with tqdm
        for epoch in tqdm(range(self.num_epochs), desc="Epochs"):
            # Track epoch time
            epoch_start_time = time.time()
            
            # For each batch in the dataloader with tqdm
            for i, data in enumerate(tqdm(dataloader, desc=f"Batches (Epoch {epoch})")):
                # Track batch time
                batch_start_time = time.time()
                
                # Train discriminator
                for _ in tqdm(range(n_critic), desc="Discriminator Training", leave=False):
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
                
                # Calculate batch time
                batch_time = time.time() - batch_start_time
                
                # Output training stats
                if i % 50 == 0:
                    print(f'[Epoch {epoch}/{self.num_epochs}][Batch {i}/{len(dataloader)}] '
                          f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                          f'Batch Time: {batch_time:.4f}s')
                
                # Save Losses for plotting later
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())
                
                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == self.num_epochs-1) and (i == len(dataloader)-1)):
                    with torch.no_grad():
                        fake = self.netG(fixed_noise).detach().cpu()
                    self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                
                iters += 1
            
            # Calculate and print epoch time
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            print(f'Epoch {epoch} completed in {epoch_time:.2f} seconds')
        
        # Calculate and print total training time
        self.total_training_time = time.time() - self.total_start_time
        print(f'Total training time: {self.total_training_time:.2f} seconds ({self.total_training_time/3600:.2f} hours)')
        
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
        
    def save_results(self):
        """Save the generated images and model"""
        path = f'../../generated_images/WGAN_Test_{self.image_size}'

        # Create directory for saving images if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # if the path already has content, make a new folder with a 1 at the end. if there is already a 1, make a new folder with a 2 at the end. and so on.
        if os.path.exists(path):
            i = 1
            while os.path.exists(f'{path}_{i}'):
                i += 1
            path = f'{path}_{i}'
            os.makedirs(path, exist_ok=True)

        def save_results_to_csv(self):
            """Save training results to a CSV file"""
            # Create separate DataFrames for different metrics
            # Per-batch losses
            batch_results = pd.DataFrame({
                'G_losses': self.G_losses,
                'D_losses': self.D_losses
            })
            
            # Per-epoch times
            epoch_results = pd.DataFrame({
                'epoch': range(len(self.epoch_times)),
                'epoch_time': self.epoch_times
            })
            
            # Save both DataFrames
            batch_results.to_csv(f'{path}/batch_results.csv', index=False)
            epoch_results.to_csv(f'{path}/epoch_results.csv', index=False)
            
            # Create a summary DataFrame
            summary = pd.DataFrame({
                'total_training_time': [self.total_training_time],
                'total_epochs': [len(self.epoch_times)],
                'total_batches': [len(self.G_losses)],
                'final_G_loss': [self.G_losses[-1]],
                'final_D_loss': [self.D_losses[-1]]
            })
            summary.to_csv(f'{path}/training_summary.csv', index=False)

        def save_final_grid(self):
            # Save the final grid of fake images
            plt.figure(figsize=(15,15))
            plt.axis("off")
            plt.title("Final Fake Images Grid")
            grid_img = np.transpose(self.img_list[-1],(1,2,0))
            plt.imshow(grid_img)
            plt.savefig(f'{path}/final_grid.png', bbox_inches='tight', pad_inches=0)
            plt.close()

        def save_individual_fake_images(self, number_of_images=64):
            # Generate and save individual fake images
            # Create a separate directory for individual images
            individual_images_dir = os.path.join(path, 'individual_images')
            os.makedirs(individual_images_dir, exist_ok=True)
            
            with torch.no_grad():
                # Generate a batch of fake images
                noise = torch.randn(number_of_images, self.nz, 1, 1, device=self.device)
                fake_images = self.netG(noise).detach().cpu()
                
                # Save individual images
                for idx in range(fake_images.size(0)):
                    vutils.save_image(fake_images[idx], 
                                    os.path.join(individual_images_dir, f'fake_image_{idx+1}.png'),
                                    normalize=True)

        def save_training_progress_animation(self):
            # Create and save the animation of training progress
            try:
                fig = plt.figure(figsize=(8,8))
                plt.axis("off")
                ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in self.img_list]
                ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
                
                # Save the animation
                ani.save(f'{path}/training_progress.gif', writer='pillow')
            except Exception as e:
                print(f"Error creating animation: {str(e)}")
            finally:
                plt.close(fig)

        def save_real_vs_fake_comparison(self):
            # Create a new dataloader for real images
            dataloader = self.create_dataloader()
            
            # Save image of real vs fake 
            # Grab a batch of real images from the dataloader
            real_batch = next(iter(dataloader))

            # Plot the real images
            plt.figure(figsize=(15,15))
            plt.subplot(1,2,1)
            plt.axis("off")
            plt.title("Real Images")
            real_grid = vutils.make_grid(real_batch[0].to(self.device)[:64], padding=5, normalize=True).cpu()
            plt.imshow(np.transpose(real_grid,(1,2,0)))

            # Plot the fake images from the last epoch
            plt.subplot(1,2,2)
            plt.axis("off")
            plt.title("Fake Images")
            plt.imshow(np.transpose(self.img_list[-1],(1,2,0)))
            plt.savefig(f'{path}/real_vs_fake_comparison.png', bbox_inches='tight', pad_inches=0)
            plt.close()

        def save_trained_model(self):
            # Save the trained model
            torch.save({
                'generator_state_dict': self.netG.state_dict(),
                'discriminator_state_dict': self.netD.state_dict(),
                'optimizerG_state_dict': self.optimizerG.state_dict(),
                'optimizerD_state_dict': self.optimizerD.state_dict(),
            }, f'{path}/gan_model_{self.image_size}.pth')

        save_results_to_csv(self)
        save_final_grid(self)
        save_individual_fake_images(self, 256)
        save_training_progress_animation(self)
        save_real_vs_fake_comparison(self)
        save_trained_model(self)


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
