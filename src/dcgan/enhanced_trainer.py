import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

class EnhancedDCGANTrainer:
    """
    Enhanced DCGAN trainer with improved stability and monitoring
    """
    
    def __init__(self, 
                 manual_seed=42,
                 workers=8,
                 batch_size=8,
                 image_size=256,
                 nc=3,
                 nz=100,
                 ngf=64,
                 ndf=64,
                 num_epochs=150,
                 lr=0.0002,
                 beta1=0.5,
                 ngpu=1,
                 input_dirs=None,
                 output_dir="content/TRAIN_ENHANCED",
                 gradient_clip=1.0,
                 label_smoothing=0.1,
                 adaptive_lr=True):
        
        # Set random seed for reproducibility
        self.manual_seed = manual_seed
        torch.manual_seed(manual_seed)
        np.random.seed(manual_seed)
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
        self.gradient_clip = gradient_clip
        self.label_smoothing = label_smoothing
        self.adaptive_lr = adaptive_lr
        
        # Directories
        self.input_dirs = input_dirs or [
            "../../content/OCT/train/DRUSEN",
            "../../content/OCT/train/CNV",
            "../../content/OCT/train/NORMAL"
        ]
        self.output_dir = output_dir
        
        # Initialize device
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        
        # Training history with enhanced metrics
        self.G_losses = []
        self.D_losses = []
        self.G_grad_norms = []
        self.D_grad_norms = []
        self.D_real_accuracies = []
        self.D_fake_accuracies = []
        self.epoch_times = []
        self.lr_history = []
        
        # Monitoring
        self.best_g_loss = float('inf')
        self.best_d_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = 10
        
        # Track overall training time
        self.total_start_time = time.time()
        
    def preprocess_data(self):
        """Preprocess the input datasets"""
        try:
            from data_loader import DataLoader
            
            print("🔄 Starting data preprocessing...")
            data_loader = DataLoader()
            
            for input_dir in self.input_dirs:
                print(f"   Processing: {input_dir}")
                data_loader.process_dataset(input_dir, self.output_dir)
                
            print("✅ Data preprocessing completed!")
            
        except ImportError:
            print("⚠️  DataLoader not found, skipping preprocessing...")
            print("   Make sure the data is already in the output directory")
        except Exception as e:
            print(f"❌ Data preprocessing failed: {str(e)}")
            raise
        
    def initialize_models(self):
        """Initialize generator and discriminator models"""
        from generator import Generator
        from discriminator import Discriminator
        
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
        
        # Initialize optimizers with enhanced settings
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        
        # Learning rate schedulers for adaptive learning
        if self.adaptive_lr:
            self.schedulerD = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizerD, mode='min', factor=0.5, patience=5, verbose=True
            )
            self.schedulerG = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizerG, mode='min', factor=0.5, patience=5, verbose=True
            )
        
    def _weights_init(self, m):
        """Initialize network weights with improved initialization"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
            
    def create_dataloader(self):
        """Create and return the dataloader with enhanced settings"""
        import torchvision.datasets as dset
        import torchvision.transforms as transforms
        
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
            pin_memory=True,
            drop_last=True  # Ensure consistent batch sizes
        )
        
    def calculate_gradient_norm(self, model):
        """Calculate gradient norm for monitoring"""
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm
        
    def apply_label_smoothing(self, labels, smoothing=0.1):
        """Apply label smoothing for better training stability"""
        return labels * (1 - smoothing) + 0.5 * smoothing
        
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch with enhanced monitoring"""
        epoch_start_time = time.time()
        
        # Create batch of latent vectors for visualization
        fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)
        
        # Initialize BCEWithLogitsLoss
        criterion = nn.BCEWithLogitsLoss()
        
        # Enable automatic mixed precision training
        scaler = torch.amp.GradScaler()
        
        epoch_g_losses = []
        epoch_d_losses = []
        epoch_g_grads = []
        epoch_d_grads = []
        epoch_d_real_acc = []
        epoch_d_fake_acc = []
        
        # For each batch in the dataloader
        for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            self.netD.zero_grad()
            
            # Format batch
            real_cpu = data[0].to(self.device)
            b_size = real_cpu.size(0)
            
            # Apply label smoothing
            real_labels = self.apply_label_smoothing(torch.full((b_size,), 1.0, device=self.device), self.label_smoothing)
            fake_labels = self.apply_label_smoothing(torch.full((b_size,), 0.0, device=self.device), self.label_smoothing)
            
            # Forward pass real batch through D
            with torch.amp.autocast('cuda'):
                output_real = self.netD(real_cpu).view(-1)
                errD_real = criterion(output_real, real_labels)
            
            # Calculate gradients for D in backward pass
            scaler.scale(errD_real).backward()
            D_x = torch.sigmoid(output_real).mean().item()
            
            # Train with all-fake batch
            noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
            
            # Generate fake image batch with G
            with torch.amp.autocast('cuda'):
                fake = self.netG(noise)
                output_fake = self.netD(fake.detach()).view(-1)
                errD_fake = criterion(output_fake, fake_labels)
            
            # Calculate the gradients for all-fake batch
            scaler.scale(errD_fake).backward()
            D_G_z1 = torch.sigmoid(output_fake).mean().item()
            
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            
            # Gradient clipping for discriminator
            if self.gradient_clip > 0:
                scaler.unscale_(self.optimizerD)
                torch.nn.utils.clip_grad_norm_(self.netD.parameters(), self.gradient_clip)
            
            # Update D
            scaler.step(self.optimizerD)
            scaler.update()
            
            # Calculate discriminator gradient norm
            d_grad_norm = self.calculate_gradient_norm(self.netD)
            
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.netG.zero_grad()
            
            # Since we just updated D, perform another forward pass of all-fake batch through D
            with torch.amp.autocast('cuda'):
                output = self.netD(fake).view(-1)
                errG = criterion(output, real_labels)  # Use real labels for generator
            
            # Calculate gradients for G
            scaler.scale(errG).backward()
            D_G_z2 = torch.sigmoid(output).mean().item()
            
            # Gradient clipping for generator
            if self.gradient_clip > 0:
                scaler.unscale_(self.optimizerG)
                torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.gradient_clip)
            
            # Update G
            scaler.step(self.optimizerG)
            scaler.update()
            
            # Calculate generator gradient norm
            g_grad_norm = self.calculate_gradient_norm(self.netG)
            
            # Calculate accuracies
            d_real_acc = (torch.sigmoid(output_real) > 0.5).float().mean().item()
            d_fake_acc = (torch.sigmoid(output_fake) < 0.5).float().mean().item()
            
            # Store metrics
            epoch_g_losses.append(errG.item())
            epoch_d_losses.append(errD.item())
            epoch_g_grads.append(g_grad_norm)
            epoch_d_grads.append(d_grad_norm)
            epoch_d_real_acc.append(d_real_acc)
            epoch_d_fake_acc.append(d_fake_acc)
            
            # Output training stats
            if i % 100 == 0:
                tqdm.write(f'[{epoch}/{self.num_epochs}][{i}/{len(dataloader)}] '
                          f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                          f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f} '
                          f'G_grad: {g_grad_norm:.4f} D_grad: {d_grad_norm:.4f}')
        
        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start_time
        avg_g_loss = np.mean(epoch_g_losses)
        avg_d_loss = np.mean(epoch_d_losses)
        avg_g_grad = np.mean(epoch_g_grads)
        avg_d_grad = np.mean(epoch_d_grads)
        avg_d_real_acc = np.mean(epoch_d_real_acc)
        avg_d_fake_acc = np.mean(epoch_d_fake_acc)
        
        # Update learning rates if adaptive
        if self.adaptive_lr:
            self.schedulerD.step(avg_d_loss)
            self.schedulerG.step(avg_g_loss)
            current_lr_d = self.optimizerD.param_groups[0]['lr']
            current_lr_g = self.optimizerG.param_groups[0]['lr']
            self.lr_history.append((current_lr_d, current_lr_g))
        
        # Store epoch metrics
        self.G_losses.extend(epoch_g_losses)
        self.D_losses.extend(epoch_d_losses)
        self.G_grad_norms.extend(epoch_g_grads)
        self.D_grad_norms.extend(epoch_d_grads)
        self.D_real_accuracies.extend(epoch_d_real_acc)
        self.D_fake_accuracies.extend(epoch_d_fake_acc)
        self.epoch_times.append(epoch_time)
        
        # Early stopping check
        if avg_g_loss < self.best_g_loss:
            self.best_g_loss = avg_g_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        print(f'Epoch {epoch} completed in {epoch_time:.2f}s - '
              f'G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}, '
              f'D_real_acc: {avg_d_real_acc:.3f}, D_fake_acc: {avg_d_fake_acc:.3f}')
        
        return avg_g_loss, avg_d_loss, epoch_time
        
    def train(self):
        """Main training loop with enhanced monitoring"""
        # Initialize models if not already done
        if not hasattr(self, 'netG') or not hasattr(self, 'netD'):
            self.initialize_models()
            
        # Create dataloader
        dataloader = self.create_dataloader()
        
        print("Starting Enhanced Training Loop...")
        
        # For each epoch
        for epoch in range(self.num_epochs):
            avg_g_loss, avg_d_loss, epoch_time = self.train_epoch(dataloader, epoch)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Calculate total training time
        self.total_training_time = time.time() - self.total_start_time
        print(f'Total training time: {self.total_training_time:.2f} seconds ({self.total_training_time/3600:.2f} hours)')
        
    def save_enhanced_results(self):
        """Save enhanced training results with additional metrics"""
        path = f'../../generated_images/DCGAN_Enhanced_{self.image_size}'
        
        # Create directory for saving images if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Create unique folder name
        if os.path.exists(path):
            i = 1
            while os.path.exists(f'{path}_{i}'):
                i += 1
            path = f'{path}_{i}'
            os.makedirs(path, exist_ok=True)
        
        # Save enhanced batch results
        batch_results = pd.DataFrame({
            'G_losses': self.G_losses,
            'D_losses': self.D_losses,
            'G_grad_norms': self.G_grad_norms,
            'D_grad_norms': self.D_grad_norms,
            'D_real_accuracies': self.D_real_accuracies,
            'D_fake_accuracies': self.D_fake_accuracies
        })
        batch_results.to_csv(f'{path}/enhanced_batch_results.csv', index=False)
        
        # Save epoch results
        epoch_results = pd.DataFrame({
            'epoch': range(len(self.epoch_times)),
            'epoch_time': self.epoch_times
        })
        epoch_results.to_csv(f'{path}/epoch_results.csv', index=False)
        
        # Save enhanced training summary
        summary = pd.DataFrame({
            'total_training_time': [self.total_training_time],
            'total_epochs': [len(self.epoch_times)],
            'total_batches': [len(self.G_losses)],
            'final_G_loss': [self.G_losses[-1]],
            'final_D_loss': [self.D_losses[-1]],
            'best_G_loss': [self.best_g_loss],
            'final_G_grad_norm': [self.G_grad_norms[-1]],
            'final_D_grad_norm': [self.D_grad_norms[-1]],
            'final_D_real_accuracy': [self.D_real_accuracies[-1]],
            'final_D_fake_accuracy': [self.D_fake_accuracies[-1]],
            'mean_epoch_time': [np.mean(self.epoch_times)],
            'std_epoch_time': [np.std(self.epoch_times)]
        })
        summary.to_csv(f'{path}/enhanced_training_summary.csv', index=False)
        
        # Save learning rate history if adaptive
        if self.adaptive_lr and self.lr_history:
            lr_df = pd.DataFrame(self.lr_history, columns=['D_lr', 'G_lr'])
            lr_df.to_csv(f'{path}/learning_rate_history.csv', index=False)
        
        print(f"Enhanced results saved to: {path}")
        return path 