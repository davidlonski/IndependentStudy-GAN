# System Dependencies 
import preprocess
import generator
import discriminator

# Dependencies 
import argparse
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

if __name__ == '__main__':
    # Move all your existing code inside this block
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)
    
    

    # Number of workers for dataloader
    workers = 4

    # Batch size during training
    batch_size = 256

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64

    # Number of channels in the training images. For color images this is 3
    nc = 3

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Size of feature maps in generator
    ngf = 64

    # Size of feature maps in discriminator
    ndf = 64

    # Number of training epochs
    num_epochs = 5

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparameter for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    #################################################################
    #                       Preprocessing                           #
    #################################################################

    # Root directory for datasets
    input_directory1 = r"C:\Users\dlons1\Downloads\archive\OCTUCSD-3class\OCTUCSD-3class\OCT\train\DRUSEN"
    input_directory2 = r"C:\Users\dlons1\Downloads\archive\OCTUCSD-3class\OCTUCSD-3class\OCT\train\CNV"
    input_directory3 = r"C:\Users\dlons1\Downloads\archive\OCTUCSD-3class\OCTUCSD-3class\OCT\train\NORMAL"

    output_directory = "content\TRAIN"

    preprocess.process_dataset(input_directory1, output_directory)
    preprocess.process_dataset(input_directory2, output_directory)
    preprocess.process_dataset(input_directory3, output_directory)








    #################################################################
    #                       Data Loader                              #
    #################################################################

    # Data set to be used for training 
    dataroot = "./content"

    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers,
                                             pin_memory=True)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()








    #################################################################
    #                       Initialize the Models                   #
    #################################################################

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # Create the Discriminator
    netD = discriminator.Discriminator(ngpu, ndf, nc).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    # Create the generator
    netG = generator.Generator(ngpu, nz, ngf, nc).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)

    # Print the model
    print(netG)







    #################################################################
    #                       Initialize for training                 #
    #################################################################

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.RMSprop(netD.parameters(), lr=0.00005)
    optimizerG = optim.RMSprop(netG.parameters(), lr=0.00005)

    # Number of critic iterations per generator iteration
    n_critic = 5

    # Clip parameter for WGAN
    clip_value = 0.01

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    # Enable automatic mixed precision training
    scaler = torch.amp.GradScaler('cuda')

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize E[D(x)] - E[D(G(z))]
            ###########################
            
            # Train the critic more frequently
            for _ in range(n_critic):
                netD.zero_grad(set_to_none=True)
                
                # Format batch
                real_cpu = data[0].to(device, non_blocking=True)
                b_size = real_cpu.size(0)
                
                # Train with real
                with torch.amp.autocast('cuda'):
                    errD_real = -torch.mean(netD(real_cpu))
                
                scaler.scale(errD_real).backward()

                # Train with fake
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                
                with torch.amp.autocast('cuda'):
                    fake = netG(noise)
                    errD_fake = torch.mean(netD(fake.detach()))
                    errD = errD_real + errD_fake
                
                scaler.scale(errD_fake).backward()
                
                scaler.step(optimizerD)
                scaler.update()

                # Clip weights of discriminator
                for p in netD.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            ############################
            # (2) Update G network: maximize E[D(G(z))]
            ###########################
            netG.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                fake = netG(noise)
                errG = -torch.mean(netD(fake))
            
            scaler.scale(errG).backward()
            scaler.step(optimizerG)
            scaler.update()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item()))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1








    #################################################################
    #                       Plotting the training results           #
    #################################################################

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()













    #################################################################
    #                       Saving the results                      #
    #################################################################

    # Create directory for saving images if it doesn't exist
    os.makedirs('generated_images', exist_ok=True)

    # Save the final grid of fake images
    plt.figure(figsize=(15,15))
    plt.axis("off")
    plt.title("Final Fake Images Grid")
    grid_img = np.transpose(img_list[-1],(1,2,0))
    plt.imshow(grid_img)
    plt.savefig('generated_images/final_grid.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Generate and save individual fake images
    print("Generating and saving individual fake images...")
    with torch.no_grad():
        # Generate a batch of fake images
        noise = torch.randn(64, nz, 1, 1, device=device)
        fake_images = netG(noise).detach().cpu()
        
        # Save individual images
        for idx in range(fake_images.size(0)):
            vutils.save_image(fake_images[idx], 
                             f'generated_images/fake_image_{idx+1}.png',
                             normalize=True)

    print(f"Images saved in the 'generated_images' directory")

    # Save the trained model
    torch.save({
        'generator_state_dict': netG.state_dict(),
        'discriminator_state_dict': netD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
    }, 'gan_model.pth')

    print("Saving model weights and optimizer weights")
    # save model weights
    torch.save(netG.state_dict(), 'netG.pth')
    torch.save(netD.state_dict(), 'netD.pth')
    # save optimizer weights
    torch.save(optimizerG.state_dict(), 'optimizerG.pth')
    torch.save(optimizerD.state_dict(), 'optimizerD.pth')
