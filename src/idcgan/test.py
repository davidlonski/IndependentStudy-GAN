import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from generator import Generator

def load_model(model_path, device):
    """Load the trained generator model"""
    # Initialize generator
    netG = Generator(ngpu=1, nz=100, ngf=64, nc=3, image_size=64).to(device)
    
    # Load state dict
    checkpoint = torch.load(model_path, map_location=device)
    netG.load_state_dict(checkpoint['generator_state_dict'])
    
    # Set to evaluation mode
    netG.eval()
    
    return netG

def process_image(image_path, model, device, image_size=64):
    """Process a single rainy image to remove rain"""
    # Load and transform image
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Load image
    rainy_image = Image.open(image_path).convert('RGB')
    rainy_tensor = transform(rainy_image).unsqueeze(0).to(device)
    
    # Generate random noise
    noise = torch.randn(1, 100, 1, 1, device=device)
    
    # Generate clean image
    with torch.no_grad():
        clean_tensor = model(rainy_tensor, noise)
    
    # Convert back to image
    clean_image = clean_tensor.squeeze(0).cpu()
    clean_image = clean_image * 0.5 + 0.5  # Unnormalize
    clean_image = transforms.ToPILImage()(clean_image)
    
    return rainy_image, clean_image

def main():
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model_path = "../../models/IDCGAN/gan_model.pth"
    model = load_model(model_path, device)
    
    # Create output directory
    os.makedirs("../../results/IDCGAN", exist_ok=True)
    
    # Process test images
    test_dir = "../../content/test_rainy_images"
    for image_name in os.listdir(test_dir):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            # Process image
            image_path = os.path.join(test_dir, image_name)
            rainy_image, clean_image = process_image(image_path, model, device)
            
            # Save results
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.imshow(rainy_image)
            plt.title("Rainy Image")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(clean_image)
            plt.title("Generated Clean Image")
            plt.axis('off')
            
            plt.savefig(f"../../results/IDCGAN/{os.path.splitext(image_name)[0]}_result.png",
                       bbox_inches='tight', pad_inches=0)
            plt.close()

if __name__ == "__main__":
    main()
