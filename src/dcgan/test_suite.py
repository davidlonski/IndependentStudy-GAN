from train import DCGANTrainer
from multiprocessing import freeze_support
import torch

def run_test(test_num, **kwargs):
    print(f"\n--- Starting DCGAN Test {test_num} ---\n")
    trainer = DCGANTrainer(**kwargs)
    trainer.preprocess_data()
    trainer.train()
    trainer.save_results()

if __name__ == '__main__':
    freeze_support()

    # DCGAN-optimized base parameters for 256x256 images
    base_params = dict(
        manual_seed=42,
        workers=8,  # Reduced from WGAN for DCGAN stability
        batch_size=128,  # Optimal for DCGAN stability and quality
        image_size=256,
        nc=3,
        nz=100,  # Standard for DCGAN
        ngf=64,  # Base feature maps
        ndf=64,  # Base feature maps
        num_epochs=100,
        lr=0.0002,  # DCGAN standard learning rate
        beta1=0.5,  # DCGAN standard beta1
        ngpu=1,
        input_dirs=None,
        output_dir="content/TRAIN"
    )

    # DCGAN-optimized test configurations for 256x256 images
    test_configs = [
        # Test 1: Increased Model Capacity (more feature maps)
        dict(
            ngf=128, 
            ndf=128,
            description="Increased model capacity with more feature maps"
        ),
        
        # Test 2: Higher Resolution Optimized (deeper network)
        dict(
            ngf=128, 
            ndf=128,
            nz=128,  # Larger latent space for more variety
            description="Deeper network with larger latent space"
        ),
        
        # Test 3: Stability Optimized (smaller batch, lower LR)
        dict(
            batch_size=64,
            lr=0.0001,
            description="Stability-focused with smaller batch and lower learning rate"
        ),
        
        # Test 4: Quality Optimized (larger model, longer training)
        dict(
            ngf=128,
            ndf=128,
            nz=128,
            num_epochs=150,
            lr=0.00015,
            description="Quality-focused with larger model and extended training"
        ),
        
        # Test 5: Balanced Performance (optimal DCGAN settings)
        dict(
            ngf=96,
            ndf=96,
            nz=100,
            batch_size=96,
            lr=0.0002,
            num_epochs=120,
            description="Balanced performance with optimized DCGAN parameters"
        ),
        
        # Test 6: High Quality Production (best settings)
        dict(
            ngf=128,
            ndf=128,
            nz=128,
            batch_size=64,
            lr=0.0001,
            num_epochs=200,
            description="High quality production settings with maximum training"
        ),
    ]

    # Enable cuDNN auto-tuner for optimal hardware performance
    torch.backends.cudnn.benchmark = True

    # Run each test configuration
    for i, config in enumerate(test_configs, 1):
        # Extract description for logging
        description = config.pop('description', f'Test {i}')
        
        # Merge base and test-specific parameters
        params = base_params.copy()
        params.update(config)

        print(f"\n{'='*60}")
        print(f"Running DCGAN Test {i}: {description}")
        print(f"{'='*60}")
        print(f"Parameters:")
        for key, value in params.items():
            if key not in ['input_dirs', 'output_dir']:
                print(f"  {key}: {value}")
        print(f"{'='*60}")

        # Setup mixed precision training for better performance
        scaler = torch.amp.GradScaler('cuda')

        # Run test with automatic mixed precision
        with torch.amp.autocast('cuda'):
            run_test(i, **params)

    # Restore default PyTorch behavior
    torch.use_deterministic_algorithms(False)
    
    print(f"\n{'='*60}")
    print("All DCGAN tests completed!")
    print("Check the generated_images/DCGAN_Test_256_* directories for results.")
    print(f"{'='*60}") 