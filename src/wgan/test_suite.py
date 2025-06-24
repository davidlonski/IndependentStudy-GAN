from train import WGANTrainer
from multiprocessing import freeze_support
import torch

def run_test(test_num, **kwargs):
    print(f"\n--- Starting WGAN Test {test_num} ---\n")
    trainer = WGANTrainer(**kwargs)
    trainer.preprocess_data()
    trainer.train()
    trainer.save_results()

if __name__ == '__main__':
    freeze_support()

    base_params = dict(
        manual_seed=42,
        workers=24,
        batch_size=256,
        image_size=256,
        nc=3,
        nz=128,
        ngf=64,
        ndf=64,
        num_epochs=100,
        lr=0.0001,
        beta1=0.5,
        ngpu=1,
        input_dirs=None,
    )

    test_configs = [
        # Test 1: Increase Model Capacity
        dict(ngf=128, ndf=128),
        # Test 2: Lower Batch Size
        dict(batch_size=64),
        # Test 3: Higher Learning Rate
        dict(lr=0.0002),
        # Test 4: Increase Latent Vector Size
        dict(nz=256),
        # Test 5: More Training Epochs
        dict(num_epochs=200),
    ]

    # Enable cuDNN auto-tuner for optimal hardware performance
    torch.backends.cudnn.benchmark = True

    # Run each test configuration
    for i, config in enumerate(test_configs, 1):
        # Merge base and test-specific parameters
        params = base_params.copy()
        params.update(config)

        # Setup mixed precision training for better performance
        scaler = torch.amp.GradScaler('cuda')

        # Run test with automatic mixed precision
        with torch.amp.autocast('cuda'):
            run_test(i, **params)

    # Restore default PyTorch behavior
    torch.use_deterministic_algorithms(False)