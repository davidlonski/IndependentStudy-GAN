from train import DCGANTrainer
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

    # Optimized test for stability and learning
    print("\n--- Running Optimized DCGAN Test ---\n")
    optimized_trainer = DCGANTrainer(
        manual_seed=42,
        workers=24,
        batch_size=64,
        image_size=256,
        nc=3,
        nz=100,
        ngf=64,
        ndf=64,
        num_epochs=100,
        lr=0.00005,
        beta1=0.5,
        ngpu=1,
        input_dirs=None,
        output_dir="content/TRAIN")
    optimized_trainer.preprocess_data()
    optimized_trainer.train()
    optimized_trainer.save_results()