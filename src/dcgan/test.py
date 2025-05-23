from train import DCGANTrainer
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    
    # Create a trainer with default parameters
    trainer = DCGANTrainer(
        manual_seed=999,
        workers=4,
        batch_size=256,
        # Image size can be 64, 128, or 256
        image_size=128,
        nc=3,
        nz=100,
        ngf=64,
        ndf=64,
        num_epochs=5,
        lr=0.0002,
        beta1=0.5,
        ngpu=1,
        input_dirs=None,
        output_dir="content/TRAIN")

    # Run the training pipeline
    trainer.preprocess_data()
    trainer.train()
    trainer.plot_training_results()
    trainer.save_results()
