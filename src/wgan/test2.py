from train import WGANTrainer
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    
    # Optimized trainer parameters for better 256x256 images
    trainer = WGANTrainer(
                     manual_seed=42,         # More common seed
                     workers=4,
                     batch_size=128,         # Lower for stability
                     image_size=256,
                     nc=3,
                     nz=128,                 # Higher latent dim
                     ngf=64,
                     ndf=64,
                     num_epochs=100,         # More training
                     lr=0.0001,              # Lower learning rate
                     beta1=0.5,
                     ngpu=1,
                     input_dirs=None,
                     output_dir="content/TRAIN_OPTIMIZED")

    trainer.preprocess_data()
    trainer.train()
    trainer.plot_training_results()
    trainer.save_results()