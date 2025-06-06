from train import WGANTrainer
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    
    for run in range(1, 6):
        print(f"\n--- Starting WGAN Test Run {run} ---\n")
        trainer = WGANTrainer(
            manual_seed=42 + run,         # Different seed for each run
            workers=4,
            batch_size=128,
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
            output_dir=f"content/TRAIN_RUN_{run}"
        )
        trainer.preprocess_data()
        trainer.train()
        trainer.plot_training_results()
        trainer.save_results() 