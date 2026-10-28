from train import DCGANTrainer
from multiprocessing import freeze_support
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Add the common module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
from utils import send_dcgan_completion_email


if __name__ == '__main__':
    freeze_support()

    # Improved test configuration for better training stability
    print("\n--- Running Improved DCGAN Test (v6) ---\n")
    
    # Training configuration with improvements
    training_config = {
        'manual_seed': 42,
        'workers': 24,
        'batch_size': 32,  # Increased from 16 for better stability
        'image_size': 256,
        'nc': 3,
        'nz': 100,
        'ngf': 128,  # Increased from 64 for better generator capacity
        'ndf': 64,   # Kept same for discriminator
        'num_epochs': 150,  # Reduced from 200 to prevent overtraining
        'lr': 0.0002,  # Increased from 0.0001 for better convergence
        'beta1': 0.5,
        'ngpu': 1
    }
    
    improved_trainer = DCGANTrainer(
        **training_config,
        input_dirs=None,
        output_dir="content/TRAIN"
    )
    
    try:
        improved_trainer.preprocess_data()
        improved_trainer.train()
        improved_trainer.save_results()
        
        # Get training time from the trainer
        training_time = getattr(improved_trainer, 'total_training_time', None)
        
        # Send email notification
        print("\n--- Sending Email Notification ---\n")
        
        recipient_email = os.getenv('EMAIL_RECIPIENT')  # Replace with your email
        
        send_dcgan_completion_email(
            recipient_email=recipient_email,
            training_config=training_config,
            training_time=training_time,
            output_dir="content/TRAIN"
        )
        
        print("\n--- Training Complete! ---\n")
    
    except KeyboardInterrupt:
        print("\n--- Training Interrupted by User ---\n")
        # just close the program
        sys.exit(0)
    except Exception as e:
        print(f"\n--- Training Failed: {str(e)} ---\n")
        
        # Send failure notification
        try:
            send_dcgan_completion_email(
                recipient_email=recipient_email,
                training_config=training_config,
                training_time=None,
                output_dir=None
            )
        except:
            print("Failed to send failure notification email")
    