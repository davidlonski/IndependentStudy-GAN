from train import DCGANTrainer
from multiprocessing import freeze_support
import sys
import os
import psutil
import gc
import time
from dotenv import load_dotenv

load_dotenv()

# Add the common module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
from utils import send_dcgan_completion_email


def monitor_system_resources():
    """Monitor system resources to identify bottlenecks"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    print(f"System Resources:")
    print(f"  CPU Usage: {cpu_percent}%")
    print(f"  Memory Usage: {memory.percent}% ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")
    print(f"  Disk Usage: {disk.percent}% ({disk.used/1024**3:.1f}GB / {disk.total/1024**3:.1f}GB)")
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'disk_percent': disk.percent
    }


def cleanup_resources():
    """Clean up resources to prevent memory leaks"""
    gc.collect()
    if hasattr(torch, 'cuda'):
        torch.cuda.empty_cache()


if __name__ == '__main__':
    freeze_support()

    # Improved test configuration based on analysis
    print("\n--- Running Improved DCGAN Test ---\n")
    
    # Monitor initial system state
    print("🔍 Initial System State:")
    initial_resources = monitor_system_resources()
    
    # Improved training configuration
    training_config = {
        'manual_seed': 42,
        'workers': 8,  # Reduced from 24 to prevent resource contention
        'batch_size': 8,  # Reduced from 16 to improve stability
        'image_size': 256,
        'nc': 3,
        'nz': 100,
        'ngf': 64,
        'ndf': 64,
        'num_epochs': 150,  # Same as original for comparison
        'lr': 0.0002,  # Slightly higher learning rate for better convergence
        'beta1': 0.5,
        'ngpu': 1
    }
    
    print(f"📋 Improved Training Configuration:")
    for key, value in training_config.items():
        print(f"   {key}: {value}")
    
    improved_trainer = DCGANTrainer(
        **training_config,
        input_dirs=None,
        output_dir="content/TRAIN_IMPROVED"
    )
    
    try:
        # Preprocess data with monitoring
        print("\n🔄 Preprocessing data...")
        improved_trainer.preprocess_data()
        
        # Monitor resources after preprocessing
        print("\n🔍 Post-Preprocessing System State:")
        post_preprocess_resources = monitor_system_resources()
        
        # Clean up before training
        cleanup_resources()
        
        # Train with improved monitoring
        print("\n🚀 Starting improved training...")
        improved_trainer.train()
        
        # Save results
        print("\n💾 Saving results...")
        improved_trainer.save_results()
        
        # Get training time from the trainer
        training_time = getattr(improved_trainer, 'total_training_time', None)
        
        # Final resource check
        print("\n🔍 Final System State:")
        final_resources = monitor_system_resources()
        
        # Send email notification
        print("\n--- Sending Email Notification ---\n")
        
        recipient_email = os.getenv('EMAIL_RECIPIENT')
        
        send_dcgan_completion_email(
            recipient_email=recipient_email,
            training_config=training_config,
            training_time=training_time,
            output_dir="content/TRAIN_IMPROVED"
        )
        
        print("\n--- Improved Training Complete! ---\n")
        
        # Print resource comparison
        print("📊 Resource Usage Comparison:")
        print(f"  CPU: {initial_resources['cpu_percent']:.1f}% → {final_resources['cpu_percent']:.1f}%")
        print(f"  Memory: {initial_resources['memory_percent']:.1f}% → {final_resources['memory_percent']:.1f}%")
        print(f"  Disk: {initial_resources['disk_percent']:.1f}% → {final_resources['disk_percent']:.1f}%")
    
    except KeyboardInterrupt:
        print("\n--- Training Interrupted by User ---\n")
        cleanup_resources()
        sys.exit(0)
    except Exception as e:
        print(f"\n--- Training Failed: {str(e)} ---\n")
        cleanup_resources()
        
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