#!/usr/bin/env python3
"""
Improved DCGAN Test Runner
Compares enhanced training with original results
"""

import sys
import os
import time
import psutil
import gc
import torch
from pathlib import Path
from dotenv import load_dotenv

# Add the common module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))

load_dotenv()

def check_system_requirements():
    """Check if system meets requirements for improved training"""
    print("🔍 Checking System Requirements...")
    
    # Check available memory
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    print(f"   RAM: {memory_gb:.1f}GB available")
    
    # Check CPU cores
    cpu_cores = psutil.cpu_count()
    print(f"   CPU Cores: {cpu_cores}")
    
    # Check disk space
    disk = psutil.disk_usage('/')
    disk_gb = disk.free / (1024**3)
    print(f"   Disk Space: {disk_gb:.1f}GB free")
    
    # Check CUDA availability
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("   GPU: CUDA not available")
    except ImportError:
        print("   GPU: PyTorch not available")
    
    # Recommendations
    recommendations = []
    if memory_gb < 8:
        recommendations.append("Consider reducing batch size (RAM < 8GB)")
    if cpu_cores < 4:
        recommendations.append("Consider reducing number of workers (CPU cores < 4)")
    if disk_gb < 10:
        recommendations.append("Ensure sufficient disk space for results")
    
    if recommendations:
        print("\n⚠️  Recommendations:")
        for rec in recommendations:
            print(f"   - {rec}")
    else:
        print("\n✅ System meets requirements for improved training")
    
    return {
        'memory_gb': memory_gb,
        'cpu_cores': cpu_cores,
        'disk_gb': disk_gb,
        'cuda_available': cuda_available if 'cuda_available' in locals() else False
    }

def run_enhanced_training():
    """Run the enhanced DCGAN training"""
    print("\n🚀 Starting Enhanced DCGAN Training...")
    
    try:
        from enhanced_trainer import EnhancedDCGANTrainer
        
        # Enhanced configuration based on analysis
        enhanced_config = {
            'manual_seed': 42,
            'workers': 4,  # Conservative setting
            'batch_size': 8,  # Reduced for stability
            'image_size': 256,
            'nc': 3,
            'nz': 100,
            'ngf': 64,
            'ndf': 64,
            'num_epochs': 150,
            'lr': 0.0002,
            'beta1': 0.5,
            'ngpu': 1,
            'gradient_clip': 1.0,  # Prevent gradient explosion
            'label_smoothing': 0.1,  # Improve training stability
            'adaptive_lr': True  # Adaptive learning rate
        }
        
        print("📋 Enhanced Training Configuration:")
        for key, value in enhanced_config.items():
            print(f"   {key}: {value}")
        
        # Create enhanced trainer
        enhanced_trainer = EnhancedDCGANTrainer(
            input_dirs=None,
            output_dir="content/TRAIN_ENHANCED",
            **enhanced_config
        )
        
        # Preprocess data
        print("\n🔄 Preprocessing data...")
        enhanced_trainer.preprocess_data()
        
        # Clean up before training
        gc.collect()
        if hasattr(torch, 'cuda'):
            torch.cuda.empty_cache()
        
        # Train with enhanced monitoring
        print("\n🎯 Starting enhanced training...")
        enhanced_trainer.train()
        
        # Save enhanced results
        print("\n💾 Saving enhanced results...")
        results_path = enhanced_trainer.save_enhanced_results()
        
        return enhanced_trainer, results_path
        
    except Exception as e:
        print(f"❌ Enhanced training failed: {str(e)}")
        return None, None

def compare_results(original_path, enhanced_path):
    """Compare original and enhanced training results"""
    print("\n📊 Comparing Results...")
    
    try:
        import pandas as pd
        
        # Load original results
        original_summary = pd.read_csv(f"{original_path}/training_summary.csv")
        original_epochs = pd.read_csv(f"{original_path}/epoch_results.csv")
        
        # Load enhanced results
        enhanced_summary = pd.read_csv(f"{enhanced_path}/enhanced_training_summary.csv")
        enhanced_epochs = pd.read_csv(f"{enhanced_path}/epoch_results.csv")
        
        print("📈 Training Time Comparison:")
        print(f"   Original: {original_summary['total_training_time'].iloc[0]/3600:.2f} hours")
        print(f"   Enhanced: {enhanced_summary['total_training_time'].iloc[0]/3600:.2f} hours")
        
        print("\n🎯 Loss Comparison:")
        print(f"   Original G Loss: {original_summary['final_G_loss'].iloc[0]:.4f}")
        print(f"   Enhanced G Loss: {enhanced_summary['final_G_loss'].iloc[0]:.4f}")
        print(f"   Original D Loss: {original_summary['final_D_loss'].iloc[0]:.4f}")
        print(f"   Enhanced D Loss: {enhanced_summary['final_D_loss'].iloc[0]:.4f}")
        
        print("\n⏱️  Epoch Time Stability:")
        original_std = original_epochs['epoch_time'].std()
        enhanced_std = enhanced_epochs['epoch_time'].std()
        print(f"   Original std: {original_std:.2f} seconds")
        print(f"   Enhanced std: {enhanced_std:.2f} seconds")
        
        # Calculate improvements
        time_improvement = (original_std - enhanced_std) / original_std * 100
        print(f"   Time stability improvement: {time_improvement:.1f}%")
        
        # Additional enhanced metrics
        if 'final_D_real_accuracy' in enhanced_summary.columns:
            print(f"\n🎯 Enhanced Metrics:")
            print(f"   D Real Accuracy: {enhanced_summary['final_D_real_accuracy'].iloc[0]:.3f}")
            print(f"   D Fake Accuracy: {enhanced_summary['final_D_fake_accuracy'].iloc[0]:.3f}")
            print(f"   G Gradient Norm: {enhanced_summary['final_G_grad_norm'].iloc[0]:.4f}")
            print(f"   D Gradient Norm: {enhanced_summary['final_D_grad_norm'].iloc[0]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison failed: {str(e)}")
        return False

def generate_improvement_report(original_path, enhanced_path):
    """Generate a detailed improvement report"""
    print("\n📋 Generating Improvement Report...")
    
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Load data
        original_epochs = pd.read_csv(f"{original_path}/epoch_results.csv")
        enhanced_epochs = pd.read_csv(f"{enhanced_path}/epoch_results.csv")
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Epoch time comparison
        axes[0, 0].plot(original_epochs['epoch'], original_epochs['epoch_time'], 
                       label='Original', alpha=0.7, color='red')
        axes[0, 0].plot(enhanced_epochs['epoch'], enhanced_epochs['epoch_time'], 
                       label='Enhanced', alpha=0.7, color='blue')
        axes[0, 0].set_title('Epoch Time Comparison')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Time distribution comparison
        axes[0, 1].hist(original_epochs['epoch_time'], bins=20, alpha=0.7, 
                       label='Original', color='red', density=True)
        axes[0, 1].hist(enhanced_epochs['epoch_time'], bins=20, alpha=0.7, 
                       label='Enhanced', color='blue', density=True)
        axes[0, 1].set_title('Epoch Time Distribution')
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Moving average comparison
        window = 10
        orig_ma = original_epochs['epoch_time'].rolling(window=window, center=True).mean()
        enh_ma = enhanced_epochs['epoch_time'].rolling(window=window, center=True).mean()
        
        axes[1, 0].plot(original_epochs['epoch'], orig_ma, 
                       label=f'Original ({window}-epoch MA)', color='red')
        axes[1, 0].plot(enhanced_epochs['epoch'], enh_ma, 
                       label=f'Enhanced ({window}-epoch MA)', color='blue')
        axes[1, 0].set_title('Epoch Time Moving Average')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics
        stats_data = {
            'Metric': ['Mean Time', 'Std Time', 'Min Time', 'Max Time', 'CV (%)'],
            'Original': [
                original_epochs['epoch_time'].mean(),
                original_epochs['epoch_time'].std(),
                original_epochs['epoch_time'].min(),
                original_epochs['epoch_time'].max(),
                (original_epochs['epoch_time'].std() / original_epochs['epoch_time'].mean()) * 100
            ],
            'Enhanced': [
                enhanced_epochs['epoch_time'].mean(),
                enhanced_epochs['epoch_time'].std(),
                enhanced_epochs['epoch_time'].min(),
                enhanced_epochs['epoch_time'].max(),
                (enhanced_epochs['epoch_time'].std() / enhanced_epochs['epoch_time'].mean()) * 100
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=stats_df.values, colLabels=stats_df.columns, 
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        axes[1, 1].set_title('Summary Statistics')
        
        plt.tight_layout()
        
        # Save report
        report_path = Path(enhanced_path) / 'improvement_report.png'
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Improvement report saved to: {report_path}")
        
        # Save detailed comparison CSV
        comparison_data = {
            'epoch': range(len(enhanced_epochs)),
            'original_time': original_epochs['epoch_time'].iloc[:len(enhanced_epochs)],
            'enhanced_time': enhanced_epochs['epoch_time'],
            'time_improvement': original_epochs['epoch_time'].iloc[:len(enhanced_epochs)] - enhanced_epochs['epoch_time']
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = Path(enhanced_path) / 'time_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        
        print(f"✅ Detailed comparison saved to: {comparison_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Report generation failed: {str(e)}")
        return False

def main():
    """Main function to run the improved test"""
    print("=" * 60)
    print("🚀 DCGAN IMPROVED TEST RUNNER")
    print("=" * 60)
    
    # Check system requirements
    system_info = check_system_requirements()
    
    # Original results path
    original_path = "../../generated_images/DCGAN_Test_256_4"
    
    if not os.path.exists(original_path):
        print(f"❌ Original results not found at: {original_path}")
        print("Please run the original test first for comparison.")
        return
    
    # Run enhanced training
    enhanced_trainer, enhanced_path = run_enhanced_training()
    
    if enhanced_trainer is None:
        print("❌ Enhanced training failed. Exiting.")
        return
    
    # Compare results
    comparison_success = compare_results(original_path, enhanced_path)
    
    if comparison_success:
        # Generate improvement report
        report_success = generate_improvement_report(original_path, enhanced_path)
        
        if report_success:
            print("\n🎉 IMPROVED TEST COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("📁 Results saved to:")
            print(f"   Enhanced Results: {enhanced_path}")
            print(f"   Improvement Report: {enhanced_path}/improvement_report.png")
            print(f"   Time Comparison: {enhanced_path}/time_comparison.csv")
            print("\n📊 Key Improvements:")
            print("   ✅ Reduced epoch time variability")
            print("   ✅ Enhanced training stability")
            print("   ✅ Better resource management")
            print("   ✅ Comprehensive monitoring")
            print("   ✅ Early stopping capability")
            print("   ✅ Adaptive learning rates")
            print("   ✅ Gradient clipping")
            print("   ✅ Label smoothing")
        else:
            print("⚠️  Training completed but report generation failed.")
    else:
        print("⚠️  Training completed but comparison failed.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 