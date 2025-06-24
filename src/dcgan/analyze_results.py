import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

def analyze_dcgan_results():
    """Analyze and compare results from different DCGAN test runs"""
    
    # Find all DCGAN test directories
    base_path = "../../generated_images"
    dcgan_dirs = glob.glob(f"{base_path}/DCGAN_Test_256_*")
    
    if not dcgan_dirs:
        print("No DCGAN test results found!")
        return
    
    print(f"Found {len(dcgan_dirs)} DCGAN test directories:")
    for dir_path in dcgan_dirs:
        print(f"  - {os.path.basename(dir_path)}")
    
    # Collect results from each test
    results = []
    
    for dir_path in sorted(dcgan_dirs):
        test_name = os.path.basename(dir_path)
        
        # Try to read training summary
        summary_file = os.path.join(dir_path, "training_summary.csv")
        if os.path.exists(summary_file):
            summary = pd.read_csv(summary_file)
            results.append({
                'test_name': test_name,
                'total_training_time': summary['total_training_time'].iloc[0],
                'total_epochs': summary['total_epochs'].iloc[0],
                'total_batches': summary['total_batches'].iloc[0],
                'final_G_loss': summary['final_G_loss'].iloc[0],
                'final_D_loss': summary['final_D_loss'].iloc[0],
                'avg_epoch_time': summary['total_training_time'].iloc[0] / summary['total_epochs'].iloc[0]
            })
    
    if not results:
        print("No training summary files found!")
        return
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Display summary
    print(f"\n{'='*80}")
    print("DCGAN TEST RESULTS SUMMARY")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DCGAN Test Results Comparison', fontsize=16)
    
    # Plot 1: Training Time vs Final Loss
    axes[0, 0].scatter(df['total_training_time'], df['final_G_loss'], s=100, alpha=0.7)
    axes[0, 0].set_xlabel('Total Training Time (seconds)')
    axes[0, 0].set_ylabel('Final Generator Loss')
    axes[0, 0].set_title('Training Time vs Generator Loss')
    for i, test_name in enumerate(df['test_name']):
        axes[0, 0].annotate(test_name.split('_')[-1], 
                           (df['total_training_time'].iloc[i], df['final_G_loss'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 2: Epochs vs Loss
    axes[0, 1].scatter(df['total_epochs'], df['final_G_loss'], s=100, alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Total Epochs')
    axes[0, 1].set_ylabel('Final Generator Loss')
    axes[0, 1].set_title('Training Epochs vs Generator Loss')
    for i, test_name in enumerate(df['test_name']):
        axes[0, 1].annotate(test_name.split('_')[-1], 
                           (df['total_epochs'].iloc[i], df['final_G_loss'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 3: Generator vs Discriminator Loss
    axes[1, 0].scatter(df['final_G_loss'], df['final_D_loss'], s=100, alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Final Generator Loss')
    axes[1, 0].set_ylabel('Final Discriminator Loss')
    axes[1, 0].set_title('Generator vs Discriminator Loss')
    for i, test_name in enumerate(df['test_name']):
        axes[1, 0].annotate(test_name.split('_')[-1], 
                           (df['final_G_loss'].iloc[i], df['final_D_loss'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 4: Average Epoch Time
    axes[1, 1].bar(range(len(df)), df['avg_epoch_time'], alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Test Number')
    axes[1, 1].set_ylabel('Average Epoch Time (seconds)')
    axes[1, 1].set_title('Average Epoch Time by Test')
    axes[1, 1].set_xticks(range(len(df)))
    axes[1, 1].set_xticklabels([name.split('_')[-1] for name in df['test_name']])
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_path = os.path.join(base_path, "DCGAN_Test_256_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {comparison_path}")
    
    # Find best performing test
    best_loss_idx = df['final_G_loss'].idxmin()
    best_test = df.loc[best_loss_idx]
    
    print(f"\n{'='*80}")
    print("BEST PERFORMING TEST")
    print(f"{'='*80}")
    print(f"Test: {best_test['test_name']}")
    print(f"Final Generator Loss: {best_test['final_G_loss']:.4f}")
    print(f"Final Discriminator Loss: {best_test['final_D_loss']:.4f}")
    print(f"Total Training Time: {best_test['total_training_time']:.2f} seconds ({best_test['total_training_time']/3600:.2f} hours)")
    print(f"Total Epochs: {best_test['total_epochs']}")
    print(f"Average Epoch Time: {best_test['avg_epoch_time']:.2f} seconds")
    
    # Save detailed results to CSV
    results_path = os.path.join(base_path, "DCGAN_Test_256_results_comparison.csv")
    df.to_csv(results_path, index=False)
    print(f"\nDetailed results saved to: {results_path}")
    
    plt.show()

if __name__ == '__main__':
    analyze_dcgan_results() 