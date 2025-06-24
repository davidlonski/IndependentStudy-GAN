import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

def analyze_wgan_results():
    """Analyze and compare results from different WGAN test runs"""
    
    # Find all WGAN test directories
    base_path = "../../generated_images"
    wgan_dirs = glob.glob(f"{base_path}/WGAN_Test_256_*")
    
    if not wgan_dirs:
        print("No WGAN test results found!")
        return
    
    print(f"Found {len(wgan_dirs)} WGAN test directories:")
    for dir_path in wgan_dirs:
        print(f"  - {os.path.basename(dir_path)}")
    
    # Collect results from each test
    results = []
    
    for dir_path in sorted(wgan_dirs):
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
    print("WGAN TEST RESULTS SUMMARY")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('WGAN Test Results Comparison', fontsize=16)
    
    # Plot 1: Training Time vs Generator Loss
    axes[0, 0].scatter(df['total_training_time'], df['final_G_loss'], s=100, alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('Total Training Time (seconds)')
    axes[0, 0].set_ylabel('Final Generator Loss')
    axes[0, 0].set_title('Training Time vs Generator Loss')
    for i, test_name in enumerate(df['test_name']):
        axes[0, 0].annotate(test_name.split('_')[-1], 
                           (df['total_training_time'].iloc[i], df['final_G_loss'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 2: Epochs vs Generator Loss
    axes[0, 1].scatter(df['total_epochs'], df['final_G_loss'], s=100, alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Total Epochs')
    axes[0, 1].set_ylabel('Final Generator Loss')
    axes[0, 1].set_title('Training Epochs vs Generator Loss')
    for i, test_name in enumerate(df['test_name']):
        axes[0, 1].annotate(test_name.split('_')[-1], 
                           (df['total_epochs'].iloc[i], df['final_G_loss'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 3: Generator vs Discriminator Loss (WGAN specific)
    axes[0, 2].scatter(df['final_G_loss'], df['final_D_loss'], s=100, alpha=0.7, color='green')
    axes[0, 2].set_xlabel('Final Generator Loss')
    axes[0, 2].set_ylabel('Final Discriminator Loss')
    axes[0, 2].set_title('Generator vs Discriminator Loss')
    for i, test_name in enumerate(df['test_name']):
        axes[0, 2].annotate(test_name.split('_')[-1], 
                           (df['final_G_loss'].iloc[i], df['final_D_loss'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 4: Average Epoch Time
    axes[1, 0].bar(range(len(df)), df['avg_epoch_time'], alpha=0.7, color='purple')
    axes[1, 0].set_xlabel('Test Number')
    axes[1, 0].set_ylabel('Average Epoch Time (seconds)')
    axes[1, 0].set_title('Average Epoch Time by Test')
    axes[1, 0].set_xticks(range(len(df)))
    axes[1, 0].set_xticklabels([name.split('_')[-1] for name in df['test_name']])
    
    # Plot 5: Loss Convergence (WGAN specific - should converge to 0)
    axes[1, 1].scatter(df['final_G_loss'], df['final_D_loss'], s=100, alpha=0.7, color='red')
    axes[1, 1].set_xlabel('Final Generator Loss')
    axes[1, 1].set_ylabel('Final Discriminator Loss')
    axes[1, 1].set_title('WGAN Loss Convergence\n(Closer to 0 is better)')
    # Add reference lines for optimal convergence
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    for i, test_name in enumerate(df['test_name']):
        axes[1, 1].annotate(test_name.split('_')[-1], 
                           (df['final_G_loss'].iloc[i], df['final_D_loss'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 6: Training Efficiency (Time per epoch vs final loss)
    axes[1, 2].scatter(df['avg_epoch_time'], df['final_G_loss'], s=100, alpha=0.7, color='brown')
    axes[1, 2].set_xlabel('Average Epoch Time (seconds)')
    axes[1, 2].set_ylabel('Final Generator Loss')
    axes[1, 2].set_title('Training Efficiency\n(Lower left is better)')
    for i, test_name in enumerate(df['test_name']):
        axes[1, 2].annotate(test_name.split('_')[-1], 
                           (df['avg_epoch_time'].iloc[i], df['final_G_loss'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_path = os.path.join(base_path, "WGAN_Test_256_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {comparison_path}")
    
    # WGAN-specific analysis
    print(f"\n{'='*80}")
    print("WGAN-SPECIFIC ANALYSIS")
    print(f"{'='*80}")
    
    # Find best performing test (lowest generator loss)
    best_loss_idx = df['final_G_loss'].idxmin()
    best_test = df.loc[best_loss_idx]
    
    # Find most stable test (closest to 0 for both losses)
    df['loss_distance_from_optimal'] = ((df['final_G_loss'] - 0)**2 + (df['final_D_loss'] - 0)**2)**0.5
    most_stable_idx = df['loss_distance_from_optimal'].idxmin()
    most_stable_test = df.loc[most_stable_idx]
    
    # Find fastest test
    fastest_idx = df['avg_epoch_time'].idxmin()
    fastest_test = df.loc[fastest_idx]
    
    print(f"BEST LOSS PERFORMANCE:")
    print(f"  Test: {best_test['test_name']}")
    print(f"  Final Generator Loss: {best_test['final_G_loss']:.4f}")
    print(f"  Final Discriminator Loss: {best_test['final_D_loss']:.4f}")
    print(f"  Total Training Time: {best_test['total_training_time']:.2f} seconds ({best_test['total_training_time']/3600:.2f} hours)")
    
    print(f"\nMOST STABLE CONVERGENCE:")
    print(f"  Test: {most_stable_test['test_name']}")
    print(f"  Final Generator Loss: {most_stable_test['final_G_loss']:.4f}")
    print(f"  Final Discriminator Loss: {most_stable_test['final_D_loss']:.4f}")
    print(f"  Distance from Optimal (0,0): {most_stable_test['loss_distance_from_optimal']:.4f}")
    
    print(f"\nFASTEST TRAINING:")
    print(f"  Test: {fastest_test['test_name']}")
    print(f"  Average Epoch Time: {fastest_test['avg_epoch_time']:.2f} seconds")
    print(f"  Final Generator Loss: {fastest_test['final_G_loss']:.4f}")
    print(f"  Total Training Time: {fastest_test['total_training_time']:.2f} seconds ({fastest_test['total_training_time']/3600:.2f} hours)")
    
    # WGAN convergence analysis
    print(f"\nWGAN CONVERGENCE ANALYSIS:")
    print(f"  Tests with G loss < 0.1: {len(df[df['final_G_loss'] < 0.1])}/{len(df)}")
    print(f"  Tests with D loss < 0.1: {len(df[df['final_D_loss'] < 0.1])}/{len(df)}")
    print(f"  Tests with both losses < 0.1: {len(df[(df['final_G_loss'] < 0.1) & (df['final_D_loss'] < 0.1)])}/{len(df)}")
    
    # Training time analysis
    print(f"\nTRAINING TIME ANALYSIS:")
    print(f"  Average training time: {df['total_training_time'].mean():.2f} seconds ({df['total_training_time'].mean()/3600:.2f} hours)")
    print(f"  Fastest training: {df['total_training_time'].min():.2f} seconds ({df['total_training_time'].min()/3600:.2f} hours)")
    print(f"  Slowest training: {df['total_training_time'].max():.2f} seconds ({df['total_training_time'].max()/3600:.2f} hours)")
    
    # Save detailed results to CSV
    results_path = os.path.join(base_path, "WGAN_Test_256_results_comparison.csv")
    df.to_csv(results_path, index=False)
    print(f"\nDetailed results saved to: {results_path}")
    
    # Create additional WGAN-specific analysis
    create_wgan_detailed_analysis(df, base_path)
    
    plt.show()

def create_wgan_detailed_analysis(df, base_path):
    """Create additional WGAN-specific analysis plots"""
    
    # Create a second figure for detailed WGAN analysis
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))
    fig2.suptitle('WGAN Detailed Analysis', fontsize=16)
    
    # Plot 1: Loss distribution
    axes2[0, 0].hist(df['final_G_loss'], bins=10, alpha=0.7, color='blue', label='Generator Loss')
    axes2[0, 0].hist(df['final_D_loss'], bins=10, alpha=0.7, color='red', label='Discriminator Loss')
    axes2[0, 0].set_xlabel('Loss Value')
    axes2[0, 0].set_ylabel('Frequency')
    axes2[0, 0].set_title('Distribution of Final Losses')
    axes2[0, 0].legend()
    
    # Plot 2: Training time distribution
    axes2[0, 1].hist(df['total_training_time']/3600, bins=10, alpha=0.7, color='green')
    axes2[0, 1].set_xlabel('Training Time (hours)')
    axes2[0, 1].set_ylabel('Frequency')
    axes2[0, 1].set_title('Distribution of Training Times')
    
    # Plot 3: Loss ratio (G/D) - WGAN should have balanced losses
    loss_ratio = df['final_G_loss'] / df['final_D_loss']
    axes2[1, 0].scatter(range(len(df)), loss_ratio, s=100, alpha=0.7, color='purple')
    axes2[1, 0].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Balanced (G/D = 1)')
    axes2[1, 0].set_xlabel('Test Number')
    axes2[1, 0].set_ylabel('Generator/Discriminator Loss Ratio')
    axes2[1, 0].set_title('Loss Balance Analysis\n(Closer to 1 is better)')
    axes2[1, 0].legend()
    axes2[1, 0].set_xticks(range(len(df)))
    axes2[1, 0].set_xticklabels([name.split('_')[-1] for name in df['test_name']])
    
    # Plot 4: Efficiency score (lower loss / time)
    efficiency_score = df['final_G_loss'] / (df['total_training_time'] / 3600)  # Loss per hour
    axes2[1, 1].bar(range(len(df)), efficiency_score, alpha=0.7, color='orange')
    axes2[1, 1].set_xlabel('Test Number')
    axes2[1, 1].set_ylabel('Loss per Hour of Training')
    axes2[1, 1].set_title('Training Efficiency Score\n(Lower is better)')
    axes2[1, 1].set_xticks(range(len(df)))
    axes2[1, 1].set_xticklabels([name.split('_')[-1] for name in df['test_name']])
    
    plt.tight_layout()
    
    # Save detailed analysis plot
    detailed_path = os.path.join(base_path, "WGAN_Test_256_detailed_analysis.png")
    plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
    print(f"Detailed analysis plot saved to: {detailed_path}")

if __name__ == '__main__':
    analyze_wgan_results() 