import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_dcgan_results(results_dir):
    """
    Analyze DCGAN training results and create comprehensive visualizations
    
    Args:
        results_dir (str): Path to the results directory
    """
    results_path = Path(results_dir)
    
    # Load the data
    try:
        batch_results = pd.read_csv(results_path / 'batch_results.csv')
        epoch_results = pd.read_csv(results_path / 'epoch_results.csv')
        training_summary = pd.read_csv(results_path / 'training_summary.csv')
        
        print("✅ Data loaded successfully!")
        print(f"📊 Training Summary:")
        print(f"   - Total Training Time: {training_summary['total_training_time'].iloc[0]/3600:.2f} hours")
        print(f"   - Total Epochs: {training_summary['total_epochs'].iloc[0]}")
        print(f"   - Total Batches: {training_summary['total_batches'].iloc[0]}")
        print(f"   - Final G Loss: {training_summary['final_G_loss'].iloc[0]:.4f}")
        print(f"   - Final D Loss: {training_summary['final_D_loss'].iloc[0]:.4f}")
        
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Create output directory for analysis
    analysis_dir = results_path / 'analysis'
    analysis_dir.mkdir(exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    
    # 1. Loss Curves Analysis
    print("\n📈 Creating loss curves analysis...")
    create_loss_curves(batch_results, epoch_results, analysis_dir)
    
    # 2. Epoch Time Analysis
    print("⏱️  Analyzing epoch time spikes...")
    analyze_epoch_times(epoch_results, analysis_dir)
    
    # 3. Training Dynamics Analysis
    print("🔄 Analyzing training dynamics...")
    analyze_training_dynamics(batch_results, epoch_results, analysis_dir)
    
    # 4. Statistical Summary
    print("📋 Generating statistical summary...")
    generate_statistical_summary(batch_results, epoch_results, training_summary, analysis_dir)
    
    print(f"\n✅ Analysis complete! Results saved to: {analysis_dir}")

def create_loss_curves(batch_results, epoch_results, output_dir):
    """Create comprehensive loss curve visualizations"""
    
    # Calculate batches per epoch
    batches_per_epoch = len(batch_results) // len(epoch_results)
    
    # Create epoch-based loss data
    epoch_losses = []
    for i in range(len(epoch_results)):
        start_idx = i * batches_per_epoch
        end_idx = min((i + 1) * batches_per_epoch, len(batch_results))
        epoch_g_loss = batch_results['G_losses'].iloc[start_idx:end_idx].mean()
        epoch_d_loss = batch_results['D_losses'].iloc[start_idx:end_idx].mean()
        epoch_losses.append({'epoch': i, 'G_loss': epoch_g_loss, 'D_loss': epoch_d_loss})
    
    epoch_losses_df = pd.DataFrame(epoch_losses)
    
    # 1. Overall Loss Curves
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Batch-level losses
    axes[0, 0].plot(batch_results['G_losses'], label='Generator Loss', alpha=0.7, linewidth=0.5, color='blue')
    axes[0, 0].plot(batch_results['D_losses'], label='Discriminator Loss', alpha=0.7, linewidth=0.5, color='orange')
    axes[0, 0].set_title('Training Losses (Batch Level)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Batch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Epoch-level losses
    axes[0, 1].plot(epoch_losses_df['epoch'], epoch_losses_df['G_loss'], 
                    label='Generator Loss', marker='o', markersize=3, color='blue')
    axes[0, 1].plot(epoch_losses_df['epoch'], epoch_losses_df['D_loss'], 
                    label='Discriminator Loss', marker='s', markersize=3, color='orange')
    axes[0, 1].set_title('Training Losses (Epoch Level)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Average Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss ratio analysis
    loss_ratio = batch_results['G_losses'] / (batch_results['D_losses'] + 1e-8)
    axes[1, 0].plot(loss_ratio, alpha=0.7, linewidth=0.5, color='purple')
    axes[1, 0].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal Loss Line')
    axes[1, 0].set_title('G/D Loss Ratio Over Time', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Batch')
    axes[1, 0].set_ylabel('G Loss / D Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss distribution
    axes[1, 1].hist(batch_results['G_losses'], bins=50, alpha=0.7, label='Generator Loss', density=True, color='blue')
    axes[1, 1].hist(batch_results['D_losses'], bins=50, alpha=0.7, label='Discriminator Loss', density=True, color='orange')
    axes[1, 1].set_title('Loss Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Loss Value')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed Loss Analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Early training (first 10% of batches)
    early_batches = int(len(batch_results) * 0.1)
    axes[0].plot(batch_results['G_losses'].iloc[:early_batches], label='G Loss', alpha=0.8, color='blue')
    axes[0].plot(batch_results['D_losses'].iloc[:early_batches], label='D Loss', alpha=0.8, color='orange')
    axes[0].set_title('Early Training (First 10% of Batches)', fontweight='bold')
    axes[0].set_xlabel('Batch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Middle training (40-60% of batches)
    mid_start = int(len(batch_results) * 0.4)
    mid_end = int(len(batch_results) * 0.6)
    axes[1].plot(batch_results['G_losses'].iloc[mid_start:mid_end], label='G Loss', alpha=0.8, color='blue')
    axes[1].plot(batch_results['D_losses'].iloc[mid_start:mid_end], label='D Loss', alpha=0.8, color='orange')
    axes[1].set_title('Middle Training (40-60% of Batches)', fontweight='bold')
    axes[1].set_xlabel('Batch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Late training (last 10% of batches)
    late_batches = int(len(batch_results) * 0.1)
    axes[2].plot(batch_results['G_losses'].iloc[-late_batches:], label='G Loss', alpha=0.8, color='blue')
    axes[2].plot(batch_results['D_losses'].iloc[-late_batches:], label='D Loss', alpha=0.8, color='orange')
    axes[2].set_title('Late Training (Last 10% of Batches)', fontweight='bold')
    axes[2].set_xlabel('Batch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detailed_loss_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_epoch_times(epoch_results, output_dir):
    """Analyze epoch time spikes and patterns"""
    
    # 1. Epoch Time Analysis
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Overall epoch times
    axes[0, 0].plot(epoch_results['epoch'], epoch_results['epoch_time'], 
                    marker='o', markersize=3, alpha=0.7, color='green')
    axes[0, 0].set_title('Epoch Times Over Training', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Identify spikes (epochs that are > 2 standard deviations above mean)
    mean_time = epoch_results['epoch_time'].mean()
    std_time = epoch_results['epoch_time'].std()
    spike_threshold = mean_time + 2 * std_time
    
    spike_epochs = epoch_results[epoch_results['epoch_time'] > spike_threshold]
    normal_epochs = epoch_results[epoch_results['epoch_time'] <= spike_threshold]
    
    # Highlight spikes
    axes[0, 0].scatter(spike_epochs['epoch'], spike_epochs['epoch_time'], 
                       color='red', s=50, zorder=5, label=f'Spikes (> {spike_threshold:.1f}s)')
    axes[0, 0].legend()
    
    # Time distribution
    axes[0, 1].hist(epoch_results['epoch_time'], bins=30, alpha=0.7, edgecolor='black', color='lightblue')
    axes[0, 1].axvline(mean_time, color='red', linestyle='--', label=f'Mean: {mean_time:.1f}s')
    axes[0, 1].axvline(spike_threshold, color='orange', linestyle='--', 
                       label=f'Spike Threshold: {spike_threshold:.1f}s')
    axes[0, 1].set_title('Epoch Time Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Moving average
    window_size = 10
    moving_avg = epoch_results['epoch_time'].rolling(window=window_size, center=True).mean()
    axes[1, 0].plot(epoch_results['epoch'], epoch_results['epoch_time'], 
                    alpha=0.5, label='Raw Times', color='green')
    axes[1, 0].plot(epoch_results['epoch'], moving_avg, 
                    linewidth=2, color='red', label=f'{window_size}-Epoch Moving Average')
    axes[1, 0].set_title('Epoch Times with Moving Average', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Spike analysis
    if len(spike_epochs) > 0:
        axes[1, 1].bar(range(len(spike_epochs)), spike_epochs['epoch_time'], 
                       color='red', alpha=0.7)
        axes[1, 1].set_title('Spike Epochs Analysis', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Spike Index')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_xticks(range(len(spike_epochs)))
        axes[1, 1].set_xticklabels([f'E{int(epoch)}' for epoch in spike_epochs['epoch']], rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No significant spikes detected', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Spike Epochs Analysis', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'epoch_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print spike analysis
    print(f"\n🔍 Epoch Time Spike Analysis:")
    print(f"   - Mean epoch time: {mean_time:.2f} seconds")
    print(f"   - Standard deviation: {std_time:.2f} seconds")
    print(f"   - Spike threshold: {spike_threshold:.2f} seconds")
    print(f"   - Number of spikes: {len(spike_epochs)}")
    
    if len(spike_epochs) > 0:
        print(f"   - Spike epochs: {list(spike_epochs['epoch'].astype(int))}")
        print(f"   - Spike times: {list(spike_epochs['epoch_time'].round(2))}")

def analyze_training_dynamics(batch_results, epoch_results, output_dir):
    """Analyze training dynamics and convergence"""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Convergence analysis
    window_size = 1000
    g_moving_avg = batch_results['G_losses'].rolling(window=window_size, center=True).mean()
    d_moving_avg = batch_results['D_losses'].rolling(window=window_size, center=True).mean()
    
    axes[0, 0].plot(batch_results['G_losses'], alpha=0.3, label='G Loss (Raw)', linewidth=0.5, color='blue')
    axes[0, 0].plot(batch_results['D_losses'], alpha=0.3, label='D Loss (Raw)', linewidth=0.5, color='orange')
    axes[0, 0].plot(g_moving_avg, linewidth=2, label=f'G Loss ({window_size}-batch MA)', color='blue')
    axes[0, 0].plot(d_moving_avg, linewidth=2, label=f'D Loss ({window_size}-batch MA)', color='orange')
    axes[0, 0].set_title('Training Convergence Analysis', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Batch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss stability analysis
    g_loss_std = batch_results['G_losses'].rolling(window=window_size, center=True).std()
    d_loss_std = batch_results['D_losses'].rolling(window=window_size, center=True).std()
    
    axes[0, 1].plot(g_loss_std, label='G Loss Std Dev', alpha=0.8, color='blue')
    axes[0, 1].plot(d_loss_std, label='D Loss Std Dev', alpha=0.8, color='orange')
    axes[0, 1].set_title('Loss Stability (Standard Deviation)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Batch')
    axes[0, 1].set_ylabel('Standard Deviation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training phases
    total_batches = len(batch_results)
    phase1_end = int(total_batches * 0.2)  # First 20%
    phase2_end = int(total_batches * 0.8)  # Up to 80%
    
    # Phase 1: Initial learning
    axes[1, 0].plot(batch_results['G_losses'].iloc[:phase1_end], 
                    label='G Loss', alpha=0.8, color='blue')
    axes[1, 0].plot(batch_results['D_losses'].iloc[:phase1_end], 
                    label='D Loss', alpha=0.8, color='orange')
    axes[1, 0].set_title('Phase 1: Initial Learning (First 20%)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Batch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Phase 3: Final convergence
    axes[1, 1].plot(batch_results['G_losses'].iloc[phase2_end:], 
                    label='G Loss', alpha=0.8, color='blue')
    axes[1, 1].plot(batch_results['D_losses'].iloc[phase2_end:], 
                    label='D Loss', alpha=0.8, color='orange')
    axes[1, 1].set_title('Phase 3: Final Convergence (Last 20%)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Batch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_dynamics_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_statistical_summary(batch_results, epoch_results, training_summary, output_dir):
    """Generate comprehensive statistical summary"""
    
    # Create summary statistics
    summary_stats = {
        'Metric': [
            'Total Training Time (hours)',
            'Total Epochs',
            'Total Batches',
            'Batches per Epoch',
            'Final G Loss',
            'Final D Loss',
            'Mean G Loss',
            'Mean D Loss',
            'Std G Loss',
            'Std D Loss',
            'Min G Loss',
            'Min D Loss',
            'Max G Loss',
            'Max D Loss',
            'Mean Epoch Time (seconds)',
            'Std Epoch Time (seconds)',
            'Min Epoch Time (seconds)',
            'Max Epoch Time (seconds)',
            'G/D Loss Ratio (Final)',
            'G/D Loss Ratio (Mean)'
        ],
        'Value': [
            training_summary['total_training_time'].iloc[0] / 3600,
            training_summary['total_epochs'].iloc[0],
            training_summary['total_batches'].iloc[0],
            training_summary['total_batches'].iloc[0] / training_summary['total_epochs'].iloc[0],
            training_summary['final_G_loss'].iloc[0],
            training_summary['final_D_loss'].iloc[0],
            batch_results['G_losses'].mean(),
            batch_results['D_losses'].mean(),
            batch_results['G_losses'].std(),
            batch_results['D_losses'].std(),
            batch_results['G_losses'].min(),
            batch_results['D_losses'].min(),
            batch_results['G_losses'].max(),
            batch_results['D_losses'].max(),
            epoch_results['epoch_time'].mean(),
            epoch_results['epoch_time'].std(),
            epoch_results['epoch_time'].min(),
            epoch_results['epoch_time'].max(),
            training_summary['final_G_loss'].iloc[0] / training_summary['final_D_loss'].iloc[0],
            batch_results['G_losses'].mean() / batch_results['D_losses'].mean()
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df['Value'] = summary_df['Value'].round(4)
    
    # Save summary
    summary_df.to_csv(output_dir / 'statistical_summary.csv', index=False)
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss statistics
    loss_stats = ['Mean G Loss', 'Mean D Loss', 'Std G Loss', 'Std D Loss']
    loss_values = [summary_df[summary_df['Metric'] == stat]['Value'].iloc[0] for stat in loss_stats]
    
    axes[0, 0].bar(loss_stats, loss_values, color=['blue', 'orange', 'lightblue', 'lightcoral'])
    axes[0, 0].set_title('Loss Statistics', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Loss Value')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Time statistics
    time_stats = ['Mean Epoch Time', 'Std Epoch Time', 'Min Epoch Time', 'Max Epoch Time']
    time_values = [summary_df[summary_df['Metric'] == stat]['Value'].iloc[0] for stat in time_stats]
    
    axes[0, 1].bar(time_stats, time_values, color=['green', 'lightgreen', 'darkgreen', 'red'])
    axes[0, 1].set_title('Epoch Time Statistics', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Training overview
    overview_stats = ['Total Training Time (hours)', 'Total Epochs', 'Total Batches', 'Batches per Epoch']
    overview_values = [summary_df[summary_df['Metric'] == stat]['Value'].iloc[0] for stat in overview_stats]
    
    axes[1, 0].bar(overview_stats, overview_values, color=['purple', 'pink', 'violet', 'lavender'])
    axes[1, 0].set_title('Training Overview', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Loss ratios
    ratio_stats = ['G/D Loss Ratio (Final)', 'G/D Loss Ratio (Mean)']
    ratio_values = [summary_df[summary_df['Metric'] == stat]['Value'].iloc[0] for stat in ratio_stats]
    
    axes[1, 1].bar(ratio_stats, ratio_values, color=['gold', 'yellow'])
    axes[1, 1].set_title('Loss Ratios', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Ratio')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Loss Line')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'statistical_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print key insights
    print(f"\n📊 Key Training Insights:")
    print(f"   - Training completed successfully with {training_summary['total_epochs'].iloc[0]} epochs")
    print(f"   - Final G/D loss ratio: {summary_df[summary_df['Metric'] == 'G/D Loss Ratio (Final)']['Value'].iloc[0]:.4f}")
    print(f"   - Average epoch time: {summary_df[summary_df['Metric'] == 'Mean Epoch Time (seconds)']['Value'].iloc[0]:.2f} seconds")
    print(f"   - Loss stability (G std): {summary_df[summary_df['Metric'] == 'Std G Loss']['Value'].iloc[0]:.4f}")
    print(f"   - Loss stability (D std): {summary_df[summary_df['Metric'] == 'Std D Loss']['Value'].iloc[0]:.4f}")

if __name__ == "__main__":
    # Analyze the DCGAN Test 256_4 results
    results_directory = "../../generated_images/DCGAN_Test_256_4"
    analyze_dcgan_results(results_directory) 