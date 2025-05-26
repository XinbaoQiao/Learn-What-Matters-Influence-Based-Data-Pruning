import os
import matplotlib.pyplot as plt

def plot_stage_comparison(stage1_train_losses, stage1_test_accs, stage2_train_losses, stage2_test_accs, result_dir):
    """
    Plot comparison of training losses between two stages and save as training_loss_comparison.png.
    Args:
        stage1_train_losses: List of training losses from stage 1
        stage1_test_accs: List of test accuracies from stage 1 (unused)
        stage2_train_losses: List of training losses from stage 2
        stage2_test_accs: List of test accuracies from stage 2 (unused)
        result_dir: Directory to save the plots
    """
    # Create comparison directory
    comparison_dir = os.path.join(result_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Plot training losses comparison
    plt.figure(figsize=(12, 8))
    plt.plot(stage1_train_losses, label='Stage 1 (Full Dataset)', linewidth=3, color='blue')
    plt.plot(stage2_train_losses, label='Stage 2 (Pruned Dataset)', linewidth=3, color='red')
    plt.xlabel('Epoch', fontsize=32, fontweight='bold')
    plt.ylabel('Training Loss', fontsize=32, fontweight='bold')
    plt.title('Training Loss Comparison', fontsize=40, fontweight='bold')
    plt.legend(fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(comparison_dir, 'training_loss_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
