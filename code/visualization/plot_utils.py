import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics(train_losses, train_accs, valid_losses, valid_accs, test_losses, test_accs, result_dir):
    """
    Plot and save training, validation and testing metrics (losses and accuracies) separately.
    
    Args:
        train_losses: List of training losses
        train_accs: List of training accuracies
        valid_losses: List of validation losses
        valid_accs: List of validation accuracies
        test_losses: List of testing losses
        test_accs: List of testing accuracies
        result_dir: Directory to save the plots
    """
    # Create train, valid and test subdirectories
    train_dir = os.path.join(result_dir, 'train')
    valid_dir = os.path.join(result_dir, 'valid')
    test_dir = os.path.join(result_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Plot training loss
    plt.figure(figsize=(12, 8))
    plt.plot(train_losses, label='Train Loss', linewidth=3, color='blue')
    plt.xlabel('Step', fontsize=32, fontweight='bold')
    plt.ylabel('Loss', fontsize=32, fontweight='bold')
    plt.title('Training Loss', fontsize=40, fontweight='bold')
    plt.legend(fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(train_dir, 'loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot training accuracy
    plt.figure(figsize=(12, 8))
    plt.plot(train_accs, label='Train Accuracy', linewidth=3, color='green')
    plt.xlabel('Step', fontsize=32, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=32, fontweight='bold')
    plt.title('Training Accuracy', fontsize=40, fontweight='bold')
    plt.legend(fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.1f}%'))
    plt.savefig(os.path.join(train_dir, 'accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot validation loss
    plt.figure(figsize=(12, 8))
    plt.plot(valid_losses, label='Valid Loss', linewidth=3, color='orange')
    plt.xlabel('Step', fontsize=32, fontweight='bold')
    plt.ylabel('Loss', fontsize=32, fontweight='bold')
    plt.title('Validation Loss', fontsize=40, fontweight='bold')
    plt.legend(fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(valid_dir, 'loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot validation accuracy
    plt.figure(figsize=(12, 8))
    plt.plot(valid_accs, label='Valid Accuracy', linewidth=3, color='purple')
    plt.xlabel('Step', fontsize=32, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=32, fontweight='bold')
    plt.title('Validation Accuracy', fontsize=40, fontweight='bold')
    plt.legend(fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.1f}%'))
    plt.savefig(os.path.join(valid_dir, 'accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot test loss
    plt.figure(figsize=(12, 8))
    plt.plot(test_losses, label='Test Loss', linewidth=3, color='red')
    plt.xlabel('Step', fontsize=32, fontweight='bold')
    plt.ylabel('Loss', fontsize=32, fontweight='bold')
    plt.title('Test Loss', fontsize=40, fontweight='bold')
    plt.legend(fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(test_dir, 'loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot test accuracy
    plt.figure(figsize=(12, 8))
    plt.plot(test_accs, label='Test Accuracy', linewidth=3, color='brown')
    plt.xlabel('Step', fontsize=32, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=32, fontweight='bold')
    plt.title('Test Accuracy', fontsize=40, fontweight='bold')
    plt.legend(fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.1f}%'))
    plt.savefig(os.path.join(test_dir, 'accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
