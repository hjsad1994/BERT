"""
Visualize the relationship between Focal Loss and Contrastive Loss
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_scenarios():
    """Plot different training scenarios"""
    
    epochs = np.arange(1, 16)
    
    # Scenario 1: Good training (both decrease)
    focal_good = 0.8 * np.exp(-0.15 * epochs) + 0.2
    contr_good = 0.7 * np.exp(-0.12 * epochs) + 0.3
    
    # Scenario 2: Focal decreases, Contrastive stuck
    focal_bad1 = 0.8 * np.exp(-0.20 * epochs) + 0.15
    contr_bad1 = 0.65 + 0.05 * np.sin(epochs * 0.5)
    
    # Scenario 3: Contrastive decreases, Focal stuck
    focal_bad2 = 0.75 + 0.05 * np.sin(epochs * 0.5)
    contr_bad2 = 0.7 * np.exp(-0.20 * epochs) + 0.25
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Good training
    ax1 = axes[0]
    ax1.plot(epochs, focal_good, 'b-o', label='Focal Loss', linewidth=2)
    ax1.plot(epochs, contr_good, 'r-s', label='Contrastive Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('GOOD: Both Decrease Together', fontsize=14, fontweight='bold', color='green')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.annotate('Both decrease\nModel learning well!', 
                xy=(10, 0.4), xytext=(10, 0.7),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, color='green', fontweight='bold')
    
    # Plot 2: Focal decreases, Contrastive stuck
    ax2 = axes[1]
    ax2.plot(epochs, focal_bad1, 'b-o', label='Focal Loss', linewidth=2)
    ax2.plot(epochs, contr_bad1, 'r-s', label='Contrastive Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('BAD: Contrastive Not Decreasing', fontsize=14, fontweight='bold', color='red')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.annotate('Contrastive stuck!\nIncrease contrastive_weight', 
                xy=(10, 0.65), xytext=(5, 0.85),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', fontweight='bold')
    
    # Plot 3: Contrastive decreases, Focal stuck
    ax3 = axes[2]
    ax3.plot(epochs, focal_bad2, 'b-o', label='Focal Loss', linewidth=2)
    ax3.plot(epochs, contr_bad2, 'r-s', label='Contrastive Loss', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax3.set_title('BAD: Focal Not Decreasing', fontsize=14, fontweight='bold', color='red')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.annotate('Focal stuck!\nIncrease focal_weight', 
                xy=(10, 0.75), xytext=(5, 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('loss_relationship.png', dpi=150, bbox_inches='tight')
    print("Saved: loss_relationship.png")
    plt.show()


def plot_embedding_space():
    """Visualize how contrastive loss improves embedding space"""
    
    np.random.seed(42)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Before training (high contrastive loss = 0.8)
    ax1 = axes[0]
    pos_x_before = np.random.randn(30) * 2 + 2
    pos_y_before = np.random.randn(30) * 2 + 2
    neg_x_before = np.random.randn(30) * 2 + 2
    neg_y_before = np.random.randn(30) * 2 - 2
    
    ax1.scatter(pos_x_before, pos_y_before, c='red', s=100, alpha=0.6, label='Battery=Positive', edgecolors='black')
    ax1.scatter(neg_x_before, neg_y_before, c='blue', s=100, alpha=0.6, label='Battery=Negative', edgecolors='black')
    ax1.set_xlabel('Dimension 1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Dimension 2', fontsize=12, fontweight='bold')
    ax1.set_title('Before Training\nContrastive Loss = 0.80 (High)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.5, 0.95, 'Mixed up, not organized', 
            transform=ax1.transAxes, ha='center', va='top',
            fontsize=11, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Mid training (medium contrastive loss = 0.5)
    ax2 = axes[1]
    pos_x_mid = np.random.randn(30) * 1.5 - 3
    pos_y_mid = np.random.randn(30) * 1.5 + 3
    neg_x_mid = np.random.randn(30) * 1.5 + 3
    neg_y_mid = np.random.randn(30) * 1.5 - 3
    
    ax2.scatter(pos_x_mid, pos_y_mid, c='red', s=100, alpha=0.6, label='Battery=Positive', edgecolors='black')
    ax2.scatter(neg_x_mid, neg_y_mid, c='blue', s=100, alpha=0.6, label='Battery=Negative', edgecolors='black')
    ax2.set_xlabel('Dimension 1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Dimension 2', fontsize=12, fontweight='bold')
    ax2.set_title('Mid Training\nContrastive Loss = 0.50 (Medium)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.5, 0.95, 'Getting better...', 
            transform=ax2.transAxes, ha='center', va='top',
            fontsize=11, color='orange', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # After training (low contrastive loss = 0.3)
    ax3 = axes[2]
    pos_x_after = np.random.randn(30) * 0.8 - 4
    pos_y_after = np.random.randn(30) * 0.8 + 4
    neg_x_after = np.random.randn(30) * 0.8 + 4
    neg_y_after = np.random.randn(30) * 0.8 - 4
    
    ax3.scatter(pos_x_after, pos_y_after, c='red', s=100, alpha=0.6, label='Battery=Positive', edgecolors='black')
    ax3.scatter(neg_x_after, neg_y_after, c='blue', s=100, alpha=0.6, label='Battery=Negative', edgecolors='black')
    ax3.set_xlabel('Dimension 1', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Dimension 2', fontsize=12, fontweight='bold')
    ax3.set_title('After Training\nContrastive Loss = 0.30 (Low)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.text(0.5, 0.95, 'Well organized!', 
            transform=ax3.transAxes, ha='center', va='top',
            fontsize=11, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('embedding_space_evolution.png', dpi=150, bbox_inches='tight')
    print("Saved: embedding_space_evolution.png")
    plt.show()


def print_summary():
    """Print text summary"""
    print("\n" + "="*80)
    print("FOCAL vs CONTRASTIVE LOSS - SUMMARY")
    print("="*80)
    
    print("\nKEY INSIGHT:")
    print("   - Both losses should DECREASE together during good training")
    print("   - NOT 'one decreases, other increases'")
    
    print("\nLOSS MEANINGS:")
    print("\n   Focal Loss:")
    print("      - High (0.6-0.8) = Poor classification")
    print("      - Low (0.2-0.3)  = Good classification")
    
    print("\n   Contrastive Loss:")
    print("      - High (0.6-0.8) = Embeddings not organized")
    print("      - Low (0.3-0.4)  = Embeddings well organized")
    
    print("\nWHAT HAPPENS WHEN LOSS DECREASES:")
    print("\n   Contrastive Loss Decreases:")
    print("      x NOT: 'Push samples away'")
    print("      v YES: 'Similar samples pulled CLOSER'")
    print("      v YES: 'Different samples pushed FARTHER'")
    print("      - Result: Better organized embedding space")
    
    print("\nWEIGHT ADJUSTMENT:")
    print("\n   Scenario 1: Both decrease evenly")
    print("      - Keep weights (focal=0.8, contrastive=0.2)")
    
    print("\n   Scenario 2: Focal decreases, Contrastive stuck")
    print("      - Increase contrastive_weight: 0.2 -> 0.3")
    
    print("\n   Scenario 3: Contrastive decreases, Focal stuck")
    print("      - Increase focal_weight: 0.8 -> 0.9")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    print("\nVisualizing Loss Relationships...\n")
    
    try:
        print("1. Plotting loss scenarios...")
        plot_loss_scenarios()
        
        print("\n2. Plotting embedding space evolution...")
        plot_embedding_space()
        
        print("\n3. Printing summary...")
        print_summary()
        
        print("\nDone! Check the generated PNG files.")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure matplotlib is installed: pip install matplotlib")
