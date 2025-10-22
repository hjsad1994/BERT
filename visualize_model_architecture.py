"""
Visualize ViSoBERT + Focal Loss + Contrastive Loss Architecture

Shows:
1. Model architecture
2. Loss calculation flow
3. How both losses work together (SIMULTANEOUSLY)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def draw_architecture():
    """Draw complete model architecture"""
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Title
    ax.text(50, 98, 'ViSoBERT + Focal Loss + Contrastive Loss Architecture',
           ha='center', va='top', fontsize=18, fontweight='bold')
    
    # ========== INPUT ==========
    input_box = FancyBboxPatch((35, 88), 30, 6, boxstyle="round,pad=0.5",
                               edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(50, 91, 'INPUT TEXT', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(50, 89.5, '"Pin tot camera xau"', ha='center', va='center', fontsize=10, style='italic')
    
    # ========== VISOBERT ENCODER ==========
    encoder_box = FancyBboxPatch((30, 75), 40, 10, boxstyle="round,pad=0.5",
                                 edgecolor='green', facecolor='lightgreen', linewidth=3)
    ax.add_patch(encoder_box)
    ax.text(50, 82, 'ViSoBERT ENCODER', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(50, 79.5, 'Pre-trained Vietnamese BERT', ha='center', va='center', fontsize=10)
    ax.text(50, 77.5, 'Output: [batch_size, 768]', ha='center', va='center', fontsize=9, style='italic')
    
    # Arrow: Input -> Encoder
    arrow = FancyArrowPatch((50, 88), (50, 85), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow)
    
    # ========== SPLIT INTO 2 BRANCHES ==========
    ax.text(50, 72, 'SINGLE FORWARD PASS - PRODUCES 2 OUTPUTS:', 
           ha='center', va='center', fontsize=11, fontweight='bold', color='red')
    
    # Left Branch: Classification Head
    class_branch_y = 55
    class_box1 = FancyBboxPatch((5, class_branch_y + 10), 25, 6, boxstyle="round,pad=0.3",
                                edgecolor='purple', facecolor='lavender', linewidth=2)
    ax.add_patch(class_box1)
    ax.text(17.5, class_branch_y + 13, 'Dense Layer (512)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    class_box2 = FancyBboxPatch((5, class_branch_y + 3), 25, 6, boxstyle="round,pad=0.3",
                                edgecolor='purple', facecolor='lavender', linewidth=2)
    ax.add_patch(class_box2)
    ax.text(17.5, class_branch_y + 6, 'Dropout (0.3)', ha='center', va='center', fontsize=10)
    
    class_box3 = FancyBboxPatch((5, class_branch_y - 4), 25, 6, boxstyle="round,pad=0.3",
                                edgecolor='purple', facecolor='lavender', linewidth=2)
    ax.add_patch(class_box3)
    ax.text(17.5, class_branch_y - 1, 'Output Layer', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(17.5, class_branch_y - 2.5, '[11 aspects, 3 sentiments]', ha='center', va='center', fontsize=8, style='italic')
    
    ax.text(17.5, class_branch_y + 18, 'CLASSIFICATION BRANCH', ha='center', va='center', 
           fontsize=11, fontweight='bold', color='purple')
    
    # Right Branch: Projection Head
    proj_branch_y = 55
    proj_box1 = FancyBboxPatch((70, proj_branch_y + 10), 25, 6, boxstyle="round,pad=0.3",
                               edgecolor='orange', facecolor='lightyellow', linewidth=2)
    ax.add_patch(proj_box1)
    ax.text(82.5, proj_branch_y + 13, 'Projection (256)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    proj_box2 = FancyBboxPatch((70, proj_branch_y + 3), 25, 6, boxstyle="round,pad=0.3",
                               edgecolor='orange', facecolor='lightyellow', linewidth=2)
    ax.add_patch(proj_box2)
    ax.text(82.5, proj_branch_y + 6, 'L2 Normalize', ha='center', va='center', fontsize=10)
    ax.text(82.5, proj_branch_y + 4.5, '(for contrastive)', ha='center', va='center', fontsize=8, style='italic')
    
    ax.text(82.5, proj_branch_y + 18, 'CONTRASTIVE BRANCH', ha='center', va='center', 
           fontsize=11, fontweight='bold', color='orange')
    
    # Arrows from encoder to branches
    arrow_left = FancyArrowPatch((40, 75), (17.5, proj_branch_y + 16), arrowstyle='->', mutation_scale=20, linewidth=2, color='purple')
    ax.add_patch(arrow_left)
    
    arrow_right = FancyArrowPatch((60, 75), (82.5, proj_branch_y + 16), arrowstyle='->', mutation_scale=20, linewidth=2, color='orange')
    ax.add_patch(arrow_right)
    
    # Outputs
    ax.text(17.5, class_branch_y - 8, 'LOGITS', ha='center', va='center', fontsize=11, fontweight='bold', color='purple')
    ax.text(17.5, class_branch_y - 10, '[batch, 11, 3]', ha='center', va='center', fontsize=9, style='italic')
    
    ax.text(82.5, proj_branch_y - 8, 'EMBEDDINGS', ha='center', va='center', fontsize=11, fontweight='bold', color='orange')
    ax.text(82.5, proj_branch_y - 10, '[batch, 256]', ha='center', va='center', fontsize=9, style='italic')
    
    # ========== LOSS CALCULATION ==========
    loss_y = 32
    
    # Focal Loss
    focal_box = FancyBboxPatch((5, loss_y), 25, 10, boxstyle="round,pad=0.5",
                               edgecolor='red', facecolor='mistyrose', linewidth=3)
    ax.add_patch(focal_box)
    ax.text(17.5, loss_y + 7, 'FOCAL LOSS', ha='center', va='center', fontsize=12, fontweight='bold', color='red')
    ax.text(17.5, loss_y + 5, 'Input: LOGITS + LABELS', ha='center', va='center', fontsize=9)
    ax.text(17.5, loss_y + 3, 'Focus on hard examples', ha='center', va='center', fontsize=8, style='italic')
    ax.text(17.5, loss_y + 1.5, 'gamma=2.0', ha='center', va='center', fontsize=8)
    
    # Contrastive Loss
    contr_box = FancyBboxPatch((70, loss_y), 25, 10, boxstyle="round,pad=0.5",
                               edgecolor='green', facecolor='honeydew', linewidth=3)
    ax.add_patch(contr_box)
    ax.text(82.5, loss_y + 7, 'CONTRASTIVE LOSS', ha='center', va='center', fontsize=12, fontweight='bold', color='green')
    ax.text(82.5, loss_y + 5, 'Input: EMBEDDINGS + LABELS', ha='center', va='center', fontsize=9)
    ax.text(82.5, loss_y + 3, 'Organize embedding space', ha='center', va='center', fontsize=8, style='italic')
    ax.text(82.5, loss_y + 1.5, 'temp=0.1', ha='center', va='center', fontsize=8)
    
    # Arrows to losses
    arrow_focal = FancyArrowPatch((17.5, class_branch_y - 12), (17.5, loss_y + 10), 
                                 arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
    ax.add_patch(arrow_focal)
    
    arrow_contr = FancyArrowPatch((82.5, proj_branch_y - 12), (82.5, loss_y + 10), 
                                 arrowstyle='->', mutation_scale=20, linewidth=2, color='green')
    ax.add_patch(arrow_contr)
    
    # ========== COMBINED LOSS ==========
    combined_y = 18
    combined_box = FancyBboxPatch((35, combined_y), 30, 8, boxstyle="round,pad=0.5",
                                  edgecolor='darkred', facecolor='lightcoral', linewidth=3)
    ax.add_patch(combined_box)
    ax.text(50, combined_y + 5.5, 'COMBINED LOSS', ha='center', va='center', fontsize=13, fontweight='bold')
    ax.text(50, combined_y + 3.5, '0.8 * Focal + 0.2 * Contrastive', ha='center', va='center', fontsize=10)
    ax.text(50, combined_y + 1.5, '(weights from config)', ha='center', va='center', fontsize=8, style='italic')
    
    # Arrows to combined
    arrow_focal_combine = FancyArrowPatch((30, loss_y), (40, combined_y + 8), 
                                         arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
    ax.add_patch(arrow_focal_combine)
    
    arrow_contr_combine = FancyArrowPatch((70, loss_y), (60, combined_y + 8), 
                                         arrowstyle='->', mutation_scale=20, linewidth=2, color='green')
    ax.add_patch(arrow_contr_combine)
    
    # ========== BACKPROPAGATION ==========
    backprop_y = 8
    backprop_box = FancyBboxPatch((30, backprop_y), 40, 6, boxstyle="round,pad=0.5",
                                  edgecolor='blue', facecolor='lightblue', linewidth=3)
    ax.add_patch(backprop_box)
    ax.text(50, backprop_y + 3, 'BACKPROPAGATION', ha='center', va='center', fontsize=13, fontweight='bold')
    ax.text(50, backprop_y + 1, 'Update all model weights (ViSoBERT + both heads)', ha='center', va='center', fontsize=9)
    
    arrow_backprop = FancyArrowPatch((50, combined_y), (50, backprop_y + 6), 
                                    arrowstyle='->', mutation_scale=20, linewidth=3, color='darkblue')
    ax.add_patch(arrow_backprop)
    
    # ========== TIMING INFO ==========
    timing_box = FancyBboxPatch((2, 2), 45, 4, boxstyle="round,pad=0.3",
                                edgecolor='black', facecolor='lightyellow', linewidth=2)
    ax.add_patch(timing_box)
    ax.text(3, 4.5, 'TIMING:', ha='left', va='center', fontsize=10, fontweight='bold')
    ax.text(3, 3.3, '1. Forward: Encoder -> BOTH branches SIMULTANEOUSLY (1 pass)', 
           ha='left', va='center', fontsize=8)
    ax.text(3, 2.3, '2. Loss: Calculate BOTH losses SIMULTANEOUSLY', 
           ha='left', va='center', fontsize=8)
    ax.text(3, 1.3, '3. Backward: Single backprop updates BOTH branches', 
           ha='left', va='center', fontsize=8)
    
    key_box = FancyBboxPatch((53, 2), 45, 4, boxstyle="round,pad=0.3",
                             edgecolor='black', facecolor='lightcyan', linewidth=2)
    ax.add_patch(key_box)
    ax.text(54, 4.5, 'KEY POINT:', ha='left', va='center', fontsize=10, fontweight='bold', color='red')
    ax.text(54, 3.3, 'Both losses work TOGETHER in SAME forward pass', 
           ha='left', va='center', fontsize=8)
    ax.text(54, 2.3, 'NOT sequential (not focal first, then contrastive)', 
           ha='left', va='center', fontsize=8)
    ax.text(54, 1.3, 'They complement each other: Classification + Representation', 
           ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('model_architecture.png', dpi=150, bbox_inches='tight')
    print("Saved: model_architecture.png")
    plt.show()


def draw_forward_pass_timeline():
    """Draw timeline showing both losses calculated together"""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    ax.text(50, 98, 'Forward Pass Timeline - Both Losses SIMULTANEOUSLY',
           ha='center', va='top', fontsize=18, fontweight='bold')
    
    # Timeline
    timeline_y = 90
    ax.plot([10, 90], [timeline_y, timeline_y], 'k-', linewidth=3)
    
    # Step 1: Input
    step1_x = 15
    ax.plot([step1_x, step1_x], [timeline_y - 2, timeline_y + 2], 'b-', linewidth=3)
    step1_box = FancyBboxPatch((step1_x - 7, 75), 14, 10, boxstyle="round,pad=0.5",
                               edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(step1_box)
    ax.text(step1_x, 81, 'STEP 1', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(step1_x, 79, 'Input Text', ha='center', va='center', fontsize=10)
    ax.text(step1_x, 77, '+ Tokenize', ha='center', va='center', fontsize=9)
    
    # Step 2: Encoder
    step2_x = 35
    ax.plot([step2_x, step2_x], [timeline_y - 2, timeline_y + 2], 'g-', linewidth=3)
    step2_box = FancyBboxPatch((step2_x - 10, 70), 20, 15, boxstyle="round,pad=0.5",
                               edgecolor='green', facecolor='lightgreen', linewidth=3)
    ax.add_patch(step2_box)
    ax.text(step2_x, 82, 'STEP 2', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(step2_x, 79, 'ViSoBERT', ha='center', va='center', fontsize=10)
    ax.text(step2_x, 77, 'Encoder', ha='center', va='center', fontsize=10)
    ax.text(step2_x, 74, 'Output:', ha='center', va='center', fontsize=9)
    ax.text(step2_x, 72, '[batch, 768]', ha='center', va='center', fontsize=9, style='italic')
    
    # Step 3: PARALLEL branches
    step3_x = 62
    ax.plot([step3_x, step3_x], [timeline_y - 2, timeline_y + 2], 'purple', linewidth=3)
    
    # Upper branch: Classification
    class_box = FancyBboxPatch((step3_x - 10, 75), 20, 10, boxstyle="round,pad=0.5",
                               edgecolor='purple', facecolor='lavender', linewidth=2)
    ax.add_patch(class_box)
    ax.text(step3_x, 81.5, 'STEP 3A (PARALLEL)', ha='center', va='center', fontsize=10, fontweight='bold', color='purple')
    ax.text(step3_x, 79, 'Classification Head', ha='center', va='center', fontsize=9)
    ax.text(step3_x, 77, '-> Logits', ha='center', va='center', fontsize=9)
    
    # Lower branch: Projection
    proj_box = FancyBboxPatch((step3_x - 10, 60), 20, 10, boxstyle="round,pad=0.5",
                              edgecolor='orange', facecolor='lightyellow', linewidth=2)
    ax.add_patch(proj_box)
    ax.text(step3_x, 66.5, 'STEP 3B (PARALLEL)', ha='center', va='center', fontsize=10, fontweight='bold', color='orange')
    ax.text(step3_x, 64, 'Projection Head', ha='center', va='center', fontsize=9)
    ax.text(step3_x, 62, '-> Embeddings', ha='center', va='center', fontsize=9)
    
    # Parallel arrows
    arrow1 = FancyArrowPatch((step2_x + 10, 77), (step3_x - 10, 80), 
                            arrowstyle='->', mutation_scale=15, linewidth=2, color='purple')
    ax.add_patch(arrow1)
    arrow2 = FancyArrowPatch((step2_x + 10, 77), (step3_x - 10, 65), 
                            arrowstyle='->', mutation_scale=15, linewidth=2, color='orange')
    ax.add_patch(arrow2)
    
    ax.text(48, 84, 'PARALLEL!', ha='center', va='center', fontsize=12, fontweight='bold', 
           color='red', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Step 4: PARALLEL losses
    step4_x = 85
    ax.plot([step4_x, step4_x], [timeline_y - 2, timeline_y + 2], 'red', linewidth=3)
    
    # Focal loss
    focal_loss_box = FancyBboxPatch((step4_x - 8, 75), 16, 8, boxstyle="round,pad=0.5",
                                    edgecolor='red', facecolor='mistyrose', linewidth=2)
    ax.add_patch(focal_loss_box)
    ax.text(step4_x, 80.5, 'STEP 4A (PARALLEL)', ha='center', va='center', fontsize=9, fontweight='bold', color='red')
    ax.text(step4_x, 78, 'Focal Loss', ha='center', va='center', fontsize=9)
    ax.text(step4_x, 76, 'from Logits', ha='center', va='center', fontsize=8)
    
    # Contrastive loss
    contr_loss_box = FancyBboxPatch((step4_x - 8, 60), 16, 8, boxstyle="round,pad=0.5",
                                    edgecolor='green', facecolor='honeydew', linewidth=2)
    ax.add_patch(contr_loss_box)
    ax.text(step4_x, 65.5, 'STEP 4B (PARALLEL)', ha='center', va='center', fontsize=9, fontweight='bold', color='green')
    ax.text(step4_x, 63, 'Contrastive Loss', ha='center', va='center', fontsize=9)
    ax.text(step4_x, 61, 'from Embeddings', ha='center', va='center', fontsize=8)
    
    # Parallel arrows
    arrow3 = FancyArrowPatch((step3_x + 10, 80), (step4_x - 8, 79), 
                            arrowstyle='->', mutation_scale=15, linewidth=2, color='red')
    ax.add_patch(arrow3)
    arrow4 = FancyArrowPatch((step3_x + 10, 65), (step4_x - 8, 64), 
                            arrowstyle='->', mutation_scale=15, linewidth=2, color='green')
    ax.add_patch(arrow4)
    
    ax.text(78, 71, 'PARALLEL!', ha='center', va='center', fontsize=12, fontweight='bold', 
           color='red', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Step 5: Combine
    combine_box = FancyBboxPatch((38, 45), 24, 8, boxstyle="round,pad=0.5",
                                 edgecolor='darkred', facecolor='lightcoral', linewidth=3)
    ax.add_patch(combine_box)
    ax.text(50, 50.5, 'STEP 5: COMBINE', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(50, 48, '0.8 * Focal + 0.2 * Contr', ha='center', va='center', fontsize=10)
    ax.text(50, 46, '= Total Loss', ha='center', va='center', fontsize=9)
    
    arrow5 = FancyArrowPatch((step4_x, 75), (62, 53), 
                            arrowstyle='->', mutation_scale=15, linewidth=2, color='red')
    ax.add_patch(arrow5)
    arrow6 = FancyArrowPatch((step4_x, 68), (62, 49), 
                            arrowstyle='->', mutation_scale=15, linewidth=2, color='green')
    ax.add_patch(arrow6)
    
    # Step 6: Backprop
    backprop_box = FancyBboxPatch((35, 32), 30, 8, boxstyle="round,pad=0.5",
                                  edgecolor='blue', facecolor='lightblue', linewidth=3)
    ax.add_patch(backprop_box)
    ax.text(50, 37.5, 'STEP 6: BACKPROPAGATION', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(50, 35, 'Update ALL weights', ha='center', va='center', fontsize=10)
    ax.text(50, 33, '(Encoder + both heads)', ha='center', va='center', fontsize=9)
    
    arrow7 = FancyArrowPatch((50, 45), (50, 40), 
                            arrowstyle='->', mutation_scale=20, linewidth=3, color='blue')
    ax.add_patch(arrow7)
    
    # Summary boxes
    summary1 = FancyBboxPatch((5, 15), 40, 12, boxstyle="round,pad=0.5",
                              edgecolor='black', facecolor='lightyellow', linewidth=2)
    ax.add_patch(summary1)
    ax.text(25, 23, 'KEY INSIGHTS:', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(7, 21, '1. Forward pass: ONE time through model', ha='left', va='center', fontsize=9)
    ax.text(7, 19, '2. Branches: PARALLEL (not sequential)', ha='left', va='center', fontsize=9)
    ax.text(7, 17, '3. Losses: Calculated SIMULTANEOUSLY', ha='left', va='center', fontsize=9)
    
    summary2 = FancyBboxPatch((55, 15), 40, 12, boxstyle="round,pad=0.5",
                              edgecolor='black', facecolor='lightcyan', linewidth=2)
    ax.add_patch(summary2)
    ax.text(75, 23, 'NOT LIKE THIS:', ha='center', va='center', fontsize=11, fontweight='bold', color='red')
    ax.text(57, 21, 'X Step 1: Calculate Focal Loss', ha='left', va='center', fontsize=9)
    ax.text(57, 19, 'X Step 2: Then calculate Contrastive', ha='left', va='center', fontsize=9)
    ax.text(57, 17, 'X Sequential processing', ha='left', va='center', fontsize=9)
    
    # Timing info
    timing_box = FancyBboxPatch((10, 5), 80, 7, boxstyle="round,pad=0.5",
                                edgecolor='darkgreen', facecolor='lightgreen', linewidth=3)
    ax.add_patch(timing_box)
    ax.text(50, 10, 'TIME: Single forward pass (~20-50ms), Single backward pass (~30-60ms)', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(50, 8, 'Both losses computed in SAME iteration, backprop updates BOTH branches TOGETHER', 
           ha='center', va='center', fontsize=9)
    ax.text(50, 6, 'Result: Model learns classification (Focal) AND representation (Contrastive) SIMULTANEOUSLY', 
           ha='center', va='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig('forward_pass_timeline.png', dpi=150, bbox_inches='tight')
    print("Saved: forward_pass_timeline.png")
    plt.show()


def draw_comparison():
    """Compare sequential vs parallel"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    # WRONG: Sequential
    ax1 = axes[0]
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.axis('off')
    ax1.text(50, 95, 'WRONG: Sequential (NOT how it works)', 
            ha='center', va='center', fontsize=14, fontweight='bold', color='red')
    
    y = 80
    for i, (step, color) in enumerate([
        ('Input', 'blue'),
        ('ViSoBERT', 'green'),
        ('Focal Loss', 'red'),
        ('Backprop Focal', 'red'),
        ('Contrastive Loss', 'orange'),
        ('Backprop Contrastive', 'orange')
    ]):
        box = FancyBboxPatch((20, y - i*12), 60, 8, boxstyle="round,pad=0.5",
                            edgecolor=color, facecolor='white', linewidth=2)
        ax1.add_patch(box)
        ax1.text(50, y - i*12 + 4, f'{i+1}. {step}', ha='center', va='center', fontsize=11)
        
        if i < 5:
            arrow = FancyArrowPatch((50, y - i*12), (50, y - (i+1)*12 + 8), 
                                   arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
            ax1.add_patch(arrow)
    
    ax1.text(50, 5, 'This would take 2x time!', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # RIGHT: Parallel
    ax2 = axes[1]
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.axis('off')
    ax2.text(50, 95, 'CORRECT: Parallel (Actual implementation)', 
            ha='center', va='center', fontsize=14, fontweight='bold', color='green')
    
    steps = [
        ('Input', 'blue', 80, 60),
        ('ViSoBERT Encoder', 'green', 60, 60),
    ]
    
    y = 80
    for step, color, width, x_offset in steps:
        box = FancyBboxPatch((20, y), width, 8, boxstyle="round,pad=0.5",
                            edgecolor=color, facecolor='white', linewidth=2)
        ax2.add_patch(box)
        ax2.text(50, y + 4, step, ha='center', va='center', fontsize=11)
        y -= 12
        if y >= 60:
            arrow = FancyArrowPatch((50, y + 12), (50, y + 8), 
                                   arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
            ax2.add_patch(arrow)
    
    # Parallel branches
    y = 48
    focal_box = FancyBboxPatch((10, y), 30, 8, boxstyle="round,pad=0.5",
                              edgecolor='red', facecolor='mistyrose', linewidth=2)
    ax2.add_patch(focal_box)
    ax2.text(25, y + 4, 'Focal Loss', ha='center', va='center', fontsize=11)
    
    contr_box = FancyBboxPatch((60, y), 30, 8, boxstyle="round,pad=0.5",
                              edgecolor='orange', facecolor='lightyellow', linewidth=2)
    ax2.add_patch(contr_box)
    ax2.text(75, y + 4, 'Contrastive Loss', ha='center', va='center', fontsize=11)
    
    # Arrows from encoder to both
    arrow_left = FancyArrowPatch((35, 60), (25, 56), 
                                arrowstyle='->', mutation_scale=15, linewidth=2, color='red')
    ax2.add_patch(arrow_left)
    arrow_right = FancyArrowPatch((65, 60), (75, 56), 
                                 arrowstyle='->', mutation_scale=15, linewidth=2, color='orange')
    ax2.add_patch(arrow_right)
    
    ax2.text(50, 58, 'PARALLEL', ha='center', va='center', fontsize=12, fontweight='bold',
            color='red', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Combine
    y = 32
    combine_box = FancyBboxPatch((30, y), 40, 8, boxstyle="round,pad=0.5",
                                edgecolor='darkred', facecolor='lightcoral', linewidth=3)
    ax2.add_patch(combine_box)
    ax2.text(50, y + 4, 'Combined Loss', ha='center', va='center', fontsize=11, fontweight='bold')
    
    arrow_combine1 = FancyArrowPatch((25, 48), (40, 40), 
                                    arrowstyle='->', mutation_scale=15, linewidth=2, color='red')
    ax2.add_patch(arrow_combine1)
    arrow_combine2 = FancyArrowPatch((75, 48), (60, 40), 
                                    arrowstyle='->', mutation_scale=15, linewidth=2, color='orange')
    ax2.add_patch(arrow_combine2)
    
    # Backprop
    y = 16
    backprop_box = FancyBboxPatch((20, y), 60, 8, boxstyle="round,pad=0.5",
                                  edgecolor='blue', facecolor='lightblue', linewidth=3)
    ax2.add_patch(backprop_box)
    ax2.text(50, y + 4, 'Single Backprop (updates all)', ha='center', va='center', fontsize=11, fontweight='bold')
    
    arrow_backprop = FancyArrowPatch((50, 32), (50, 24), 
                                    arrowstyle='->', mutation_scale=15, linewidth=3, color='blue')
    ax2.add_patch(arrow_backprop)
    
    ax2.text(50, 5, 'Efficient: Single pass, both losses together!', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('sequential_vs_parallel.png', dpi=150, bbox_inches='tight')
    print("Saved: sequential_vs_parallel.png")
    plt.show()


if __name__ == '__main__':
    print("="*80)
    print("Creating Model Architecture Visualizations")
    print("="*80)
    
    print("\n1. Drawing complete architecture...")
    draw_architecture()
    
    print("\n2. Drawing forward pass timeline...")
    draw_forward_pass_timeline()
    
    print("\n3. Drawing sequential vs parallel comparison...")
    draw_comparison()
    
    print("\n" + "="*80)
    print("DONE! Created 3 visualizations:")
    print("="*80)
    print("   1. model_architecture.png - Complete model structure")
    print("   2. forward_pass_timeline.png - Step-by-step timeline")
    print("   3. sequential_vs_parallel.png - Comparison")
    
    print("\nKEY ANSWER:")
    print("   Both losses work SIMULTANEOUSLY (CUNG LUC)")
    print("   NOT sequential (NOT focal truoc roi contrastive)")
    print("   Single forward pass -> Both branches -> Both losses -> Single backprop")
