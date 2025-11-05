"""
Custom Callback để rename checkpoint folders theo metrics (accuracy, F1, etc.)
"""
import os
import shutil
from transformers import TrainerCallback
from pathlib import Path


class MetricCheckpointCallback(TrainerCallback):
    """
    Callback tự động rename checkpoint folders theo metrics
    
    Example:
        checkpoint-1000 → checkpoint-acc90-f189
        checkpoint-2000 → checkpoint-acc92-f191
    """
    
    def __init__(self, 
                 primary_metric='eval_accuracy',
                 secondary_metric='eval_f1',
                 decimal_places=0):
        """
        Args:
            primary_metric: Metric chính để đặt tên (e.g., 'eval_accuracy', 'eval_f1')
            secondary_metric: Metric phụ (optional)
            decimal_places: Số chữ số thập phân (0 = integer, 1 = 90.5, 2 = 90.53)
        """
        self.primary_metric = primary_metric
        self.secondary_metric = secondary_metric
        self.decimal_places = decimal_places
        self.checkpoint_mapping = {}  # Track original -> renamed mapping
    
    def on_save(self, args, state, control, **kwargs):
        """
        Called when a checkpoint is saved
        """
        # Get the checkpoint directory that was just saved
        checkpoint_folder = f"checkpoint-{state.global_step}"
        checkpoint_path = Path(args.output_dir) / checkpoint_folder
        
        if not checkpoint_path.exists():
            return
        
        # Get metrics from state
        metrics = state.log_history
        
        # Find the latest eval metrics
        eval_metrics = None
        for log_entry in reversed(metrics):
            if self.primary_metric in log_entry:
                eval_metrics = log_entry
                break
        
        if eval_metrics is None:
            print(f"   No metrics found for {checkpoint_folder}, keeping original name")
            return
        
        # Extract metric values
        primary_value = eval_metrics.get(self.primary_metric, 0) * 100  # Convert to percentage
        
        # Format metric name (remove 'eval_' prefix)
        primary_name = self.primary_metric.replace('eval_', '').replace('_', '')[:3]
        
        # Build new name
        if self.decimal_places == 0:
            new_name = f"checkpoint-{primary_name}{int(primary_value)}"
        else:
            new_name = f"checkpoint-{primary_name}{primary_value:.{self.decimal_places}f}".replace('.', '_')
        
        # Add secondary metric if available
        if self.secondary_metric and self.secondary_metric in eval_metrics:
            secondary_value = eval_metrics.get(self.secondary_metric, 0) * 100
            secondary_name = self.secondary_metric.replace('eval_', '').replace('_', '')[:2]
            
            if self.decimal_places == 0:
                new_name += f"-{secondary_name}{int(secondary_value)}"
            else:
                new_name += f"-{secondary_name}{secondary_value:.{self.decimal_places}f}".replace('.', '_')
        
        new_path = Path(args.output_dir) / new_name
        
        # Rename if new name is different
        if checkpoint_path != new_path:
            try:
                # Check if target already exists
                if new_path.exists():
                    print(f"   {new_name} already exists, adding suffix")
                    counter = 1
                    while new_path.exists():
                        new_name_with_suffix = f"{new_name}-v{counter}"
                        new_path = Path(args.output_dir) / new_name_with_suffix
                        counter += 1
                
                # Rename
                checkpoint_path.rename(new_path)
                
                # Track mapping
                self.checkpoint_mapping[checkpoint_folder] = new_path.name
                
                print(f"   Renamed: {checkpoint_folder} → {new_path.name}")
                
            except Exception as e:
                print(f"   Failed to rename {checkpoint_folder}: {e}")


class SimpleMetricCheckpointCallback(TrainerCallback):
    """
    Simple version: Chỉ dùng accuracy (integer)
    
    Example:
        checkpoint-1000 → checkpoint-90  (90% accuracy)
        checkpoint-2000 → checkpoint-92  (92% accuracy)
    
    FIXED: Delays renaming until training ends to avoid conflicts with load_best_model_at_end
    """
    
    def __init__(self, metric_name='eval_accuracy'):
        """
        Args:
            metric_name: Metric để đặt tên (default: 'eval_accuracy')
        """
        self.metric_name = metric_name
        self.pending_renames = {}  # Store {original_name: new_name} mapping
    
    def on_save(self, args, state, control, **kwargs):
        """Called when checkpoint is saved - just track the rename, don't execute yet"""
        checkpoint_folder = f"checkpoint-{state.global_step}"
        checkpoint_path = Path(args.output_dir) / checkpoint_folder
        
        if not checkpoint_path.exists():
            return
        
        # Get latest eval metrics
        eval_metrics = None
        for log_entry in reversed(state.log_history):
            if self.metric_name in log_entry:
                eval_metrics = log_entry
                break
        
        if eval_metrics is None:
            return
        
        # Get metric value - lấy 4 chữ số đầu tiên không làm tròn
        # Ví dụ: 0.9185180... → 9185
        metric_value = int(eval_metrics.get(self.metric_name, 0) * 10000)
        epoch = int(eval_metrics.get('epoch', 0))
        
        # Create new name with epoch (format: checkpoint-9185-e3)
        new_name = f"checkpoint-{metric_value}-e{epoch}"
        
        # Store for later renaming
        self.pending_renames[checkpoint_folder] = new_name
        print(f"   Will rename: {checkpoint_folder} → {new_name}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Rename checkpoints after training ends (after load_best_model_at_end)"""
        print(f"\nRenaming checkpoints after training...")
        
        for old_name, new_name in self.pending_renames.items():
            old_path = Path(args.output_dir) / old_name
            new_path = Path(args.output_dir) / new_name
            
            if not old_path.exists():
                # Already renamed or deleted
                continue
            
            if old_path == new_path:
                # Same name, skip
                continue
            
            try:
                # Handle duplicates
                counter = 2
                original_new_path = new_path
                while new_path.exists():
                    new_path = Path(args.output_dir) / f"{original_new_path.name}-v{counter}"
                    counter += 1
                
                old_path.rename(new_path)
                print(f"   Renamed: {old_name} → {new_path.name}")
                
            except Exception as e:
                print(f"   Rename failed for {old_name}: {e}")


# Example usage in train.py:
if __name__ == '__main__':
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    print("="*70)
    print("Checkpoint Renamer - Examples")
    print("="*70)
    
    print("\n1. Simple callback (4 chữ số, không làm tròn):")
    print("   checkpoint-227 → checkpoint-9005-e1  (F1: 0.9005...)")
    print("   checkpoint-681 → checkpoint-9185-e3  (F1: 0.9185...)")
    
    print("\n2. Full callback (accuracy + F1):")
    print("   checkpoint-1000 → checkpoint-acc90-f189")
    print("   checkpoint-2000 → checkpoint-acc92-f191")
    
    print("\n3. With decimal places:")
    print("   checkpoint-1000 → checkpoint-acc90_5-f189_3")
    
    print("\n" + "="*70)
    print("Usage in train.py:")
    print("="*70)
    print("""
from checkpoint_renamer import SimpleMetricCheckpointCallback

# Add to trainer
callback = SimpleMetricCheckpointCallback(metric_name='eval_accuracy')
trainer.add_callback(callback)

# Or use full version
from checkpoint_renamer import MetricCheckpointCallback
callback = MetricCheckpointCallback(
    primary_metric='eval_accuracy',
    secondary_metric='eval_f1',
    decimal_places=0
)
trainer.add_callback(callback)
""")
    print("="*70)
