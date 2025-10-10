"""
Custom Trainer với Focal Loss
"""
from transformers import Trainer


class CustomTrainer:
    """
    Custom Trainer sử dụng Focal Loss
    
    Wrapper around HuggingFace Trainer để inject custom loss function
    """
    
    @staticmethod
    def create_trainer_with_focal_loss(
        model,
        args,
        train_dataset,
        eval_dataset,
        tokenizer,
        compute_metrics,
        focal_loss,
    ):
        """
        Tạo Trainer với Focal Loss
        
        Args:
            model: Model cần train
            args: TrainingArguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer
            compute_metrics: Metrics function
            focal_loss: FocalLoss instance
        
        Returns:
            Trainer: Custom trainer với Focal Loss
        """
        
        class FocalLossTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                """
                Override compute_loss để sử dụng Focal Loss
                
                Args:
                    model: The model
                    inputs: Input dictionary
                    return_outputs: Whether to return outputs
                    num_items_in_batch: Number of items (for newer Transformers versions)
                """
                labels = inputs.pop("labels")
                
                # Forward pass
                outputs = model(**inputs)
                logits = outputs.get("logits")
                
                # Compute focal loss
                loss = focal_loss(logits, labels)
                
                return (loss, outputs) if return_outputs else loss
        
        # Create and return trainer
        trainer = FocalLossTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            compute_metrics=compute_metrics
        )
        
        return trainer
