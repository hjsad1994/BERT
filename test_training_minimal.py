"""
Test minimal training để debug lỗi
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from utils import compute_metrics

print("="*70)
print("TEST MINIMAL TRAINING VỚI PHOBERT")
print("="*70)

# 1. Load model và tokenizer
print("\n1. Loading model...")
model_name = "vinai/phobert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    use_safetensors=True
)
print("✓ Model loaded")

# 2. Load minimal data
print("\n2. Loading minimal data...")
df = pd.read_csv('data/train.csv', encoding='utf-8-sig', nrows=100)  # Only 100 samples for test
df_val = pd.read_csv('data/validation.csv', encoding='utf-8-sig', nrows=50)

# Map sentiments
sentiment_map = {'positive': 0, 'negative': 1, 'neutral': 2}
df['label'] = df['sentiment'].map(sentiment_map)
df_val['label'] = df_val['sentiment'].map(sentiment_map)

print(f"✓ Train: {len(df)} samples")
print(f"✓ Val: {len(df_val)} samples")

# 3. Tokenize
print("\n3. Tokenizing...")
def tokenize_function(examples):
    return tokenizer(
        examples['sentence'],
        examples['aspect'],
        padding='max_length',
        truncation=True,
        max_length=256
    )

train_dataset = Dataset.from_pandas(df[['sentence', 'aspect', 'label']])
val_dataset = Dataset.from_pandas(df_val[['sentence', 'aspect', 'label']])

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.rename_column('label', 'labels')
val_dataset = val_dataset.rename_column('label', 'labels')

print("✓ Tokenization done")

# 4. Training args
print("\n4. Setting up training args...")
training_args = TrainingArguments(
    output_dir="test_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",  # Using f1
    greater_is_better=True,
    logging_steps=10,
    save_total_limit=2,
    fp16=torch.cuda.is_available()
)
print("✓ Training args set")

# 5. Create trainer
print("\n5. Creating trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Add early stopping with proper threshold
print("\n6. Adding early stopping callback...")
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=1,
    early_stopping_threshold=0.01  # For F1, use 0.01 not 0.001
)
trainer.add_callback(early_stopping)
print("✓ Early stopping added")

# 6. Train
print("\n7. Starting training...")
try:
    train_result = trainer.train()
    print("\n✅ Training completed successfully!")
    print(f"   Final metrics: {train_result.metrics}")
except Exception as e:
    print(f"\n❌ Training failed: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
