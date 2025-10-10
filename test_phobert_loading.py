"""
Test PhoBERT model loading với safetensors
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

print("="*70)
print("TEST PHOBERT MODEL LOADING VỚI SAFETENSORS")
print("="*70)

model_name = "vinai/phobert-base-v2"

print(f"\n1. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"✓ Tokenizer loaded")
print(f"   Vocab size: {tokenizer.vocab_size}")

print(f"\n2. Loading model với use_safetensors=True...")
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        use_safetensors=True  # Force safetensors format
    )
    print(f"✓ Model loaded thành công!")
    print(f"   Model type: {model.__class__.__name__}")
    print(f"   Num labels: {model.config.num_labels}")
    
    # Test inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"\n3. Test inference...")
    sentence = "sản_phẩm tuyệt_vời , pin tốt"
    aspect = "Battery"
    
    inputs = tokenizer(sentence, aspect, return_tensors='pt', padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).item()
    
    id2label = {0: 'positive', 1: 'negative', 2: 'neutral'}
    print(f"   Sentence: {sentence}")
    print(f"   Aspect: {aspect}")
    print(f"   Predicted: {id2label[pred]}")
    print(f"   Logits: {logits[0].cpu().tolist()}")
    
    print(f"\n✅ TẤT CẢ TEST PASSED!")
    
except Exception as e:
    print(f"\n❌ LỖI: {str(e)}")
    import traceback
    traceback.print_exc()

print("="*70)
