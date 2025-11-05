import os
import argparse
import yaml
import torch
from transformers import AutoTokenizer

from model_multitask import DualTaskViSoBERT


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dir(directory: str) -> None:
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def build_model_from_config(cfg: dict, device: torch.device) -> DualTaskViSoBERT:
    model_cfg = cfg.get('model', {})
    model_name = model_cfg.get('name', '5CD-AI/visobert-14gb-corpus')
    num_aspects = int(model_cfg.get('num_aspects', 11))
    num_sentiments = int(model_cfg.get('num_sentiments', 3))
    hidden_size = int(model_cfg.get('hidden_size', 512))
    dropout = float(model_cfg.get('dropout', 0.3))

    model = DualTaskViSoBERT(
        model_name=model_name,
        num_aspects=num_aspects,
        num_sentiments=num_sentiments,
        hidden_size=hidden_size,
        dropout=dropout,
    )
    model.to(device)
    model.eval()
    return model


def build_dummy_batch_from_config(cfg: dict, tokenizer_name: str, device: torch.device, clamp_max_len: int):
    cfg_max_len = int(cfg.get('model', {}).get('max_length', 256))
    max_len = min(cfg_max_len, clamp_max_len)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    sample_text = "Pin trâu, camera xấu nhưng hiệu năng ổn"
    encoding = tokenizer(
        sample_text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    return input_ids, attention_mask, sample_text, max_len


def try_render_with_torchviz(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, out_path_no_ext: str, fmt: str, freeze_backbone: bool) -> bool:
    try:
        from torchviz import make_dot
    except Exception:
        return False

    if freeze_backbone:
        # Build smaller graph: stop gradients at BERT, only show MLP + heads
        with torch.no_grad():
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
        x = model.dropout(cls_output)
        x = model.dense(x)
        x = model.activation(x)
        x = model.dropout(x)
        aspect_logits = model.aspect_detection_head(x)
        sentiment_logits = model.sentiment_classification_head(x)
    else:
        # Full graph including backbone
        aspect_logits, sentiment_logits = model(input_ids, attention_mask)

    graph_anchor = aspect_logits.mean() + sentiment_logits.mean()

    dot = make_dot(
        graph_anchor,
        params=dict(model.named_parameters()),
        show_attrs=True,
        show_saved=True,
    )

    out_dir = os.path.dirname(out_path_no_ext)
    ensure_dir(out_dir)

    dot.render(out_path_no_ext, format=fmt, cleanup=True)
    return True


def export_onnx_fallback(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, out_onnx: str) -> None:
    ensure_dir(os.path.dirname(out_onnx))

    inputs = (input_ids, attention_mask)

    dynamic_axes = {
        'input_ids': {0: 'batch', 1: 'sequence'},
        'attention_mask': {0: 'batch', 1: 'sequence'},
        'aspect_logits': {0: 'batch'},
        'sentiment_logits': {0: 'batch', 1: 'num_aspects', 2: 'num_sentiments'},
    }

    class Wrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module):
            super().__init__()
            self.inner = inner
        def forward(self, input_ids, attention_mask):
            aspect_logits, sentiment_logits = self.inner(input_ids, attention_mask)
            return aspect_logits, sentiment_logits

    wrapper = Wrapper(model).eval()

    torch.onnx.export(
        wrapper,
        inputs,
        out_onnx,
        input_names=['input_ids', 'attention_mask'],
        output_names=['aspect_logits', 'sentiment_logits'],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )


def main():
    parser = argparse.ArgumentParser(description='Visualize Dual Task model graph to image')
    parser.add_argument('--config', type=str, default='config_multi.yaml', help='Path to YAML config')
    parser.add_argument('--output', type=str, default='analysis_results/model_graph', help='Output path without extension')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device selection')
    parser.add_argument('--max-len', type=int, default=32, help='Clamp tokenized max_length for manageable graph size')
    parser.add_argument('--format', type=str, default='svg', choices=['svg', 'png', 'pdf'], help='Graph output format')
    parser.add_argument('--freeze-backbone', action='store_true', help='Stop gradient at BERT to reduce graph size')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    cfg = load_yaml_config(args.config)

    model_name = cfg.get('model', {}).get('name', '5CD-AI/visobert-14gb-corpus')

    model = build_model_from_config(cfg, device)

    input_ids, attention_mask, sample_text, used_len = build_dummy_batch_from_config(
        cfg, model_name, device, clamp_max_len=args.max_len
    )

    out_no_ext = args.output
    rendered = try_render_with_torchviz(
        model,
        input_ids,
        attention_mask,
        out_no_ext,
        fmt=args.format,
        freeze_backbone=args.freeze_backbone,
    )

    if rendered:
        print(f"Saved graph to: {out_no_ext}.{args.format}")
    else:
        out_onnx = out_no_ext + '.onnx'
        export_onnx_fallback(model, input_ids, attention_mask, out_onnx)
        print("torchviz/graphviz not available. Exported ONNX graph instead.")
        print(f"Saved ONNX model to: {out_onnx}")
        print("You can visualize the ONNX with tools like Netron.")

    print(f"Sample text used for tokenization: {sample_text}")
    print(f"Effective max_length: {used_len}")


if __name__ == '__main__':
    main()
