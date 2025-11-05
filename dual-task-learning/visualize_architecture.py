import os
import argparse
import yaml

try:
    from graphviz import Digraph
except Exception as e:
    Digraph = None


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def human(label: str, width: str = '1.4', height: str = '0.8'):
    return dict(label=label, shape='box', style='rounded,filled', color='#2F4F4F', fillcolor='#E8F5E9', fontname='Inter,Helvetica,Arial', fontsize='12', width=width, height=height)


def component(label: str, fill: str = '#E3F2FD'):
    return dict(label=label, shape='box', style='rounded,filled', color='#1E88E5', fillcolor=fill, fontname='Inter,Helvetica,Arial', fontsize='12')


def head(label: str, color: str = '#8E24AA', fill: str = '#F3E5F5'):
    return dict(label=label, shape='box', style='rounded,filled', color=color, fillcolor=fill, fontname='Inter,Helvetica,Arial', fontsize='12')


def make_architecture_graph(cfg: dict, out_no_ext: str, fmt: str = 'svg', orientation: str = 'TB'):
    if Digraph is None:
        raise RuntimeError('graphviz python package not available. Install with: pip install graphviz')

    model_cfg = cfg.get('model', {})
    model_name = model_cfg.get('name', '5CD-AI/visobert-14gb-corpus')
    hidden_size = int(model_cfg.get('hidden_size', 512))
    num_aspects = int(model_cfg.get('num_aspects', 11))
    num_sentiments = int(model_cfg.get('num_sentiments', 3))
    max_length = int(model_cfg.get('max_length', 256))

    dot = Digraph('DualTaskArchitecture', format=fmt)
    # orientation: 'TB' (top->bottom) or 'LR' (left->right)
    dot.attr(rankdir=orientation, splines='spline', nodesep='0.4', ranksep='0.6', bgcolor='white')

    # Inputs
    dot.node('input_text', **human('Input Text\n(Vietnamese review)'))
    dot.node('tokenizer', **component(f'Tokenizer\n({model_name})'))

    # Backbone
    dot.node('bert', **component(f'BERT Encoder\n{model_name}\nOutput: [B, L, H_bert]', fill='#E1F5FE'))
    dot.node('cls', **component('CLS Pooling\n[H_bert]'))

    # Shared features
    dot.node('dense', **component(f'Dense + ReLU + Dropout\nHidden: {hidden_size}\n[H_shared]', fill='#FFF3E0'))

    # Heads
    dot.node('ad_head', **head(f'Aspect Detection Head\nLinear: {hidden_size} → {num_aspects}\nLogits: [B, {num_aspects}]', color='#43A047', fill='#E8F5E9'))
    dot.node('sc_head', **head(f'Sentiment Head\nLinear: {hidden_size} → {num_aspects*num_sentiments}\nReshape: [B, {num_aspects}, {num_sentiments}]', color='#FB8C00', fill='#FFF3E0'))

    # Outputs
    dot.node('ad_post', **component('Sigmoid → Aspect presence\n[B, aspects]', fill='#F1F8E9'))
    dot.node('sc_post', **component('Softmax → Sentiment per aspect\n[B, aspects, 3]', fill='#FFFDE7'))

    # Edges
    dot.edge('input_text', 'tokenizer', label='tokenize\nmax_length=' + str(max_length), fontsize='10', color='#546E7A')
    dot.edge('tokenizer', 'bert', label='input_ids, attention_mask', fontsize='10', color='#546E7A')
    dot.edge('bert', 'cls', label='take [CLS]', fontsize='10', color='#546E7A')
    dot.edge('cls', 'dense', label='shared features', fontsize='10', color='#546E7A')

    dot.edge('dense', 'ad_head', label='[H_shared]', fontsize='10', color='#2E7D32')
    dot.edge('dense', 'sc_head', label='[H_shared]', fontsize='10', color='#EF6C00')

    dot.edge('ad_head', 'ad_post', label='sigmoid', fontsize='10', color='#2E7D32')
    dot.edge('sc_head', 'sc_post', label='softmax (last dim)', fontsize='10', color='#EF6C00')

    # Legend/Notes
    with dot.subgraph(name='cluster_legend') as c:
        c.attr(label='Notes', fontsize='12', fontname='Inter,Helvetica,Arial', color='#BDBDBD')
        c.node('note1', **component('B = batch size, L = sequence length, H_bert = hidden size of BERT', fill='white'))
        c.node('note2', **component('Aspects: 11  |  Sentiments: 3 (pos/neg/neu)', fill='white'))
        c.node('note3', **component('Outputs: AD uses sigmoid; SC uses softmax over 3 sentiments', fill='white'))
        c.attr(style='dashed', color='#CFD8DC')

    out_dir = os.path.dirname(out_no_ext)
    ensure_dir(out_dir)
    dot.render(out_no_ext, cleanup=True)
    return f"{out_no_ext}.{fmt}"


def main():
    parser = argparse.ArgumentParser(description='Render a simple Dual-Task architecture diagram')
    parser.add_argument('--config', type=str, default='config_multi.yaml', help='Path to YAML config')
    parser.add_argument('--output', type=str, default='analysis_results/model_architecture', help='Output path without extension')
    parser.add_argument('--format', type=str, default='svg', choices=['svg', 'png', 'pdf'], help='Output format')
    parser.add_argument('--orientation', type=str, default='TB', choices=['TB', 'LR'], help='Diagram flow: TB=top-bottom, LR=left-right')
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    out_path = make_architecture_graph(cfg, args.output, fmt=args.format, orientation=args.orientation)
    print(f"Saved architecture diagram to: {out_path}")


if __name__ == '__main__':
    main()
