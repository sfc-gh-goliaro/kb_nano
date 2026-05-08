"""Convert manual_audit_shard_*.md into the paper-appendix LaTeX row format:

  \\texttt{folder} & STATUS & \\texttt{op1}, \\texttt{op2}, ... & \\texttt{L1/}\\allowbreak\\texttt{file1.py}, ... \\\\

Canonical op names are L1-only (paper convention). L2 files are expanded to
their L1 imports (parsed from `from ..L1 import` lines). Sorted alphabetically.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path('/home/olu/kb_nano')
KB_ROOT = REPO / 'tasks/baseline'
SHARD_DIR = REPO / 'audits/hf_transformers_coverage/tools'
OUT_TEX = REPO / 'audits/hf_transformers_coverage/PAPER_APPENDIX_TABLE.tex'

STATUS_MARKER = {
    'kb_nano_l4': r'$\bullet$',
    'composable': r'\cmark',
    'partial': r'\textbf{P}',
    'unsupported': r'\xmark',
    'not_inference_required': '--',
}

# Canonical L1 op name for each kb-nano L1 file
L1_TO_CANONICAL = {
    'linear.py': 'linear',
    'embedding.py': 'embedding',
    'store_kvcache.py': 'kv_cache',
    'dense_attention.py': 'sdpa',
    'flash_attn_varlen.py': 'flash_attn_varlen',
    'flash_attn_decode.py': 'flash_attn_decode',
    'flash_attn_prefill.py': 'flash_attn_prefill',
    'rms_norm.py': 'rms_norm',
    'layer_norm.py': 'layer_norm',
    'group_norm.py': 'group_norm',
    'batch_norm2d.py': 'batch_norm_2d',
    'batch_norm1d.py': 'batch_norm_1d',
    'frozen_batch_norm2d.py': 'frozen_batch_norm',
    't5_layer_norm.py': 'rms_norm_t5',
    'bitnet_rms_norm.py': 'rms_norm_bitnet',
    'gemma_rms_norm.py': 'rms_norm_gemma',
    'rms_norm_gated.py': 'rms_norm_gated',
    'rotary_emb.py': 'rotary_pos_emb',
    'yarn_rotary_emb.py': 'rotary_pos_emb_yarn',
    'mrope.py': 'rotary_pos_emb_mrope',
    'dinov3_rope.py': 'rotary_pos_emb_dinov3',
    'vision_rotary_emb.py': 'rotary_pos_emb_vision',
    'sinusoidal_embed.py': 'sinusoidal_embed',
    'silu.py': 'silu',
    'silu_and_mul.py': 'silu_and_mul',
    'gelu.py': 'gelu',
    'gelu_and_mul.py': 'gelu_and_mul',
    'quickgelu.py': 'quick_gelu',
    'relu.py': 'relu',
    'squared_relu.py': 'squared_relu',
    'squared_relu_and_mul.py': 'squared_relu_and_mul',
    'tanh.py': 'tanh',
    'sigmoid.py': 'sigmoid',
    'softmax.py': 'softmax',
    'leaky_relu.py': 'leaky_relu',
    'log_sigmoid.py': 'log_sigmoid',
    'hardsigmoid.py': 'hardsigmoid',
    'conv1d.py': 'conv1d',
    'conv2d.py': 'conv2d',
    'conv3d.py': 'conv3d',
    'conv_transpose1d.py': 'conv_transpose_1d',
    'conv_transpose2d.py': 'conv_transpose_2d',
    'conv_transpose3d.py': 'conv_transpose_3d',
    'causal_conv1d.py': 'causal_conv1d',
    'max_pool2d.py': 'max_pool_2d',
    'avg_pool2d.py': 'avg_pool_2d',
    'avg_pool1d.py': 'avg_pool_1d',
    'adaptive_avg_pool2d.py': 'adaptive_avg_pool_2d',
    'adaptive_avg_pool1d.py': 'adaptive_avg_pool_1d',
    'global_avg_pool2d.py': 'global_avg_pool_2d',
    'dropout.py': 'dropout',
    'interpolate.py': 'interpolate',
    'tensor_ops.py': 'pad',
    'grn.py': 'grn',
    'moe_grouped_gemm.py': 'moe_grouped_gemm',
    'grouped_topk.py': 'moe_topk',
    'sigmoid_topk.py': 'sigmoid_topk',
    'topk_softmax.py': 'topk_softmax',
    'mxfp4_moe.py': 'mxfp4_moe',
    'rtdetrv2_deformable_attention.py': 'deformable_attention',
    'sparse_attn_indexer.py': 'sparse_attn_indexer',
    'rg_lru.py': 'rg_lru',
    'rwkv7_recurrence.py': 'rwkv7_recurrence',
    'lstm.py': 'lstm',
    # SSM / mamba
    'mamba_chunk_scan.py': 'mamba_scan',
    'mamba2_chunk_scan.py': 'mamba2_scan',
    'selective_scan.py': 'mamba_scan',
}

# Canonical name for L4 files commonly referenced
L4_TO_CANONICAL = {
    'recurrent_cache.py': 'encoder_decoder_cache',
}

# Manual L2 → L1 fallback for L2 files we can't parse
L2_FALLBACK = {
    'mamba_mixer.py': ['causal_conv1d.py', 'silu.py', 'linear.py'],  # + mamba_scan synthetic
    'mamba2_mixer.py': ['causal_conv1d.py', 'silu.py', 'linear.py', 'rms_norm_gated.py'],
}


def parse_l2_imports(path: Path) -> list[str]:
    """Parse a kb-nano L2/L3 file for `from ..L1 import` and `from ..L1.X import Y` lines."""
    if not path.exists():
        return []
    out = []
    text = path.read_text()
    # Match `from ..L1.<file> import <Cls>` (single L1 file)
    for m in re.finditer(r'from\s+\.\.L1\.([a-zA-Z0-9_]+)\s+import', text):
        out.append(m.group(1) + '.py')
    # Match `from ..L1 import (X, Y, Z)` — we'd need to map class names back; skip for now
    return out


def build_l2_l1_map() -> dict:
    """For each L2 file, list its L1 imports."""
    m = {}
    for f in (KB_ROOT / 'L2').glob('*.py'):
        m[f.name] = parse_l2_imports(f)
    for f in (KB_ROOT / 'L3').glob('*.py'):
        m[f.name] = parse_l2_imports(f)
    return m


def status_marker(s: str) -> str:
    return STATUS_MARKER.get(s, r'\cmark')


def le(s: str) -> str:
    return s.replace('_', r'\_').replace('&', r'\&')


def fmt_kb_path(path: str) -> str:
    if '/' not in path:
        return r'\texttt{' + le(path) + '}'
    layer, fname = path.split('/', 1)
    return r'\texttt{' + le(layer) + r'/}\allowbreak\texttt{' + le(fname) + '}'


def parse_md(text: str) -> list[dict]:
    folders = []
    current = None
    for line in text.split('\n'):
        if line.startswith('## '):
            if current:
                folders.append(current)
            current = {'folder': line[3:].strip(), 'status': 'composable', 'paths': []}
            continue
        if not current:
            continue
        m = re.match(r'^- \*\*status\*\*:\s*([\w_]+)', line)
        if m:
            current['status'] = m.group(1)
            continue
        for p in re.findall(r'(L\d/[A-Za-z0-9_]+\.py)', line):
            current['paths'].append(p)
    if current:
        folders.append(current)
    return folders


def expand_to_l1(paths: list[str], l2l1: dict) -> set[str]:
    """For each path: keep L1 as-is, expand L2/L3 to their L1 imports.
       L4 paths kept as-is (will be handled separately)."""
    out = set()
    for p in paths:
        layer, fname = p.split('/', 1)
        if layer == 'L1':
            out.add(p)
        elif layer in ('L2', 'L3'):
            # Try to expand via L2_L1 map
            l1_files = l2l1.get(fname, [])
            if not l1_files:
                # Fallback for known
                l1_files = L2_FALLBACK.get(fname, [])
            for f in l1_files:
                out.add(f'L1/{f}')
            # Special-case mamba L2 files: add mamba_scan as a synthetic op via L2 file path
            if 'mamba2' in fname:
                out.add('L2/mamba2_mixer.py')  # no L1 mamba scan
            elif 'mamba' in fname:
                out.add('L2/mamba_mixer.py')
        elif layer == 'L4':
            # Some L4 files have canonical names (e.g., recurrent_cache → encoder_decoder_cache)
            if fname in L4_TO_CANONICAL:
                out.add(p)
    return out


def to_canonical(path: str) -> str | None:
    """Return canonical paper-appendix name for an L1/L2/L4 path, or None to skip."""
    layer, fname = path.split('/', 1)
    if layer == 'L1':
        return L1_TO_CANONICAL.get(fname)
    if layer == 'L4' and fname in L4_TO_CANONICAL:
        return L4_TO_CANONICAL[fname]
    if layer == 'L2':
        # Special-case mamba (no L1 selective-scan kernel)
        if 'mamba2' in fname:
            return 'mamba2_scan'
        if 'mamba' in fname:
            return 'mamba_scan'
    return None


def render_row(folder: dict, l2l1: dict) -> str:
    expanded = expand_to_l1(folder['paths'], l2l1)
    pairs = []
    for p in expanded:
        c = to_canonical(p)
        if c:
            pairs.append((c, p))
    seen = {}
    for c, p in pairs:
        if c not in seen:
            seen[c] = p
    sorted_pairs = sorted(seen.items())
    canonicals = [c for c, _ in sorted_pairs]
    files = [p for _, p in sorted_pairs]
    if not canonicals:
        return (r'  \texttt{' + le(folder['folder']) + r'} & '
                + status_marker(folder['status']) + r' & --- & --- \\')
    op_str = ', '.join(r'\texttt{' + le(c) + '}' for c in canonicals)
    file_str = ', '.join(fmt_kb_path(p) for p in files)
    return (r'  \texttt{' + le(folder['folder']) + r'} & '
            + status_marker(folder['status'])
            + ' & ' + op_str + ' & ' + file_str + r' \\')


def load_reclassifications() -> dict:
    """Read reclassify_A.md and reclassify_B.md and return {folder: new_status}."""
    out = {}
    for letter in ('A', 'B'):
        p = SHARD_DIR / f'reclassify_{letter}.md'
        if not p.exists():
            continue
        for line in p.read_text().split('\n'):
            m = re.match(r'^([a-z][a-z0-9_]*):\s*(composable|partial|unsupported|kb_nano_l4)\s*[—-]', line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def main():
    l2l1 = build_l2_l1_map()
    reclass = load_reclassifications()
    all_folders = []
    for i in range(1, 17):
        p = SHARD_DIR / f'manual_audit_shard_{i:02d}.md'
        if not p.exists():
            continue
        folders = parse_md(p.read_text())
        all_folders.extend(folders)
    # Apply reclassifications (looser paper-definition status)
    n_reclassified = 0
    for f in all_folders:
        new_status = reclass.get(f['folder'])
        if new_status and new_status != f['status']:
            f['status'] = new_status
            n_reclassified += 1
    print(f'Reclassified {n_reclassified} folders to looser paper-definition status')
    all_folders.sort(key=lambda f: f['folder'])
    rows = []
    for f in all_folders:
        rows.append(render_row(f, l2l1))
    OUT_TEX.write_text('\n'.join(rows) + '\n')
    print(f'Wrote {len(rows)} rows to {OUT_TEX}')


if __name__ == '__main__':
    main()
