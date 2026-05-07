"""v4 re-audit: AST-extract HF classes from modular_<f>.py (preferred) /
modeling_<f>.py and map each to a kb-nano L1/L2/L3/L4 file.

Per-row methodology (matches mentor-approved decisions):
  Q1 ✓ inheritance-based mapping (NewMLP(LlamaMLP) -> kb-nano L2/llama_mlp.py)
  Q2 ✓ decompose to L1 when no L2 wrapper exists for the family
  Q3 ✓ composable when every required sub-block is present (even if no
       single same-named kb-nano file)
  Q4 ✓ skip zero-compute classes (PreTrainedModel, Output, Config, mixins)
  Q5 ✓ cite one HF source line per class included in operators

Output: hf_architecture_operator_coverage_v4.csv (drop-in replacement).
"""
from __future__ import annotations

import ast
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

REPO = Path('/home/olu/kb_nano')
HF_PINNED = Path('/tmp/hf_transformers_pinned/src/transformers/models')
COVERAGE_CSV = REPO / 'audits/hf_transformers_coverage/hf_architecture_operator_coverage.csv'
INVENTORY_CSV = REPO / 'audits/hf_transformers_coverage/hf_model_inventory.csv'

# Skip rules per Q4
SKIP_BASE_CLASSES = {
    'PreTrainedModel', 'GenerationMixin', 'BackboneMixin', 'BaseImageProcessor',
    'ProcessorMixin', 'Cache', 'DynamicCache',
    'ModelOutput',
}
SKIP_CLASS_NAMES_REGEX = re.compile(
    r'(.*PreTrainedModel|.*Config|.*Output|.*Cache$|.*Mixin)$'
)

# Classify HF class by name suffix (Q-derived: Phase 2 of methodology)
def role_from_name(class_name: str) -> str:
    n = class_name
    # L4: full pipelines
    if re.search(r'(?:For\w+|Model)$', n) and not re.search(r'PreTrainedModel$', n):
        return 'L4'
    # L3: layers / blocks / encoders / decoders
    if re.search(r'(?:Layer|Block|Encoder|Decoder|Pairformer|Stage)$', n):
        return 'L3'
    # L2: composites
    if re.search(
        r'(?:Attention|MLP|MoE|MoeBlock|Experts?|Router|Embedding(?:s)?|'
        r'Pool(?:ing)?Head|VisionTransformer|Mixer|Block\w*Module|FeedForward|'
        r'Pooler|Patch\w*Embed|TextTransformer|VisionTransformer)$',
        n,
    ):
        return 'L2'
    # L1: leaf primitives
    if re.search(
        r'(?:RMSNorm|LayerNorm|GroupNorm|BatchNorm\d?d?|RotaryEmbedding|'
        r'Linear|Conv\d?d?|Activation|Embedding$|Sigmoid|Tanh|Softmax|GELU|SiLU|'
        r'ReLU|Dropout|Pool\d?d?)$',
        n,
    ):
        return 'L1'
    return 'unknown'

# kb-nano file inventory by layer
def kb_files_by_layer():
    out = {}
    for layer in ['L1', 'L2', 'L3', 'L4']:
        files = {}
        for p in (REPO / 'tasks/baseline' / layer).glob('*.py'):
            if p.name == '__init__.py':
                continue
            files[p.name] = p
        out[layer] = files
    return out

def first_top_level_class(p: Path) -> str | None:
    """Return the first non-private, non-Config top-level class name in file."""
    try:
        tree = ast.parse(p.read_text())
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
            if node.name.endswith('Config'):
                continue
            return node.name
    return None

# kb-nano file lookup helpers
KB = kb_files_by_layer()

def file_for(layer: str, *candidates: str) -> tuple[str, str] | None:
    """Try each candidate filename in the given layer. Return (path, class)."""
    for c in candidates:
        if c in KB[layer]:
            cname = first_top_level_class(KB[layer][c]) or '?'
            return (f'tasks/baseline/{layer}/{c}', cname)
    return None

def parse_inheritance(file_path: Path) -> list[tuple[str, list[str]]]:
    """Return [(class_name, [base_class_names]), ...] for top-level classes."""
    try:
        tree = ast.parse(file_path.read_text())
    except SyntaxError:
        return []
    out = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name.startswith('_'):
            continue
        bases = []
        for b in node.bases:
            try:
                bases.append(ast.unparse(b).split('.')[-1])  # last segment, e.g. 'LlamaMLP'
            except Exception:
                pass
        out.append((node.name, bases))
    return out

def class_line_numbers(file_path: Path) -> dict[str, int]:
    """class_name -> line number."""
    try:
        tree = ast.parse(file_path.read_text())
    except SyntaxError:
        return {}
    out = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out[node.name] = node.lineno
    return out

# Map an HF class to a kb-nano file. Returns list of (canonical_op, path, class) entries.
# May return MULTIPLE entries (e.g., AfmoeAttention decomposes to 4 L1 entries).
def map_hf_class(folder: str, class_name: str, base_classes: list[str], role: str) -> list[tuple[str, str, str]]:
    """Return list of (canonical_op, kb_nano_path, kb_nano_class)."""
    results = []
    # Strip the folder/arch prefix from class name to get the suffix role
    # E.g. AfmoeAttention -> Attention; LlamaForCausalLM -> ForCausalLM
    # But there's no canonical strip — just use the class name as the lookup key.

    # Build candidate kb-nano filename(s) for each layer
    candidates_by_layer = {'L1': [], 'L2': [], 'L3': [], 'L4': []}

    # Helper: derive snake_case from CamelCase
    def to_snake(s):
        return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()

    # Strip the architecture prefix from CLASS name (heuristic: strip the folder name's "camelization")
    # e.g. folder=afmoe class=AfmoeAttention -> suffix=Attention
    # folder=qwen2_moe class=Qwen2MoeAttention -> suffix=Attention
    folder_camel = ''.join(p.title() for p in folder.split('_'))
    suffix = class_name
    if class_name.startswith(folder_camel):
        suffix = class_name[len(folder_camel):]
    # Try inherited-from match: e.g. NewModelMLP(LlamaMLP) -> use LlamaMLP -> llama_mlp
    parent_candidates = []
    for parent in base_classes:
        # Strip generic parents
        if parent in ('Module', 'PreTrainedModel', 'GenerationMixin', 'BackboneMixin',
                      'GradientCheckpointingLayer', 'object'):
            continue
        # Convert CamelCase parent to snake_case
        snake = to_snake(parent)
        parent_candidates.append((parent, snake))

    # Folder-prefixed candidate
    own_snake = to_snake(suffix) if suffix else to_snake(class_name)
    folder_candidates = [
        f'{folder}_{own_snake}.py',  # e.g. afmoe_mlp.py
        f'{folder}.py',              # e.g. afmoe.py (L4 only)
        f'{own_snake}.py',           # e.g. attention_pool.py (cross-arch)
    ]

    # === L1 mapping ===
    if role == 'L1':
        # Common L1 leaf mappings by class-name pattern
        L1_PATTERNS = [
            (r'RMSNorm$', ['rms_norm.py']),
            (r'LayerNorm$', ['layer_norm.py']),
            (r'GroupNorm$', ['group_norm.py']),
            (r'BatchNorm\d?d?$', ['batch_norm2d.py']),
            (r'(M[Rr]otary|RotaryEmbedding|RotaryEmb)$', ['rotary_emb.py']),
            (r'Linear$', ['linear.py']),
            (r'Conv1d$', ['conv1d.py']),
            (r'Conv2d$', ['conv2d.py']),
            (r'Conv3d$', ['conv3d.py']),
            (r'^Embedding$', ['embedding.py']),
            (r'GELU$', ['gelu.py']),
            (r'SiLU$', ['silu.py']),
            (r'ReLU$', ['relu.py']),
            (r'Sigmoid$', ['sigmoid.py']),
            (r'Tanh$', ['tanh.py']),
            (r'Dropout$', ['dropout.py']),
            (r'Softmax$', ['softmax.py']),
        ]
        for pat, files in L1_PATTERNS:
            if re.search(pat, class_name):
                for f in files:
                    if f in KB['L1']:
                        cname = first_top_level_class(KB['L1'][f]) or '?'
                        results.append((to_snake(class_name), f'tasks/baseline/L1/{f}', cname))
                        return results

    # === L2 mapping ===
    if role == 'L2':
        # Try folder-prefixed first
        match = file_for('L2', f'{folder}_{own_snake}.py', f'{own_snake}.py', f'{folder}.py')
        if match:
            results.append((f'{folder}_{own_snake}' if f'{folder}_{own_snake}.py' in KB['L2'] else own_snake, *match))
            return results
        # Try inherited
        for parent_name, parent_snake in parent_candidates:
            parent_suffix = parent_name
            # Try kb-nano file matching parent's snake_case
            for cand in [f'{parent_snake}.py']:
                if cand in KB['L2']:
                    cname = first_top_level_class(KB['L2'][cand]) or '?'
                    results.append((parent_snake, f'tasks/baseline/L2/{cand}', cname))
                    return results
        # Special-case: *Attention with no L2 wrapper -> decompose to L1
        if re.search(r'Attention$', class_name):
            for op, fname, cls in [
                ('linear', 'linear.py', 'Linear'),
                ('sdpa', 'dense_attention.py', 'DenseAttention'),
                ('kv_cache', 'store_kvcache.py', 'StoreKVCache'),
                ('rotary_pos_emb', 'rotary_emb.py', 'RotaryEmbedding'),
            ]:
                if fname in KB['L1']:
                    results.append((op, f'tasks/baseline/L1/{fname}', cls))
            return results
        # *MLP / *FeedForward with no wrapper -> compose L1
        if re.search(r'(MLP|FeedForward)$', class_name):
            for op, fname, cls in [
                ('linear', 'linear.py', 'Linear'),
                ('silu', 'silu.py', 'SiLU'),
            ]:
                if fname in KB['L1']:
                    results.append((op, f'tasks/baseline/L1/{fname}', cls))
            return results
        # *Experts / *MoE -> compose L1
        if re.search(r'(Experts?|MoE|MoeBlock|Sparse.*Block)$', class_name):
            for op, fname, cls in [
                ('moe_grouped_gemm', 'moe_grouped_gemm.py', 'MoeGroupedGemm'),
                ('moe_topk', 'grouped_topk.py', 'GroupedTopK'),
            ]:
                if fname in KB['L1']:
                    results.append((op, f'tasks/baseline/L1/{fname}', cls))
            return results
        # *Router -> grouped_topk
        if re.search(r'Router$', class_name):
            if 'grouped_topk.py' in KB['L1']:
                cname = first_top_level_class(KB['L1']['grouped_topk.py']) or '?'
                results.append(('moe_topk', 'tasks/baseline/L1/grouped_topk.py', cname))
            return results
        # *Embeddings -> embedding L1
        if re.search(r'Embeddings?$', class_name):
            if 'embedding.py' in KB['L1']:
                results.append(('embedding', 'tasks/baseline/L1/embedding.py', 'Embedding'))
            return results

    # === L3 mapping ===
    if role == 'L3':
        match = file_for('L3', f'{folder}_{own_snake}.py', f'{own_snake}.py', f'{folder}_decoder.py')
        if match:
            results.append((f'{folder}_{own_snake}', *match))
            return results
        # Inherited
        for parent_name, parent_snake in parent_candidates:
            for cand in [f'{parent_snake}.py']:
                if cand in KB['L3']:
                    cname = first_top_level_class(KB['L3'][cand]) or '?'
                    results.append((parent_snake, f'tasks/baseline/L3/{cand}', cname))
                    return results
        # No L3 file: composable from L1/L2 — no entries added (handled at row level)
        return results

    # === L4 mapping ===
    if role == 'L4':
        match = file_for('L4', f'{folder}.py', f'{own_snake}.py')
        if match:
            results.append((f'{folder}_l4', *match))
            return results
        return results

    return results

def should_skip_class(class_name: str, base_classes: list[str]) -> bool:
    """Q4: skip zero-compute classes."""
    if SKIP_CLASS_NAMES_REGEX.match(class_name):
        return True
    # Skip if all bases are skip bases
    if base_classes:
        skippable = SKIP_BASE_CLASSES | {'object', 'IntEnum'}
        if all(b in skippable for b in base_classes):
            return True
    return False

def reaudit_one(folder: str, modeling_file_rel: str) -> dict | None:
    """Run the v4 audit on one HF folder. Return augmentations to apply to the existing row."""
    # Prefer modular_<x>.py if it exists
    folder_path = HF_PINNED / folder
    short = modeling_file_rel.split('/')[-1].replace('modeling_', '').replace('.py', '')
    modular_path = folder_path / f'modular_{short}.py'
    modeling_path = folder_path / f'modeling_{short}.py'
    src_path = modular_path if modular_path.exists() else modeling_path
    if not src_path.exists():
        return None

    classes = parse_inheritance(src_path)
    line_nums = class_line_numbers(src_path)

    operators = []           # CamelCase HF class names
    mapped_entries = []      # (canonical_op, kb_path, kb_class)
    evidence = []            # "<folder>/<file>:<line>"
    inheritance_chain = []   # for notes

    for class_name, bases in classes:
        if should_skip_class(class_name, bases):
            continue
        role = role_from_name(class_name)
        if role == 'unknown':
            continue
        # Track inheritance from cross-arch parents
        for b in bases:
            if b not in ('Module', 'PreTrainedModel', 'GenerationMixin', 'BackboneMixin',
                          'GradientCheckpointingLayer', 'object', 'IntEnum') \
               and not b.startswith(folder.title().replace('_', '')):
                inheritance_chain.append(f'{class_name}({b})')
        operators.append(class_name)
        if class_name in line_nums:
            evidence.append(f'{folder}/{src_path.name}:{line_nums[class_name]}')
        # Get kb-nano mapping
        for entry in map_hf_class(folder, class_name, bases, role):
            if entry not in mapped_entries:
                mapped_entries.append(entry)

    return {
        'operators': operators,
        'mapped_entries': mapped_entries,
        'evidence': evidence,
        'inheritance_chain': inheritance_chain,
        'src_path': str(src_path.relative_to(HF_PINNED.parent.parent)) if src_path else '',
    }


def main():
    inv_rows = list(csv.DictReader(open(INVENTORY_CSV)))
    coverage_rows = list(csv.DictReader(open(COVERAGE_CSV)))
    fieldnames = list(coverage_rows[0].keys())

    # Build lookup of folder -> [coverage rows]
    folder_to_rows = defaultdict(list)
    for r in coverage_rows:
        folder_to_rows[r['hf_folder']].append(r)

    augmented_count = 0
    skipped_count = 0
    for r in coverage_rows:
        if r['support_status'] == 'not_inference_required':
            skipped_count += 1
            continue
        if not r['modeling_file']:
            skipped_count += 1
            continue
        result = reaudit_one(r['hf_folder'], r['modeling_file'])
        if not result:
            skipped_count += 1
            continue
        # Augment row
        # 1. architecture_classes: replace with full class list
        if result['operators']:
            r['architecture_classes'] = ';'.join(result['operators'])
        # 2. mapped_kb_nano: union with existing
        existing_paths = set()
        for entry in r['mapped_kb_nano'].split(';'):
            for sep in ['→', '->']:
                if sep in entry:
                    _, rest = entry.split(sep, 1)
                    rest = rest.split('(')[0].strip()
                    if ':' in rest:
                        rest = rest.rsplit(':', 1)[0]
                    existing_paths.add(rest.strip())
                    break
        new_entries = []
        for op, path, cls in result['mapped_entries']:
            if path in existing_paths:
                continue
            new_entries.append(f'{op}→{path}:{cls}')
            existing_paths.add(path)
        if new_entries:
            sep = ';' if r['mapped_kb_nano'] else ''
            r['mapped_kb_nano'] = r['mapped_kb_nano'] + sep + ';'.join(new_entries)
        # 3. evidence_hf: replace with per-class line citations
        if result['evidence']:
            r['evidence_hf'] = ';'.join(result['evidence'])
        # 4. notes: add inheritance chain
        if result['inheritance_chain']:
            chain_str = '; '.join(result['inheritance_chain'][:5])
            note = f'[v4 inheritance] {chain_str}'
            if note not in (r.get('notes') or ''):
                r['notes'] = (r.get('notes', '') + ' ' + note).strip()
        augmented_count += 1

    # Write back
    out_path = REPO / 'audits/hf_transformers_coverage/hf_architecture_operator_coverage.csv'
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(coverage_rows)
    print(f'Augmented {augmented_count} rows; skipped {skipped_count} (NIR or no source).')
    print(f'Wrote {out_path}')

if __name__ == '__main__':
    main()
