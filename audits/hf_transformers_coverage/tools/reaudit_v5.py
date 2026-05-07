"""v5 re-audit: per-HF-class kb-nano kernel mapping with variant detection.

Methodology baked in (mentor-approved + LESSONS_LEARNED.md):
  - Read modular_<f>.py preferred, modeling_<f>.py fallback.
  - For each non-skip class, determine kb-nano mapping using:
    (1) direct prefix: <folder>_<role>.py
    (2) inherited from another HF arch: <parent>_<role>.py
    (3) variant-specific: bitnet_rms_norm, t5_layer_norm, yarn_rotary_emb,
        mrope, mxfp4_moe, deepseek_mla_attention, encoder_attention, etc.
    (4) generic fallback: rms_norm.py, layer_norm.py, etc.
  - Sibling enumeration: ls all candidate filenames in each layer dir;
    prefer the most specific match.
  - Wiring classes (DecoderLayer / Model / ForXxx): map to L3/L4 if exists,
    else mark "composes (wiring)".
  - Verdicts:
    * kb_nano_l4 if L4/<folder>.py exists
    * composable if every kernel exists
    * partial / unsupported preserved from existing CSV (manual verdicts)

Output: rewrites the CSV with per-class arrows in mapped_kb_nano column.
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

# Skip rules (zero-compute classes per Q4)
# Skip a class if its name matches SKIP_NAME_RE (boilerplate/output containers).
# Skip if ALL bases are in SKIP_BASES_ONLY (pure boilerplate parents).
# Note: Module / GradientCheckpointingLayer are NOT in SKIP_BASES because
# legitimate compute classes inherit from them (e.g. AfmoeAttention(Module),
# BertLayer(GradientCheckpointingLayer)). They are filtered when listing the
# "inheritance chain" notes but never cause the class itself to be skipped.
SKIP_NAME_RE = re.compile(r'^(.*PreTrainedModel|.*Config|.*Output|.*Cache$|.*Mixin|.*Embedder|.*Tokenizer)$')
SKIP_BASES_ONLY = {'PreTrainedModel', 'GenerationMixin', 'BackboneMixin', 'BaseImageProcessor',
                    'ProcessorMixin', 'Cache', 'DynamicCache', 'ModelOutput', 'IntEnum'}
# Bases that contribute no compute info but DON'T force-skip the child class
NEUTRAL_BASES = SKIP_BASES_ONLY | {'Module', 'object', 'GradientCheckpointingLayer'}

# kb-nano file inventory by layer
KB_FILES = {layer: {p.name: p for p in (REPO / 'tasks/baseline' / layer).glob('*.py') if p.name != '__init__.py'}
            for layer in ('L1', 'L2', 'L3', 'L4')}


def kb_path(layer, fname):
    return f'tasks/baseline/{layer}/{fname}'


def has_kb(layer, fname):
    return fname in KB_FILES[layer]


def to_snake(s):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()


def folder_camel(folder):
    return ''.join(p.title() for p in folder.split('_'))


def find_kb_file(layer, *candidates):
    """Return first kb-nano filename in `layer` matching any candidate."""
    for c in candidates:
        if has_kb(layer, c):
            return c
    return None


# Variant detection: HF class name + base classes -> kb-nano file path(s)
def map_class(folder, class_name, bases, hf_src_path=None):
    """Return list of kb_path strings (kernels the class's compute invokes)."""
    out = []
    fc = folder_camel(folder)
    suffix = class_name[len(fc):] if class_name.startswith(fc) else class_name
    folder_prefix = folder + '_'

    # ---- Strip folder/arch prefix to get role ----
    name = class_name
    suffix_snake = to_snake(suffix) if suffix else to_snake(class_name)

    # =========================================================
    # L1 LEAF PRIMITIVES (variant-specific routing)
    # =========================================================

    # RMSNorm variants
    if re.search(r'RMSNorm$', name):
        # Look for arch-specific: bitnet_rms_norm.py, gemma_rms_norm.py, etc.
        f = find_kb_file('L1', f'{folder}_rms_norm.py')
        if f: out.append(kb_path('L1', f)); return out
        # T5 norm style (no centering) -> t5_layer_norm
        # GptOss/Llama -> standard rms_norm
        for parent in bases:
            ps = to_snake(parent)
            f = find_kb_file('L1', f'{ps}.py')
            if f and ('rms' in f or 'layer_norm' in f):
                out.append(kb_path('L1', f)); return out
        out.append(kb_path('L1', 'rms_norm.py'))
        return out

    # T5LayerNorm (special: no centering, uses t5_layer_norm.py)
    if re.search(r'T5LayerNorm$|T5Norm$', name):
        out.append(kb_path('L1', 't5_layer_norm.py'))
        return out

    # LayerNorm
    if re.search(r'LayerNorm$', name):
        out.append(kb_path('L1', 'layer_norm.py'))
        return out

    if re.search(r'GroupNorm$', name):
        out.append(kb_path('L1', 'group_norm.py'))
        return out

    if re.search(r'BatchNorm', name):
        out.append(kb_path('L1', 'batch_norm2d.py'))
        return out

    # FrozenBatchNorm
    if re.search(r'FrozenBatchNorm', name):
        out.append(kb_path('L1', 'frozen_batch_norm2d.py'))
        return out

    # RotaryEmbedding variants — order matters
    if re.search(r'(Rotary|Rope|RoPE)(Embedding|Emb)?$', name):
        # M-RoPE for multimodal Qwen
        if 'MRotary' in name or re.search(r'MRoPE', name, re.I):
            out.append(kb_path('L1', 'mrope.py')); return out
        # Vision RoPE
        if 'Vision' in name and 'Rotary' in name:
            out.append(kb_path('L1', 'vision_rotary_emb.py')); return out
        # Check inheritance for YaRN
        for parent in bases:
            if 'Yarn' in parent or 'YaRN' in parent:
                out.append(kb_path('L1', 'yarn_rotary_emb.py')); return out
        # Arch-specific RoPE (dinov3, flux, oasis, sam3, hunyuan_video, diffusion)
        for arch_name in ('dinov3_rope', 'flux_pos_embed', 'oasis_rotary',
                          'sam3_rope', 'hunyuan_video_rope', 'diffusion_rope',
                          'vjepa2_rope', 'ttt_e2e_rope'):
            if folder.startswith(arch_name.split('_')[0]) and has_kb('L1', f'{arch_name}.py'):
                out.append(kb_path('L1', f'{arch_name}.py')); return out
        # Default
        out.append(kb_path('L1', 'rotary_emb.py'))
        return out

    if name == 'Embedding' or name.endswith('Embedding') and 'Rotary' not in name and 'Pos' not in name:
        # Could be arch-specific, but most are just Embedding
        out.append(kb_path('L1', 'embedding.py'))
        return out

    # Activations
    L1_ACTS = [
        (r'^GELU$|^GeLU$|GELU$', 'gelu.py'),
        (r'^SiLU$|SiLU$', 'silu.py'),
        (r'^ReLU$', 'relu.py'),
        (r'^Sigmoid$', 'sigmoid.py'),
        (r'^Tanh$', 'tanh.py'),
        (r'^QuickGELU$|QuickGELU$', 'quickgelu.py'),
        (r'^Mish$', 'mish.py'),
        (r'^Softmax$', 'softmax.py'),
        (r'^Softplus$', 'softplus.py'),
        (r'^LogSigmoid$', 'log_sigmoid.py'),
        (r'^Dropout$', 'dropout.py'),
        (r'^LeakyReLU$', 'leaky_relu.py'),
        (r'^ELU$', 'elu.py'),
        (r'^Hardsigmoid$|^HardSigmoid$', 'hardsigmoid.py'),
        (r'^Hardswish$|^HardSwish$', 'hardswish.py'),
    ]
    for pat, fname in L1_ACTS:
        if re.search(pat, name):
            if has_kb('L1', fname):
                out.append(kb_path('L1', fname))
                return out

    # Linear variants (BitNet, FP8)
    if re.search(r'^Linear$|Linear$', name) and 'KVCache' not in name and 'Project' not in name:
        if 'BitNet' in name or 'BitLinear' in name:
            out.append(kb_path('L1', 'bitnet_linear.py')); return out
        if 'FP8' in name.upper():
            out.append(kb_path('L1', 'fp8_linear.py')); return out
        out.append(kb_path('L1', 'linear.py'))
        return out

    # Conv variants
    for nd in ('1', '2', '3'):
        if re.search(rf'^Conv{nd}d$|Conv{nd}d$', name):
            out.append(kb_path('L1', f'conv{nd}d.py')); return out
        if re.search(rf'^ConvTranspose{nd}d$', name):
            out.append(kb_path('L1', f'conv_transpose{nd}d.py')); return out

    # =========================================================
    # L2 COMPOSITES (variant-specific routing)
    # =========================================================

    # Routers
    if re.search(r'Router$', name):
        # Distinguish topk-softmax (DeepSeek) vs sigmoid (Kimi/Qwen3Next/Afmoe)
        f = find_kb_file('L1', 'sigmoid_topk.py')
        if f: out.append(kb_path('L1', f))
        out.append(kb_path('L1', 'linear.py'))  # router gate is a Linear
        return out

    # Experts (the kernel doing the actual MoE compute)
    if re.search(r'Experts$', name):
        # MXFP4 quantized?
        if 'MXFP4' in name.upper() or 'MxFp4' in name or folder == 'gpt_oss':
            f = find_kb_file('L1', 'mxfp4_moe.py')
            if f: out.append(kb_path('L1', f)); return out
        out.append(kb_path('L1', 'moe_grouped_gemm.py'))
        return out

    # MoE block (router + experts + output)
    if re.search(r'(SparseMoeBlock|MoeBlock|MoEBlock|MoE|Moe)$', name) and 'Layer' not in name:
        # Arch-specific MoE block first
        for arch_moe in (f'{folder}_moe.py',):
            f = find_kb_file('L2', arch_moe)
            if f: out.append(kb_path('L2', f)); return out
        # Inherited from MoE class? E.g. GptOssMLP isn't *MoE-suffixed
        # MXFP4
        if folder == 'gpt_oss':
            out.append(kb_path('L2', 'gpt_oss_moe.py')); return out
        # Shared+routed pattern
        if has_kb('L2', 'shared_expert_moe.py'):
            out.append(kb_path('L2', 'shared_expert_moe.py')); return out
        # Fallback compose
        out.append(kb_path('L1', 'moe_grouped_gemm.py'))
        if has_kb('L1', 'sigmoid_topk.py'):
            out.append(kb_path('L1', 'sigmoid_topk.py'))
        return out

    # T5-family LayerSelfAttention / LayerCrossAttention — WIRING classes
    # (norm + T5Attention + residual). NOT direct attention; mark composes.
    if re.search(r'(LayerSelfAttention|LayerCrossAttention)$', name):
        return out

    # DAB-DETR / similar: DecoderLayer{Self,Cross}Attention — also wiring
    if re.search(r'DecoderLayer(Self|Cross)Attention$', name):
        return out

    # MultiHeadAttention (flaubert, ctrl, xlm, qformer variants) — generic MHA
    if re.search(r'MultiHeadAttention$', name):
        if folder in ('flaubert', 'ctrl', 'xlm'):
            f = find_kb_file('L2', 'encoder_attention.py')
            if f: out.append(kb_path('L2', f)); return out
        # QFormer / etc. — also encoder-style
        f = find_kb_file('L2', 'attention.py')
        if f: out.append(kb_path('L2', f)); return out
        return out

    # RG-LRU (recurrent_gemma)
    if re.search(r'(Rglru|RGLRU)$', name):
        f = find_kb_file('L1', 'rg_lru.py')
        if f: out.append(kb_path('L1', f)); return out
        return out

    # Attention variants
    if re.search(r'(?<!Pool)Attention$', name):
        # MLA / DeepSeek
        if 'MLA' in name or any('Deepseek' in b or 'DeepSeek' in b for b in bases) or folder.startswith('deepseek'):
            f = find_kb_file('L2', 'deepseek_mla_attention.py')
            if f: out.append(kb_path('L2', f)); return out
        # Encoder attention (BERT-family)
        if folder in ('bert', 'roberta', 'electra', 'distilbert', 'albert', 'big_bird',
                      'bigbird_pegasus', 'data2vec', 'mobilebert', 'mpnet', 'xlnet',
                      'xlm_roberta', 'flaubert', 'camembert', 'ernie', 'ernie4_5',
                      'bert_generation', 'roformer', 'rembert'):
            f = find_kb_file('L2', 'encoder_attention.py')
            if f: out.append(kb_path('L2', f)); return out
        # Whisper-family attention (3 variants)
        if folder == 'whisper' or any('Whisper' in b for b in bases):
            f = find_kb_file('L2', 'whisper_attention.py')
            if f: out.append(kb_path('L2', f)); return out
        # Arch-specific attention
        f = find_kb_file('L2', f'{folder}_attention.py')
        if f: out.append(kb_path('L2', f)); return out
        # CLIP-pattern (text/vision encoder, manual SDPA via BMM)
        if folder.startswith('clip') or 'CLIP' in str(bases):
            f = find_kb_file('L2', 'clip_attention.py')
            if f: out.append(kb_path('L2', f)); return out
        # SigLIP pattern
        if folder.startswith('siglip') or any('Siglip' in b for b in bases):
            f = find_kb_file('L2', 'siglip_attention.py')
            if f: out.append(kb_path('L2', f)); return out
        # Llama-family default (decoder attention with paged KV)
        f = find_kb_file('L2', 'attention.py')
        if f: out.append(kb_path('L2', f)); return out
        # Last resort: decompose
        for op_file in ('linear.py', 'dense_attention.py', 'store_kvcache.py', 'rotary_emb.py'):
            if has_kb('L1', op_file):
                out.append(kb_path('L1', op_file))
        return out

    # T5 dense FFN variants
    if name in ('T5DenseActDense', 'T5DenseGatedActDense'):
        f = find_kb_file('L2', 't5_dense.py')
        if f:
            out.append(kb_path('L2', f)); return out

    # BERT-family Intermediate/Output (the two-layer MLP split into 2 classes)
    if re.search(r'Intermediate$', name) and folder in (
        'bert', 'roberta', 'electra', 'distilbert', 'albert', 'mobilebert',
        'mpnet', 'xlnet', 'xlm_roberta', 'flaubert', 'camembert', 'ernie',
        'ernie4_5', 'bert_generation', 'roformer', 'rembert', 'big_bird',
        'bigbird_pegasus'):
        f = find_kb_file('L2', 'encoder_mlp.py')
        if f:
            out.append(kb_path('L2', f)); return out

    # MLP variants (SwiGLU vs two-layer) — case-insensitive Mlp/MLP
    if re.search(r'(MLP|Mlp|FeedForward|FFN)$', name):
        # Arch-specific MLP file?
        f = find_kb_file('L2', f'{folder}_mlp.py')
        if f: out.append(kb_path('L2', f)); return out
        # Inherited from a known MLP class?
        for parent in bases:
            ps = to_snake(parent)
            f = find_kb_file('L2', f'{ps}.py')
            if f and ('mlp' in f or 'dense' in f):
                out.append(kb_path('L2', f)); return out
        # Encoder MLP (BERT-family with EncoderIntermediate/Output)
        if folder in ('bert', 'roberta', 'electra', 'distilbert', 'albert',
                      'mobilebert', 'mpnet', 'xlnet', 'xlm_roberta', 'flaubert',
                      'camembert', 'ernie', 'ernie4_5', 'bert_generation',
                      'roformer', 'rembert'):
            f = find_kb_file('L2', 'encoder_mlp.py')
            if f: out.append(kb_path('L2', f)); return out
        # Whisper MLP
        if folder == 'whisper' or any('Whisper' in b for b in bases):
            f = find_kb_file('L2', 'whisper_mlp.py')
            if f: out.append(kb_path('L2', f)); return out
        # CLIP MLP
        if folder.startswith('clip') or any('CLIP' in b or 'Clip' in b for b in bases):
            f = find_kb_file('L2', 'clip_mlp.py')
            if f: out.append(kb_path('L2', f)); return out
        # SigLIP MLP
        if folder.startswith('siglip') or any('Siglip' in b for b in bases):
            f = find_kb_file('L2', 'siglip_mlp.py')
            if f: out.append(kb_path('L2', f)); return out
        # T5 dense
        if folder == 't5' or any('T5' in b for b in bases):
            f = find_kb_file('L2', 't5_dense.py')
            if f: out.append(kb_path('L2', f)); return out
        # Default: SwiGLU (Llama-pattern)
        f = find_kb_file('L2', 'llama_mlp.py')
        if f: out.append(kb_path('L2', f)); return out
        # Compose
        out.append(kb_path('L1', 'linear.py'))
        if has_kb('L1', 'silu.py'):
            out.append(kb_path('L1', 'silu.py'))
        return out

    # Embeddings (multi-component composite)
    if re.search(r'Embeddings$', name):
        # Encoder-style (token + pos + token_type)
        if folder in ('bert', 'roberta', 'electra', 'distilbert', 'albert',
                      'mobilebert', 'xlnet', 'xlm_roberta'):
            f = find_kb_file('L2', 'encoder_embeddings.py')
            if f: out.append(kb_path('L2', f)); return out
        # Vision embeddings
        if 'Vision' in name and has_kb('L1', 'conv2d.py'):
            out.append(kb_path('L1', 'conv2d.py'))
            out.append(kb_path('L1', 'embedding.py'))
            return out
        # Default
        out.append(kb_path('L1', 'embedding.py'))
        if has_kb('L1', 'layer_norm.py'):
            out.append(kb_path('L1', 'layer_norm.py'))
        return out

    # Pooling head
    if re.search(r'Pool(ing)?(Head|Latent)?$', name):
        if 'Attention' in name:
            f = find_kb_file('L2', 'attention_pool.py')
            if f: out.append(kb_path('L2', f)); return out
        out.append(kb_path('L1', 'linear.py'))
        return out

    # Mixer (Mamba family: state-space scan)
    if re.search(r'Mixer$', name):
        f = find_kb_file('L2', f'{folder}_mixer.py')
        if f: out.append(kb_path('L2', f)); return out
        # mamba_mixer used by jamba and others
        f = find_kb_file('L2', 'mamba_mixer.py')
        if f: out.append(kb_path('L2', f)); return out
        return out

    # Patch embedding / merger
    if re.search(r'PatchEmbed$', name):
        f = find_kb_file('L2', f'{folder}_patch_embed.py') or find_kb_file('L2', 'vision_patch_embed.py')
        if f: out.append(kb_path('L2', f)); return out
        out.append(kb_path('L1', 'conv2d.py'))
        return out

    if re.search(r'PatchMerg(er|ing)$', name):
        f = find_kb_file('L2', f'{folder}_patch_merger.py') or find_kb_file('L2', 'vision_patch_merger.py')
        if f: out.append(kb_path('L2', f)); return out
        out.append(kb_path('L1', 'linear.py'))
        return out

    # =========================================================
    # L3 LAYERS (DecoderLayer / EncoderLayer / Block) — wiring class
    # =========================================================
    # ORDER MATTERS: prefer the most specific name match first.
    if re.search(r'DecoderLayer$', name):
        for cand in (f'{folder}_decoder_layer.py', f'{folder}_decoder.py'):
            f = find_kb_file('L3', cand)
            if f: out.append(kb_path('L3', f)); return out
        return out  # composes (wiring)

    if re.search(r'EncoderLayer$', name):
        for cand in (f'{folder}_encoder_layer.py', f'{folder}_encoder.py'):
            f = find_kb_file('L3', cand)
            if f: out.append(kb_path('L3', f)); return out
        return out

    if re.search(r'Block$', name) and 'LayerNorm' not in name:
        for cand in (f'{folder}_block.py', f'{folder}_decoder.py',
                     f'{folder}_layer.py'):
            f = find_kb_file('L3', cand)
            if f: out.append(kb_path('L3', f)); return out
        return out

    if re.search(r'Layer$', name) and 'LayerNorm' not in name:
        # Generic *Layer (not DecoderLayer/EncoderLayer caught above)
        for cand in (f'{folder}_layer.py', f'{folder}_block.py'):
            f = find_kb_file('L3', cand)
            if f: out.append(kb_path('L3', f)); return out
        return out

    # Encoder / Decoder (multi-layer wrapper)
    if re.search(r'(Encoder|Decoder)$', name) and 'PreTrained' not in name:
        for cand in (f'{folder}_{to_snake(suffix)}.py',):
            f = find_kb_file('L3', cand)
            if f: out.append(kb_path('L3', f)); return out
        return out

    # =========================================================
    # L4 PIPELINES (Model / ForXxx)
    # =========================================================
    if re.search(r'Model$|For[A-Z]\w+$', name) and not re.search(r'PreTrainedModel$', name):
        # Direct L4 file?
        f = find_kb_file('L4', f'{folder}.py')
        if f: out.append(kb_path('L4', f)); return out
        # Vision/Text submodel of a multimodal arch?
        if 'TextModel' in name and has_kb('L4', f'{folder}_text_model.py'):
            out.append(kb_path('L4', f'{folder}_text_model.py')); return out
        if 'VisionModel' in name and has_kb('L4', f'{folder}_vision_model.py'):
            out.append(kb_path('L4', f'{folder}_vision_model.py')); return out
        return out  # composes (wiring) — empty

    # Unknown class — leave empty
    return out


def parse_classes(p: Path):
    try:
        tree = ast.parse(p.read_text())
    except SyntaxError:
        return [], {}
    classes = []
    line_nums = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.name.startswith('_'):
                continue
            bases = []
            for b in node.bases:
                try:
                    bases.append(ast.unparse(b).split('.')[-1])
                except Exception:
                    pass
            classes.append((node.name, bases))
            line_nums[node.name] = node.lineno
    return classes, line_nums


def should_skip(class_name, base_classes):
    """Skip if name matches boilerplate pattern OR all bases are boilerplate."""
    if SKIP_NAME_RE.match(class_name):
        return True
    if base_classes and all(b in SKIP_BASES_ONLY for b in base_classes):
        return True
    return False


def extract_inheritance(folder, classes):
    fc = folder_camel(folder)
    chain = []
    for cname, bases in classes:
        for b in bases:
            if b in NEUTRAL_BASES:
                continue
            if b.startswith(fc):
                continue
            chain.append(f'{cname}({b})')
            break
    return chain


def reaudit_one(folder, modeling_file_rel):
    folder_path = HF_PINNED / folder
    short = modeling_file_rel.split('/')[-1].replace('modeling_', '').replace('.py', '')
    modular = folder_path / f'modular_{short}.py'
    modeling = folder_path / f'modeling_{short}.py'
    src = modular if modular.exists() else modeling
    if not src.exists():
        return None

    classes, line_nums = parse_classes(src)
    operators = []
    per_class = []
    evidence = []
    for cname, bases in classes:
        if should_skip(cname, bases):
            continue
        kernels = map_class(folder, cname, bases, hf_src_path=src)
        operators.append(cname)
        per_class.append((cname, kernels))
        if cname in line_nums:
            evidence.append(f'{folder}/{src.name}:{line_nums[cname]}')

    inheritance = extract_inheritance(folder, classes)
    has_l4 = (REPO / 'tasks/baseline/L4' / f'{folder}.py').exists()

    return {
        'operators': operators,
        'per_class': per_class,
        'evidence': evidence,
        'inheritance': inheritance,
        'has_l4': has_l4,
        'src_name': src.name,
    }


def main():
    rows = list(csv.DictReader(open(COVERAGE_CSV)))
    fieldnames = list(rows[0].keys())
    n_rewritten = 0
    n_skipped = 0

    kb_referenced = {layer: set() for layer in ('L1', 'L2', 'L3', 'L4')}

    for r in rows:
        if r['support_status'] == 'not_inference_required':
            n_skipped += 1; continue
        if not r['modeling_file']:
            n_skipped += 1; continue
        result = reaudit_one(r['hf_folder'], r['modeling_file'])
        if not result:
            n_skipped += 1; continue

        # For partial/unsupported rows, preserve the original mapping (the
        # gap is real — claiming a kernel that exists but doesn't implement
        # the missing op is dishonest). Still update architecture_classes
        # + evidence + notes so the row has the full HF class list.
        is_unsupported = r['support_status'] in ('partial', 'unsupported')

        # Build per-class mapping
        mapping_entries = []
        for cname, kernels in result['per_class']:
            if kernels:
                paths = '+'.join(kernels)
                mapping_entries.append(f'{cname}->{paths}')
                for p in kernels:
                    m = re.match(r'tasks/baseline/(L[1-4])/(.+\.py)', p)
                    if m:
                        kb_referenced[m.group(1)].add(m.group(2))
            else:
                mapping_entries.append(f'{cname}->composes')

        r['architecture_classes'] = ';'.join(result['operators'])
        if not is_unsupported:
            r['mapped_kb_nano'] = ';'.join(mapping_entries)
        if result['evidence']:
            r['evidence_hf'] = ';'.join(result['evidence'])

        # Notes: inheritance chain + L4 marker
        notes_parts = []
        if result['inheritance']:
            notes_parts.append(f'[v5 inheritance] ' + ', '.join(result['inheritance'][:5]))
        if result['has_l4']:
            notes_parts.append(f'[v5] L4/{r["hf_folder"]}.py exists')
        if notes_parts:
            old_notes = (r.get('notes') or '').strip()
            new_note = ' '.join(notes_parts)
            if new_note not in old_notes:
                r['notes'] = (old_notes + ' ' + new_note).strip()

        n_rewritten += 1

    with open(COVERAGE_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f'\nRewrote {n_rewritten} rows; skipped {n_skipped}.')
    print()
    print('=== kb-nano reference coverage (after v5 reaudit) ===')
    for layer in ('L1', 'L2', 'L3', 'L4'):
        total = len(KB_FILES[layer])
        ref = len(kb_referenced[layer])
        pct = 100 * ref / total if total else 0
        print(f'  {layer}: {ref}/{total} ({pct:.1f}%)')


if __name__ == '__main__':
    main()
