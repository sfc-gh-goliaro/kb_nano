"""AST extractor: pulls compute operators out of an HF modeling_*.py file.

Output (JSON): per modeling file, the list of (canonical_op, count, file:line refs).

This is intentionally conservative: it canonicalizes only well-known compute
primitives. Anything it cannot resolve is emitted as an `unresolved` entry so
the auditor can extend the lookup table during pilot.

Run as a CLI:
    python ast_extract.py <path_to_modeling_file.py>
    python ast_extract.py --dir /tmp/hf_transformers_pinned/src/transformers/models  -> JSONL on stdout
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from collections import defaultdict


# Canonical-op lookup: tuple of attribute-access strings (joined dotted name)
# is mapped to a canonical op name. We match suffixes — i.e. if a call's
# dotted path ENDS WITH any of these, it counts.
_NN_LOOKUPS = {
    # linear / matmul
    "Linear": "linear",
    "Bilinear": "bilinear",
    # norms
    "LayerNorm": "layer_norm",
    "GroupNorm": "group_norm",
    "BatchNorm1d": "batch_norm_1d",
    "BatchNorm2d": "batch_norm_2d",
    "BatchNorm3d": "batch_norm_3d",
    "InstanceNorm1d": "instance_norm",
    "InstanceNorm2d": "instance_norm",
    "InstanceNorm3d": "instance_norm",
    # embeddings
    "Embedding": "embedding",
    # conv
    "Conv1d": "conv1d",
    "Conv2d": "conv2d",
    "Conv3d": "conv3d",
    "ConvTranspose1d": "conv_transpose1d",
    "ConvTranspose2d": "conv_transpose2d",
    "ConvTranspose3d": "conv_transpose3d",
    # pooling
    "MaxPool1d": "max_pool_1d",
    "MaxPool2d": "max_pool_2d",
    "MaxPool3d": "max_pool_3d",
    "AvgPool1d": "avg_pool_1d",
    "AvgPool2d": "avg_pool_2d",
    "AvgPool3d": "avg_pool_3d",
    "AdaptiveAvgPool1d": "adaptive_avg_pool_1d",
    "AdaptiveAvgPool2d": "adaptive_avg_pool_2d",
    "AdaptiveAvgPool3d": "adaptive_avg_pool_3d",
    # activations
    "GELU": "gelu",
    "ReLU": "relu",
    "SiLU": "silu",
    "Sigmoid": "sigmoid",
    "Tanh": "tanh",
    "Softmax": "softmax",
    "LeakyReLU": "leaky_relu",
    "ELU": "elu",
    "Hardsigmoid": "hardsigmoid",
    "Hardswish": "hardswish",
    "Mish": "mish",
    # dropout
    "Dropout": "dropout",
    "Dropout1d": "dropout",
    "Dropout2d": "dropout",
    "Dropout3d": "dropout",
    # attention
    "MultiheadAttention": "multihead_attention",
    # recurrent
    "LSTM": "lstm",
    "GRU": "gru",
    "RNN": "rnn",
    # other
    "Identity": "identity",
    "Flatten": "flatten",
    "Unfold": "unfold",
    "Fold": "fold",
    "PixelShuffle": "pixel_shuffle",
    "Upsample": "upsample",
    "ZeroPad2d": "zero_pad_2d",
}

_F_LOOKUPS = {
    "linear": "linear",
    "embedding": "embedding",
    "layer_norm": "layer_norm",
    "group_norm": "group_norm",
    "batch_norm": "batch_norm",
    "rms_norm": "rms_norm",
    "softmax": "softmax",
    "log_softmax": "log_softmax",
    "gelu": "gelu",
    "relu": "relu",
    "silu": "silu",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "leaky_relu": "leaky_relu",
    "elu": "elu",
    "mish": "mish",
    "hardsigmoid": "hardsigmoid",
    "hardswish": "hardswish",
    "quick_gelu": "quick_gelu",
    "scaled_dot_product_attention": "sdpa",
    "conv1d": "conv1d",
    "conv2d": "conv2d",
    "conv3d": "conv3d",
    "conv_transpose1d": "conv_transpose1d",
    "conv_transpose2d": "conv_transpose2d",
    "conv_transpose3d": "conv_transpose3d",
    "max_pool1d": "max_pool_1d",
    "max_pool2d": "max_pool_2d",
    "max_pool3d": "max_pool_3d",
    "avg_pool1d": "avg_pool_1d",
    "avg_pool2d": "avg_pool_2d",
    "avg_pool3d": "avg_pool_3d",
    "adaptive_avg_pool1d": "adaptive_avg_pool_1d",
    "adaptive_avg_pool2d": "adaptive_avg_pool_2d",
    "adaptive_avg_pool3d": "adaptive_avg_pool_3d",
    "interpolate": "interpolate",
    "grid_sample": "grid_sample",
    "pad": "pad",
    "dropout": "dropout",
    "pixel_shuffle": "pixel_shuffle",
    "pixel_unshuffle": "pixel_unshuffle",
    "one_hot": "one_hot",
    "cross_entropy": "cross_entropy",
    "binary_cross_entropy": "binary_cross_entropy",
    "binary_cross_entropy_with_logits": "binary_cross_entropy",
    "mse_loss": "mse_loss",
    "ctc_loss": "ctc_loss",
    "fold": "fold",
    "unfold": "unfold",
    "logsigmoid": "log_sigmoid",
}

_TORCH_LOOKUPS = {
    "matmul": "matmul",
    "mm": "matmul",
    "bmm": "batch_matmul",
    "einsum": "einsum",
    "softmax": "softmax",
    "log_softmax": "log_softmax",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "relu": "relu",
    "topk": "topk",
    "argmax": "argmax",
    "argmin": "argmin",
    "where": "where",
    "gather": "gather",
    "scatter": "scatter",
    "scatter_add": "scatter",
    "index_select": "index_select",
    "masked_fill": "masked_fill",
    "masked_select": "masked_select",
    "cat": "cat",
    "stack": "stack",
    "split": "split",
    "chunk": "chunk",
    "repeat_interleave": "repeat_interleave",
    "roll": "roll",
    "flip": "flip",
    "tril": "tril",
    "triu": "triu",
    "cumsum": "cumsum",
    "cumprod": "cumprod",
    "exp": "exp",
    "log": "log",
    "rsqrt": "rsqrt",
    "sqrt": "sqrt",
    "pow": "pow",
    "clamp": "clamp",
    "abs": "abs",
    "sign": "sign",
    "sin": "sin",
    "cos": "cos",
    "outer": "outer",
    "logsumexp": "logsumexp",
    "norm": "vector_norm",
    "ones_like": "ones_like",
    "zeros_like": "zeros_like",
    "full_like": "full_like",
    "eye": "eye",
    "arange": "arange",
}

# ACT2FN string keys map directly to activation canonical names
_ACT2FN_MAP = {
    "gelu": "gelu",
    "gelu_new": "gelu",
    "gelu_python": "gelu",
    "gelu_pytorch_tanh": "gelu",
    "gelu_fast": "gelu",
    "gelu_accurate": "gelu",
    "relu": "relu",
    "relu2": "relu",
    "silu": "silu",
    "swish": "silu",
    "tanh": "tanh",
    "sigmoid": "sigmoid",
    "linear": "identity",
    "mish": "mish",
    "quick_gelu": "quick_gelu",
    "laplace": "laplace_activation",
    "elu": "elu",
}

# Special HF helpers we want to flag as significant compute primitives
_HF_HELPERS = {
    "apply_rotary_pos_emb": "rotary_pos_emb",
    "apply_rotary_emb": "rotary_pos_emb",
    "rotate_half": "rotary_pos_emb",
    "_compute_default_rope_parameters": "rope_param",
    "ALL_ATTENTION_FUNCTIONS": "attention_dispatcher",
    "eager_attention_forward": "eager_attention",
    "sdpa_attention_forward": "sdpa",
    "flash_attention_forward": "flash_attention",
    "flex_attention_forward": "flex_attention",
    "DynamicCache": "kv_cache",
    "StaticCache": "kv_cache",
    "Cache": "kv_cache",
    "EncoderDecoderCache": "encoder_decoder_cache",
    "HybridCache": "kv_cache",
    "HybridChunkedCache": "kv_cache",
    "MambaCache": "ssm_cache",
    "selective_scan_fn": "selective_scan",
    "mamba_chunk_scan_combined": "mamba_scan",
    "causal_conv1d_fn": "causal_conv1d",
    "causal_conv1d_update": "causal_conv1d",
    "chunk_gated_delta_rule": "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule": "fused_recurrent_gated_delta_rule",
    "multi_scale_deformable_attention": "deformable_attention",
    "MultiScaleDeformableAttnFunction": "deformable_attention",
    "ms_deform_attn_core_pytorch": "deformable_attention",
    "_attn_implementation_autoset": "attention_dispatcher",
}


def dotted(node: ast.AST) -> str | None:
    """Return dotted-path string for an Attribute/Name chain, or None."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    elif isinstance(cur, ast.Call):
        # Like: ACT2FN[hidden_act](x); won't have dotted path
        return None
    else:
        return None
    return ".".join(reversed(parts))


def canonicalize(call_path: str, callee_node: ast.AST) -> str | None:
    """Map a callable's dotted path to a canonical op name, or None if not a known compute primitive."""
    if not call_path:
        return None
    parts = call_path.split(".")
    last = parts[-1]
    # F.X — if the immediate parent reads "F" and last is in F_LOOKUPS
    if len(parts) >= 2 and parts[-2] in ("F", "functional", "torch.nn.functional"):
        return _F_LOOKUPS.get(last)
    # nn.X
    if len(parts) >= 2 and parts[-2] == "nn":
        return _NN_LOOKUPS.get(last)
    # torch.X (top-level torch ops)
    if len(parts) >= 2 and parts[-2] == "torch":
        return _TORCH_LOOKUPS.get(last)
    # tensor.matmul / tensor.softmax / tensor.bmm — method calls on tensors
    if len(parts) >= 1:
        if last in _TORCH_LOOKUPS and last in {"matmul", "bmm", "softmax", "log_softmax",
                                                "sigmoid", "tanh", "topk", "argmax", "argmin",
                                                "gather", "scatter", "scatter_add", "index_select",
                                                "masked_fill", "masked_select", "cumsum",
                                                "exp", "log", "rsqrt", "sqrt", "pow", "abs",
                                                "tril", "triu", "outer", "logsumexp"}:
            # Likely tensor method call — treat as the same op
            return _TORCH_LOOKUPS[last]
    # HF helper functions
    if last in _HF_HELPERS:
        return _HF_HELPERS[last]
    # ACT2FN[...] — handled separately at call site
    return None


def extract_act2fn_keys(tree: ast.AST) -> set[str]:
    """Find ACT2FN[<key>] usages where key is a literal string.

    Many HF files use ACT2FN[config.hidden_act] which is dynamic — those we
    flag as `act2fn_dynamic` (the config value resolves at runtime).
    """
    keys: set[str] = set()
    dynamic = False
    class V(ast.NodeVisitor):
        def visit_Subscript(self, n: ast.Subscript):  # type: ignore[override]
            nonlocal dynamic
            target = dotted(n.value)
            if target and target.endswith("ACT2FN"):
                if isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
                    keys.add(n.slice.value)
                else:
                    dynamic = True
            self.generic_visit(n)
    V().visit(tree)
    if dynamic:
        keys.add("__dynamic__")
    return keys


def extract_attention_keys(tree: ast.AST) -> set[str]:
    """Find ALL_ATTENTION_FUNCTIONS[<key>] usages."""
    keys: set[str] = set()
    dynamic = False
    class V(ast.NodeVisitor):
        def visit_Subscript(self, n: ast.Subscript):  # type: ignore[override]
            nonlocal dynamic
            target = dotted(n.value)
            if target and target.endswith("ALL_ATTENTION_FUNCTIONS"):
                if isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
                    keys.add(n.slice.value)
                else:
                    dynamic = True
            self.generic_visit(n)
    V().visit(tree)
    if dynamic:
        keys.add("__dynamic__")
    return keys


def get_class_bases(node: ast.ClassDef) -> list[str]:
    return [ast.unparse(b) for b in node.bases]


def is_pretrained_model_class(cls: ast.ClassDef) -> bool:
    bases = " ".join(get_class_bases(cls))
    return any(k in bases for k in ("PreTrainedModel", "GenerationMixin"))


def find_classes(tree: ast.Module) -> list[ast.ClassDef]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]


def list_top_classes_for_arch(tree: ast.Module) -> list[str]:
    """Public architecture classes (typically *Model, *ForXxx, *PreTrainedModel)."""
    out = []
    for cls in find_classes(tree):
        bases = get_class_bases(cls)
        bjoin = " ".join(bases)
        if any(k in bjoin for k in ("PreTrainedModel", "GenerationMixin")):
            out.append(cls.name)
        elif cls.name.endswith("Model") or cls.name.startswith("For") or "For" in cls.name and cls.name.split("For", 1)[-1] != "":
            # Heuristic — capture *ForXxx classes too. We leave the actual filtering to coordinator.
            pass
    return sorted(set(out))


def extract_ops_for_file(path: Path) -> dict:
    src = path.read_text()
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:
        return {"file": str(path), "error": f"SyntaxError: {e}"}
    ops: dict[str, list[str]] = defaultdict(list)
    unresolved: dict[str, int] = defaultdict(int)
    # Walk all calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            path_str = dotted(node.func)
            canon = canonicalize(path_str, node.func) if path_str else None
            if canon:
                ops[canon].append(f"{path.name}:{node.lineno}")
            elif path_str:
                # Capture only short, well-formed dotted paths to keep noise down.
                last = path_str.split(".")[-1]
                if last and last[0].islower() and len(last) > 2:
                    unresolved[path_str] += 1
    # ACT2FN keys
    a2f = extract_act2fn_keys(tree)
    if a2f:
        for k in a2f:
            canon = _ACT2FN_MAP.get(k, "act2fn_other") if k != "__dynamic__" else "act2fn_dynamic"
            ops[canon].append(f"{path.name}:ACT2FN[{k!r}]")
    # ALL_ATTENTION_FUNCTIONS keys
    aaf = extract_attention_keys(tree)
    if aaf:
        for k in aaf:
            canon = "attention_dispatcher" if k == "__dynamic__" else f"attention_impl:{k}"
            ops[canon].append(f"{path.name}:ALL_ATTENTION_FUNCTIONS[{k!r}]")
    # Architecture classes
    arch_classes = list_top_classes_for_arch(tree)
    # Class definitions for cross-ref
    classes = []
    for cls in find_classes(tree):
        classes.append({
            "name": cls.name,
            "bases": get_class_bases(cls),
            "lineno": cls.lineno,
        })
    # Imports
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for n in node.names:
                imports.append(f"{mod}.{n.name}" if mod else n.name)
        elif isinstance(node, ast.Import):
            for n in node.names:
                imports.append(n.name)
    return {
        "file": str(path),
        "lines": len(src.splitlines()),
        "n_classes": len(classes),
        "architecture_classes": arch_classes,
        "classes": classes,
        "ops": {k: {"count": len(v), "refs": v[:5]} for k, v in sorted(ops.items())},
        "unresolved_top": dict(sorted(unresolved.items(), key=lambda x: -x[1])[:30]),
        "imports": sorted(set(imports)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--dir", help="Scan all modeling_*.py under this dir tree")
    ap.add_argument("--out", help="Write JSONL to this path (only with --dir)")
    args = ap.parse_args()
    if args.path:
        result = extract_ops_for_file(Path(args.path))
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    if args.dir:
        d = Path(args.dir)
        out_f = open(args.out, "w") if args.out else sys.stdout
        for modeling in sorted(d.glob("*/modeling_*.py")):
            # skip TF / Flax
            n = modeling.name
            if n.startswith("modeling_tf_") or n.startswith("modeling_flax_"):
                continue
            try:
                result = extract_ops_for_file(modeling)
            except Exception as e:
                result = {"file": str(modeling), "error": str(e)}
            out_f.write(json.dumps(result) + "\n")
        if args.out:
            out_f.close()
        return
    ap.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
