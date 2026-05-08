"""Merge all 16 audit batches into final paper-grade appendix tex (Option B+C).

B: For wiring classes (kb_nano_files=[]), parse HF source __init__ to extract
   the sub-modules they instantiate, render as "composes: ModuleA + ModuleB + ...".

C: Collapse pure task-head classes (*ForXxx where xxx ∈ {SequenceClassification,
   TokenClassification, QuestionAnswering, MultipleChoice, etc.}) into ONE row
   per folder: "*For{...} (n heads): composes <BaseModel> + Linear (per-task)".

Format: 4-column longtable with seqsplit-wrapped long names; folder + status
multirow over per-class rows; \\textit{composes: ...} for wiring rows.

Output: /home/olu/kb_nano/audits/hf_transformers_coverage/MENTOR_REVIEW_full_audit.tex
"""
from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from collections import defaultdict

REPO = Path('/home/olu/kb_nano')
HF = Path('/tmp/hf_transformers_pinned/src/transformers/models')
BATCH_DIR = REPO / 'audits/hf_transformers_coverage/tools'
OUT_TEX = REPO / 'audits/hf_transformers_coverage/MENTOR_REVIEW_full_audit.tex'
COVERAGE_CSV = REPO / 'audits/hf_transformers_coverage/hf_architecture_operator_coverage.csv'

STATUS_MARKER = {
    'kb_nano_l4': r'$\bullet$',
    'composable': r'\cmark',
    'partial': r'\textbf{P}',
    'unsupported': r'\xmark',
    'not_inference_required': '--',
}

# Task-head suffixes that compose only "BaseModel + Linear" (collapse-able)
COLLAPSE_HEADS = {
    'ForSequenceClassification', 'ForTokenClassification',
    'ForQuestionAnswering', 'ForMultipleChoice',
    'ForNextSentencePrediction', 'ForPreTraining',
    'ForImageClassification',
    'ForSemanticSegmentation', 'ForAudioClassification',
    'ForVideoClassification', 'ForAudioFrameClassification',
    'ForUniversalSegmentation', 'ForInstanceSegmentation',
    'ForMaskGeneration', 'ForKeypointDetection',
    'ForXVector', 'ForCTC', 'ForDocumentQuestionAnswering',
    'ForTableQuestionAnswering', 'ForVisualQuestionAnswering',
    'ForCausalImageModeling', 'ForMaskedImageModeling',
    'LMHeadModel',
}
# Heads that should NOT be collapsed (primary forward path)
KEEP_HEADS = {
    'ForCausalLM', 'ForConditionalGeneration', 'ForMaskedLM',
}


def latex_escape(s: str) -> str:
    return s.replace('_', r'\_').replace('&', r'\&')


def le(s: str) -> str:
    return latex_escape(s)


def texttt(s: str) -> str:
    """\\texttt{} with \\seqsplit{} for break-anywhere on long names."""
    esc = le(s)
    if len(s) > 14:
        return r'\texttt{\seqsplit{' + esc + '}}'
    return r'\texttt{' + esc + '}'


def fmt_kb_path(path: str) -> str:
    """tasks/baseline/L2/foo.py -> \\texttt{L2/}\\allowbreak\\texttt{foo.py}"""
    if not path.startswith('tasks/baseline/'):
        return r'\texttt{' + le(path) + '}'
    short = path[len('tasks/baseline/'):]
    layer, fname = short.split('/', 1)
    return r'\texttt{' + le(layer) + r'/}\allowbreak\texttt{' + le(fname) + '}'


def short_modeling_label(folder: str, src_file: str) -> str:
    """For multi-modeling-file rows like blip/blip_text, render as folder/short."""
    if not src_file:
        return texttt(folder)
    base = src_file.replace('modeling_', '').replace('modular_', '').replace('.py', '')
    if base == folder:
        return texttt(folder)
    return r'\texttt{\seqsplit{' + le(folder) + '/' + le(base) + '}}'


# --- Wiring expansion: parse HF source __init__ to find sub-modules ----

def get_init_submodules(hf_src_path: Path, class_name: str) -> list[tuple[str, str]]:
    """For class `class_name` in `hf_src_path`, parse __init__ to find
    every `self.<attr> = <ClassName>(...)` assignment.
    Returns [(attr, ClassName), ...] in source order.
    """
    if not hf_src_path.exists():
        return []
    try:
        tree = ast.parse(hf_src_path.read_text())
    except SyntaxError:
        return []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        # Find __init__
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                return _walk_init(item)
    return []


_NON_CLASS_CALLS = {
    # builtins
    'list', 'dict', 'tuple', 'set', 'frozenset', 'getattr', 'setattr',
    'super', 'len', 'int', 'float', 'str', 'bool', 'range', 'enumerate',
    'zip', 'iter', 'next', 'sorted', 'reversed', 'sum', 'min', 'max', 'any', 'all',
    'isinstance', 'issubclass', 'hasattr', 'callable', 'type', 'id',
    'print', 'repr', 'abs', 'round', 'divmod', 'pow', 'open', 'format',
    # math / torch / functional
    'sqrt', 'log', 'log2', 'log10', 'exp', 'sin', 'cos', 'tan',
    'floor', 'ceil', 'pi', 'inf',
    'tensor', 'zeros', 'ones', 'empty', 'arange', 'linspace', 'eye',
    'full', 'rand', 'randn', 'randint', 'cat', 'stack', 'as_tensor',
    'from_numpy', 'tensor_split',
    # torch param/buffer (these are not modules in our sense)
    'Parameter', 'Buffer', 'register_buffer', 'register_parameter',
    # init helpers
    'normal_', 'uniform_', 'zeros_', 'ones_', 'kaiming_normal_',
    # misc factory funcs
    'partial', 'deepcopy', 'copy', 'clone', 'detach', 'contiguous',
}


def _is_class_like(name: str) -> bool:
    """Heuristic: a class call has a name starting with an uppercase letter
    AND is not a known non-class builtin. Lowercase calls (sqrt, tensor, etc.)
    are treated as functions, not class instantiations."""
    if not name:
        return False
    if name in _NON_CLASS_CALLS:
        return False
    return name[0].isupper()


def _resolve_activation_default(folder_path: Path) -> str:
    """Find the default `hidden_act` value in the folder's configuration_*.py.
    Returns the resolved kernel filename (e.g. 'silu.py') or 'activation.py' if unknown."""
    if not folder_path.exists():
        return 'activation.py'
    cfg_files = list(folder_path.glob('configuration_*.py'))
    if not cfg_files:
        return 'activation.py'
    text = cfg_files[0].read_text()
    # Look for hidden_act = "..." or hidden_act: type = "..." (with type annotations)
    for m in re.finditer(r'hidden_act\s*(?::\s*\w+)?\s*=\s*["\'](\w+)["\']', text):
        act = m.group(1)
        return _ACT_TO_KERNEL.get(act, 'activation.py')
    return 'activation.py'


_ACT_TO_KERNEL = {
    'silu': 'silu.py',
    'swish': 'silu.py',
    'gelu': 'gelu.py',
    'gelu_new': 'gelu.py',
    'gelu_pytorch_tanh': 'gelu.py',
    'gelu_fast': 'gelu.py',
    'quick_gelu': 'quickgelu.py',
    'quickgelu': 'quickgelu.py',
    'relu': 'relu.py',
    'relu2': 'squared_relu.py',
    'relu_squared': 'squared_relu.py',
    'tanh': 'tanh.py',
    'sigmoid': 'sigmoid.py',
    'mish': 'activation.py',  # not in kb-nano
    'elu': 'activation.py',
}


def _walk_init(func: ast.FunctionDef) -> list[tuple[str, str]]:
    """Walk an __init__ function; return [(attr_name, instantiated_class), ...].
    Returns ALL instances (not deduped) so caller can compute counts.
    Handles:
      - self.X = ClassName(...)
      - self.X = nn.ModuleList([ChildClass(...) for ...])  -> extracts ChildClass
      - self.X = nn.ModuleList([ChildA(...), ChildB(...)])
      - self.X = nn.Sequential(ClassA(...), ClassB(...))   -> extracts ClassA, ClassB
      - self.X = nn.ModuleList(local_var) where local_var was built by .append(Cls(...))
    """
    # Track local-variable assignments and per-list appends so we can resolve
    # `self.X = nn.ModuleList(some_local_var)` patterns where some_local_var
    # was built up via `.append()` calls (possibly through intermediate locals).
    local_class: dict[str, str] = {}        # local_var -> ClassName
    local_appends: dict[str, list[str]] = {}  # list_var -> [ClassName, ...]
    local_assigns: dict[str, ast.AST] = {}    # local_var -> raw value AST
    for stmt in ast.walk(func):
        if isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name):
                    local_assigns[tgt.id] = stmt.value
                    # Direct: local = SomeClass(...)
                    if isinstance(stmt.value, ast.Call):
                        c = _call_class_name(stmt.value.func)
                        if c and _is_class_like(c):
                            local_class[tgt.id] = c
                    # Ternary: local = ClassA if cond else ClassB (class refs, not calls)
                    elif isinstance(stmt.value, ast.IfExp):
                        for branch in (stmt.value.body, stmt.value.orelse):
                            if isinstance(branch, ast.Name) and _is_class_like(branch.id):
                                local_class[tgt.id] = branch.id
                                break
                            if isinstance(branch, ast.Call):
                                c = _call_class_name(branch.func)
                                if c and _is_class_like(c):
                                    local_class[tgt.id] = c
                                    break
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            # Match: <list>.append(<arg>) or <list>.extend(<arg>)
            if (isinstance(call.func, ast.Attribute)
                    and call.func.attr in ('append', 'extend')
                    and isinstance(call.func.value, ast.Name)
                    and call.args):
                lname = call.func.value.id
                a = call.args[0]
                if isinstance(a, ast.Call):
                    c = _call_class_name(a.func)
                    if c:
                        local_appends.setdefault(lname, []).append(c)
                elif isinstance(a, ast.Name):
                    # blocks.append(block) where block = SomeClass(...)
                    c = local_class.get(a.id)
                    if c:
                        local_appends.setdefault(lname, []).append(c)

    def _resolve_local(name: str) -> list[str]:
        if name in local_appends:
            return local_appends[name]
        v = local_assigns.get(name)
        if v is not None:
            return _extract_inner_classes([v])
        return []

    # Collect Assign nodes that are NOT inside an `if` block. Conditional
    # sub-modules (e.g. `if config.add_cross_attention: self.crossattention =
    # ConvBertAttention(config)`) double-count the kernel — exclude them from
    # the unconditional list.
    def _collect_assigns(stmt_list, out_list):
        for stmt in stmt_list:
            if isinstance(stmt, ast.Assign):
                out_list.append(stmt)
            elif isinstance(stmt, (ast.For, ast.While)):
                _collect_assigns(stmt.body, out_list)
            elif isinstance(stmt, ast.Try):
                _collect_assigns(stmt.body, out_list)
            # Skip ast.If entirely (conditional sub-modules).

    init_assigns: list[ast.Assign] = []
    _collect_assigns(func.body, init_assigns)

    out = []
    for stmt in init_assigns:
        for target in stmt.targets:
            if not (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == 'self'):
                continue
            attr = target.attr
            value = stmt.value
            # Unwrap ternary: `self.x = Cls(...) if cond else None`
            if isinstance(value, ast.IfExp):
                if isinstance(value.body, ast.Call):
                    value = value.body
                elif isinstance(value.orelse, ast.Call):
                    value = value.orelse
                else:
                    continue
            # Capture activation lookups: `self.act = ACT2FN[config.hidden_act]` etc.
            if isinstance(value, ast.Subscript):
                if isinstance(value.value, ast.Name) and value.value.id in ('ACT2FN', 'ACT2CLS'):
                    out.append((attr, '__ACT2FN__'))
                continue
            if not isinstance(value, ast.Call):
                continue
            # Handle `Class._from_config(...)` / `Class.from_config(...)` / `Class.from_pretrained(...)`
            # — extract the receiver Class as the instantiated submodule.
            if isinstance(value.func, ast.Attribute):
                if value.func.attr in ('_from_config', 'from_config', 'from_pretrained'):
                    receiver = _call_class_name(value.func.value)
                    if receiver and _is_class_like(receiver):
                        out.append((attr, receiver))
                        continue
            top_cls = _call_class_name(value.func)
            if top_cls is None:
                continue
            # Special-case container constructors: dig into args to find the real class.
            if top_cls in ('ModuleList', 'Sequential', 'ParameterList'):
                inner_raw = _extract_inner_classes(value.args)
                # Also handle ModuleList(local_var)
                for arg in value.args:
                    if isinstance(arg, ast.Name):
                        inner_raw.extend(_resolve_local(arg.id))
                inner = [c for c in inner_raw if _is_class_like(c)]
                if inner:
                    for c in inner:
                        out.append((attr, c))
                else:
                    continue
            elif _is_class_like(top_cls):
                out.append((attr, top_cls))
    return out


def _extract_inner_classes(args: list[ast.AST]) -> list[str]:
    """For nn.ModuleList(...) / nn.Sequential(...), find class names in the args."""
    found = []
    for arg in args:
        if isinstance(arg, ast.ListComp):
            elt = arg.elt
            if isinstance(elt, ast.Call):
                c = _call_class_name(elt.func)
                if c:
                    found.append(c)
        elif isinstance(arg, ast.List):
            for elt in arg.elts:
                if isinstance(elt, ast.Call):
                    c = _call_class_name(elt.func)
                    if c:
                        found.append(c)
        elif isinstance(arg, ast.Call):
            # Sequential(ClassA(...), ClassB(...))
            c = _call_class_name(arg.func)
            if c:
                found.append(c)
        elif isinstance(arg, ast.GeneratorExp):
            elt = arg.elt
            if isinstance(elt, ast.Call):
                c = _call_class_name(elt.func)
                if c:
                    found.append(c)
    return found


def _call_class_name(node: ast.AST) -> str | None:
    """Get the class name for `func` in a Call node (handles nn.X, mod.X, X)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def get_modulelist_class(hf_src_path: Path, class_name: str, attr_name: str) -> str | None:
    """If `self.<attr_name> = nn.ModuleList([ChildClass(...) for ...])`, return ChildClass."""
    if not hf_src_path.exists():
        return None
    try:
        tree = ast.parse(hf_src_path.read_text())
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in ast.walk(node):
            if isinstance(item, ast.Assign):
                for tgt in item.targets:
                    if (isinstance(tgt, ast.Attribute)
                            and isinstance(tgt.value, ast.Name)
                            and tgt.value.id == 'self'
                            and tgt.attr == attr_name):
                        # value should be Call(nn.ModuleList, [list/listcomp of Call(Class, ...)])
                        v = item.value
                        if isinstance(v, ast.Call) and v.args:
                            arg = v.args[0]
                            if isinstance(arg, ast.ListComp):
                                elt = arg.elt
                                if isinstance(elt, ast.Call):
                                    return _call_class_name(elt.func)
                            elif isinstance(arg, ast.List) and arg.elts:
                                first = arg.elts[0]
                                if isinstance(first, ast.Call):
                                    return _call_class_name(first.func)
    return None


CLASS_NAME_DECOMPOSE_RULES = [
    # Pattern in name -> list of L1 paths to render as the decomposition
    # (used when subagent set kb_nano_files=[] for a compute class)
    # Order matters: longer/more specific patterns first.
    ('SelfAttention', ['L1/linear.py', 'L1/dense_attention.py', 'L1/store_kvcache.py']),
    ('CrossAttention', ['L1/linear.py', 'L1/dense_attention.py', 'L1/store_kvcache.py']),
    ('SdpaAttention', ['L1/linear.py', 'L1/dense_attention.py', 'L1/store_kvcache.py']),
    ('FlashAttention2', ['L1/linear.py', 'L1/flash_attn_varlen.py', 'L1/store_kvcache.py']),
    ('FlashAttention', ['L1/linear.py', 'L1/flash_attn_varlen.py', 'L1/store_kvcache.py']),
    ('Attention', ['L1/linear.py', 'L1/dense_attention.py', 'L1/store_kvcache.py']),
    ('SwiGLU', ['L1/linear.py', 'L1/silu_and_mul.py']),
    ('GeGLU', ['L1/linear.py', 'L1/gelu_and_mul.py']),
    ('MLP', ['L1/linear.py']),
    ('FeedForward', ['L1/linear.py']),
    ('FFN', ['L1/linear.py']),
    ('Mlp', ['L1/linear.py']),
    ('Intermediate', ['L1/linear.py']),
    # Norm variants (when kb_files left empty by subagent)
    ('RMSNorm', ['L1/rms_norm.py']),
    ('LayerNormPerHead', ['L1/layer_norm.py']),
    ('LayerNorm', ['L1/layer_norm.py']),
    ('GroupNorm', ['L1/group_norm.py']),
    ('BatchNorm2d', ['L1/batch_norm2d.py']),
    # RoPE / position embeddings (compute classes despite having no submodules)
    ('RotaryEmbedding', ['L1/rotary_emb.py']),
    ('RopeEmbedding', ['L1/rotary_emb.py']),
    # Convolutions
    ('Conv1d', ['L1/conv1d.py']),
    ('Conv2d', ['L1/conv2d.py']),
    ('Conv3d', ['L1/conv3d.py']),
]


def is_genuine_wiring_class(class_name: str) -> bool:
    """Classes whose forward is pure orchestration over other classes' forwards.
    Distinguished from compute classes that happen to lack a kb-nano file."""
    WIRING_SUFFIXES = (
        'DecoderLayer', 'EncoderLayer', 'Layer', 'Block', 'Stage',
        'Encoder', 'Decoder', 'Stack', 'Transformer',
        'Model', 'PreTrainedModel',
        'Pooler', 'Head', 'Classifier',
        'Embeddings',  # token/pos sum + layer_norm wiring
        'DropPath',    # training-only stochastic depth
    )
    # Catch *ForXxx separately
    if re.search(r'For[A-Z]\w+$', class_name):
        return True
    return any(class_name.endswith(s) for s in WIRING_SUFFIXES)


def decompose_compute_class(class_name: str) -> list[str]:
    """For a compute class with empty kb_nano_files, return L1 decomposition.
    Returns list of 'L1/foo.py' paths."""
    for pattern, paths in CLASS_NAME_DECOMPOSE_RULES:
        if class_name.endswith(pattern):
            return paths
    return []


# Map torch.nn primitive class names to kb-nano L1 file paths.
# When wiring references one of these (e.g. "LayerNorm"), render as the
# kb-nano file path so the reader sees the actual kernel.
NN_TO_KB_NANO = {
    'LayerNorm': 'tasks/baseline/L1/layer_norm.py',
    'BatchNorm1d': 'tasks/baseline/L1/batch_norm2d.py',  # rank-agnostic
    'BatchNorm2d': 'tasks/baseline/L1/batch_norm2d.py',
    'BatchNorm3d': 'tasks/baseline/L1/batch_norm2d.py',
    'GroupNorm': 'tasks/baseline/L1/group_norm.py',
    'Linear': 'tasks/baseline/L1/linear.py',
    'Embedding': 'tasks/baseline/L1/embedding.py',
    'Conv1d': 'tasks/baseline/L1/conv1d.py',
    'Conv2d': 'tasks/baseline/L1/conv2d.py',
    'Conv3d': 'tasks/baseline/L1/conv3d.py',
    'ConvTranspose1d': 'tasks/baseline/L1/conv_transpose1d.py',
    'ConvTranspose2d': 'tasks/baseline/L1/conv_transpose2d.py',
    'ConvTranspose3d': 'tasks/baseline/L1/conv_transpose3d.py',
    'MaxPool1d': 'tasks/baseline/L1/max_pool1d.py',
    'MaxPool2d': 'tasks/baseline/L1/max_pool2d.py',
    'AvgPool1d': 'tasks/baseline/L1/avg_pool1d.py',
    'AvgPool2d': 'tasks/baseline/L1/avg_pool2d.py',
    'AdaptiveAvgPool1d': 'tasks/baseline/L1/adaptive_avg_pool1d.py',
    'AdaptiveAvgPool2d': 'tasks/baseline/L1/adaptive_avg_pool2d.py',
    'GELU': 'tasks/baseline/L1/gelu.py',
    'SiLU': 'tasks/baseline/L1/silu.py',
    'ReLU': 'tasks/baseline/L1/relu.py',
    'LeakyReLU': 'tasks/baseline/L1/leaky_relu.py',
    'Sigmoid': 'tasks/baseline/L1/sigmoid.py',
    'Tanh': 'tasks/baseline/L1/tanh.py',
    'Softmax': 'tasks/baseline/L1/softmax.py',
    'Hardsigmoid': 'tasks/baseline/L1/hardsigmoid.py',
    'Hardswish': 'tasks/baseline/L1/hardswish.py',
    'ELU': 'tasks/baseline/L1/elu.py',
    'Mish': 'tasks/baseline/L1/mish.py',
    'LSTM': 'tasks/baseline/L1/lstm.py',
}


def fmt_wiring(submodules: list[tuple[str, str]], own_classes: set[str],
               own_class_kernels: dict[str, list[str]],
               act_kernel: str = 'activation.py') -> str:
    """Render wiring with kb-nano kernels DIRECTLY (not HF class names).

    Rules:
      - If sub-module class is in own folder AND has a kb_nano_files mapping
        in own_class_kernels: render those kb-nano paths directly.
      - If sub-module class is in own folder but is itself wiring (empty
        kernels): try AST-resolving recursively (one level); else keep HF name.
      - If sub-module is a torch.nn primitive: render as kb-nano L1 file path.
      - Otherwise (cross-arch HF parent class with no row in this folder):
        render as the HF class name (best we can do; reader can grep).
      - Skip Dropout/Identity/ModuleList/etc.
    """
    if not submodules:
        return r'\textit{composes (wiring)}'

    SKIP_TRIVIAL = {'Dropout', 'Identity', 'GradientCheckpointingLayer',
                    'ModuleList', 'ModuleDict', 'ParameterList', 'Parameter',
                    'Sequential', 'Buffer',
                    # Padding ops (parameter-free, not a kernel)
                    'ZeroPad2d', 'ZeroPad1d', 'ZeroPad3d',
                    'ConstantPad1d', 'ConstantPad2d', 'ConstantPad3d',
                    'ReflectionPad1d', 'ReflectionPad2d', 'ReplicationPad2d',
                    # Pure elementwise activations / non-kernel torch ops
                    'Sigmoid', 'Tanh',
                    # MultiheadAttention is wrapped at higher level usually
                    }

    # Count instances per (rendering_key) preserving first-seen order.
    counts = {}
    order = []
    rendered_for = {}

    def _add(key, rendered):
        if key not in counts:
            counts[key] = 0
            order.append(key)
            rendered_for[key] = rendered
        counts[key] += 1

    for attr, cls in submodules:
        if cls in SKIP_TRIVIAL:
            continue
        # ACT2FN-resolved activation: emit synthetic L1 path
        if cls == '__ACT2FN__':
            kb_path = f'tasks/baseline/L1/{act_kernel}'
            _add(('kb_nano', kb_path), fmt_kb_path(kb_path))
            continue
        # Same-folder HF class with a kb-nano mapping → resolve to kb-nano paths
        if cls in own_classes and cls in own_class_kernels and own_class_kernels[cls]:
            for kb_path in own_class_kernels[cls]:
                _add(('kb_nano', kb_path), fmt_kb_path(kb_path))
            continue
        # Same-folder HF class that is itself wiring (no kb_nano_files)
        if cls in own_classes:
            _add(('hf_wiring', cls),
                 r'\texttt{\seqsplit{' + le(cls) + r'}} \textit{(wiring)}')
            continue
        # Torch primitive → kb-nano L1 file path
        if cls in NN_TO_KB_NANO:
            _add(('kb_nano', NN_TO_KB_NANO[cls]), fmt_kb_path(NN_TO_KB_NANO[cls]))
            continue
        # Cross-arch HF class (no own-folder row) — keep name
        _add(('hf_other', cls),
             r'\texttt{\seqsplit{' + le(cls) + '}}')

    if not counts:
        return r'\textit{composes (wiring)}'

    parts = []
    for key in order:
        n = counts[key]
        rendered = rendered_for[key]
        parts.append(rendered if n == 1 else f'{n}\\,$\\times$\\,{rendered}')

    # If the wiring resolves to a single kb-nano kernel (count=1), render it
    # directly without the "composes:" prefix — there is nothing to compose.
    if len(order) == 1 and counts[order[0]] == 1 and order[0][0] == 'kb_nano':
        return rendered_for[order[0]]

    if len(parts) > 8:
        return r'\textit{composes:} ' + ' + '.join(parts[:8]) + r' \textit{+ ' + str(len(parts)-8) + r' more}'
    return r'\textit{composes:} ' + ' + '.join(parts)


# --- Task-head collapsing ----

def is_collapsible_head(class_name: str) -> str | None:
    """Return the head suffix if class is a collapsible task head; else None."""
    for suffix in COLLAPSE_HEADS:
        if class_name.endswith(suffix):
            return suffix
    return None


# --- Main build ----

def expand_from_modeling(folder, data):
    """For folders where the subagent read modular_<f>.py (which only shows
    diffs from the parent), augment with classes from modeling_<f>.py
    (the expanded form that runs at inference)."""
    src = data.get('src_file', '') or ''
    if not src.startswith('modular_'):
        return  # Already used modeling
    # Find modeling file
    folder_name = folder.split('/')[0]
    short = folder.split('/')[-1] if '/' in folder else folder_name
    modeling_path = HF / folder_name / f'modeling_{short}.py'
    if not modeling_path.exists():
        return
    try:
        tree = ast.parse(modeling_path.read_text())
    except SyntaxError:
        return
    existing_names = {c['name'] for c in data.get('classes', [])}
    # Map existing class -> kb-nano files (for inferring children's mappings)
    existing_kernels = {c['name']: c.get('kb_nano_files', []) for c in data.get('classes', [])}

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name in existing_names:
            continue
        if node.name.startswith('_'):
            continue
        # Skip boilerplate
        if re.match(r'^(.*PreTrainedModel|.*Config|.*Output(With\w+)?|.*Cache$|.*Mixin)$', node.name):
            continue
        bases = []
        for b in node.bases:
            try:
                bases.append(ast.unparse(b).split('.')[-1])
            except Exception:
                pass
        # Skip if all bases are pure boilerplate (object, IntEnum, etc.)
        if bases and all(b in {'PreTrainedModel', 'GenerationMixin', 'BackboneMixin',
                                'ProcessorMixin', 'BaseImageProcessor',
                                'IntEnum', 'object', 'ModelOutput'} for b in bases):
            continue

        # Infer kb-nano mapping from name pattern + inherited parent's mapping.
        kb_files = []
        rationale = '[expanded from modeling] '
        # First try: look up parent class kernels in existing_kernels
        for b in bases:
            if b in existing_kernels and existing_kernels[b]:
                kb_files = list(existing_kernels[b])
                rationale += f'inherits {b}; uses parent kernels'
                break

        # Fallback: pattern-match class name to common kernels
        if not kb_files:
            n = node.name
            if n.endswith('RMSNorm'):
                kb_files = ['tasks/baseline/L1/rms_norm.py']
            elif n.endswith('LayerNorm') and 'T5' in n:
                kb_files = ['tasks/baseline/L1/t5_layer_norm.py']
            elif n.endswith('LayerNorm'):
                kb_files = ['tasks/baseline/L1/layer_norm.py']
            elif re.search(r'(Rotary|Rope)Embedding$', n):
                kb_files = ['tasks/baseline/L1/rotary_emb.py']
            elif n.endswith('Attention') and not any(s in n for s in ('Self', 'Cross', 'Pool')):
                # Decompose to L1
                kb_files = ['tasks/baseline/L1/linear.py',
                            'tasks/baseline/L1/dense_attention.py',
                            'tasks/baseline/L1/store_kvcache.py']
                rationale += '(decompose; no exact L2 match)'

        data['classes'].append({
            'name': node.name,
            'line': node.lineno,
            'bases': bases,
            'kb_nano_files': kb_files,
            'rationale': rationale.strip(),
            '_from_modeling': str(modeling_path),
        })


def main():
    # Load CSV verdicts
    csv_rows = list(csv.DictReader(open(COVERAGE_CSV)))
    folder_status = {r['hf_folder']: r['support_status'] for r in csv_rows}

    # Load all 16 batch JSONs
    all_data = {}
    for i in range(1, 17):
        p = BATCH_DIR / f'audit_batch_{i:02d}.json'
        if p.exists():
            d = json.load(open(p))
            for k, v in d.items():
                if k.startswith('_'):
                    continue
                all_data[k] = v

    print(f'Loaded {len(all_data)} folder/sub-folder keys from 16 batches')

    # Expand modular-only folders with classes from modeling
    n_expanded = 0
    n_added = 0
    for folder, data in all_data.items():
        before = len(data.get('classes', []))
        expand_from_modeling(folder, data)
        after = len(data.get('classes', []))
        if after > before:
            n_expanded += 1
            n_added += (after - before)
    print(f'Expanded {n_expanded} modular-only folders; added {n_added} classes from modeling files')

    # JSON post-pass: clear kb_nano_files for classes whose rationale clearly
    # indicates they are wiring (e.g. "wiring: ..."). The subagent sometimes
    # set a partial L2 mapping on a wiring class.
    n_wiring_cleared = 0
    for folder, data in all_data.items():
        for c in data.get('classes', []):
            rat = (c.get('rationale') or '').lower().strip()
            if (rat.startswith('wiring') or 'wiring class' in rat
                    or rat.startswith('wires ')) and c.get('kb_nano_files'):
                c['kb_nano_files'] = []
                n_wiring_cleared += 1
    print(f'Cleared kb_nano_files on {n_wiring_cleared} classes whose rationale says "wiring"')

    # JSON post-pass: detect BERT-style *Attention wrapper (`self.self = X +
    # self.output = Y`) and assign L2/encoder_attention.py if not already set.
    # This way wiring references in *Layer classes resolve to the right file.
    n_bert_attn = 0
    for folder, data in all_data.items():
        if '/' in folder:
            base_folder = folder.split('/', 1)[0]
        else:
            base_folder = folder
        src_file = data.get('src_file') or ''
        # Prefer modeling_*.py for AST extraction
        candidates = []
        if src_file:
            candidates.append(HF / base_folder / src_file)
        candidates.append(HF / base_folder / f"modeling_{base_folder}.py")
        path_to_use = None
        for p in candidates:
            if p.exists():
                path_to_use = p
                break
        if not path_to_use:
            continue
        for c in data.get('classes', []):
            cname = c['name']
            if not cname.endswith('Attention') or cname.endswith(('SelfAttention', 'CrossAttention')):
                continue
            if c.get('kb_nano_files'):
                continue
            subs = get_init_submodules(path_to_use, cname)
            attrs = {a: cls for a, cls in subs}
            if (attrs.get('self', '').endswith('SelfAttention')
                    and attrs.get('output', '').endswith('SelfOutput')):
                c['kb_nano_files'] = ['tasks/baseline/L2/encoder_attention.py']
                n_bert_attn += 1
    print(f'Assigned L2/encoder_attention.py to {n_bert_attn} BERT-style *Attention wrappers')

    # Generate tex
    lines = []
    lines.append(r'\documentclass[10pt]{article}')
    lines.append(r'\usepackage[margin=0.5in,landscape,paperwidth=14in,paperheight=11in]{geometry}')
    lines.append(r'\usepackage{longtable, booktabs, array, pifont, multirow, seqsplit}')
    lines.append(r'\newcommand{\cmark}{\ding{51}}')
    lines.append(r'\newcommand{\xmark}{\ding{55}}')
    lines.append(r'\newcommand{\fastkernels}{FastKernels}')
    lines.append(r'\begin{document}')
    lines.append('')
    lines.append(r'\section*{HF Transformers $\to$ \fastkernels{} per-class kernel mapping (full audit)}')
    lines.append('')
    lines.append(r'\noindent')
    lines.append(r'\textbf{Status legend:} $\bullet$ kb-nano L4 pipeline ships; '
                 r'\cmark{} every kernel exists, only L4 wiring remains; '
                 r'\textbf{P} partial; \xmark{} unsupported.')
    lines.append('')
    lines.append(r'\textbf{Mapping legend:} '
                 r'\texttt{L<n>/file.py} -- the class has its own kernel and maps directly to that file. '
                 r'\{f1, f2\} -- the class is itself a small composition of those L1 ops (no exact L2 match). '
                 r'\textit{composes:} \textit{X + Y + Z} -- the class has no kernel of its own; '
                 r'its \texttt{forward} just calls the listed sub-modules in sequence. '
                 r'When a sub-module is itself a kb-nano kernel we render the kernel path; '
                 r'when it is another wiring class in the same folder we render its name suffixed with \textit{(wiring)} '
                 r'(see that class\textquotesingle{}s own row for its decomposition). '
                 r'Multiplicity is shown as \textit{n$\times$kernel}.')
    lines.append('')
    lines.append(r'{\scriptsize')
    lines.append(r'\setlength{\tabcolsep}{4pt}')
    lines.append(r'\renewcommand{\arraystretch}{1.18}')
    lines.append(r'\begin{longtable}{@{}p{3.0cm} c p{6.5cm} p{14.5cm}@{}}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{HF folder} & \textbf{Status} & \textbf{HF class} & \textbf{\fastkernels{} mapping} \\')
    lines.append(r'\midrule')
    lines.append(r'\endfirsthead')
    lines.append(r'\multicolumn{4}{l}{\small\textit{(continued)}} \\')
    lines.append(r'\toprule')
    lines.append(r'\textbf{HF folder} & \textbf{Status} & \textbf{HF class} & \textbf{\fastkernels{} mapping} \\')
    lines.append(r'\midrule')
    lines.append(r'\endhead')
    lines.append(r'\bottomrule')
    lines.append(r'\endfoot')

    n_folders = 0
    n_rows = 0
    for folder_key in sorted(all_data.keys()):
        data = all_data[folder_key]
        # folder_key is either 'bert' or 'blip/blip_text'
        if '/' in folder_key:
            base_folder, sub = folder_key.split('/', 1)
        else:
            base_folder = folder_key
            sub = None
        status = folder_status.get(base_folder, 'composable')
        marker = STATUS_MARKER.get(status, r'\cmark')

        src_file = data.get('src_file', '')
        # Resolve HF source path
        hf_path = HF / base_folder / src_file if src_file else None
        # Resolve the folder's default activation kernel (config.hidden_act)
        act_kernel = _resolve_activation_default(HF / base_folder)

        # Filter out non-module boilerplate that shouldn't be in audit output
        # (Kwargs dataclasses, Config dataclasses, Loss classes, etc.)
        BOILERPLATE_SUFFIXES = (
            'Kwargs', 'Config', 'Cache', 'Mixin', 'PreTrainedModel',
            'OutputWithPast', 'OutputWithHiddenStates', 'OutputWithCrossAttentions',
            'Loss', 'NormalOutput', 'StudentTOutput', 'NegativeBinomialOutput',
        )
        BOILERPLATE_EXACT = set()
        all_classes = data.get('classes', [])
        classes = [c for c in all_classes
                   if not any(c['name'].endswith(s) for s in BOILERPLATE_SUFFIXES)
                   and c['name'] not in BOILERPLATE_EXACT]
        own_classes = {c['name'] for c in classes}
        # Map: HF class name -> list of kb-nano files (used to resolve refs in wiring).
        # When a class has empty kb_nano_files but is a compute class, fill in
        # the L1 decomposition so wiring references render kernels (not "(wiring)").
        own_class_kernels = {}
        for c in classes:
            files = c.get('kb_nano_files', [])
            if not files and not is_genuine_wiring_class(c['name']):
                decomp = decompose_compute_class(c['name'])
                if decomp:
                    files = ['tasks/baseline/' + p for p in decomp]
            own_class_kernels[c['name']] = files

        # --- Apply collapse: group *ForXxx pure heads ---
        kept_classes = []
        collapsed_heads = []  # list of (name, suffix)
        for c in classes:
            cname = c['name']
            files = c.get('kb_nano_files', [])
            head_suffix = is_collapsible_head(cname)
            if head_suffix and not files:
                collapsed_heads.append((cname, head_suffix))
            else:
                kept_classes.append(c)

        # Cross-arch parents to ignore (built-ins + boilerplate)
        TORCH_BUILTINS = {
            'nn.Module', 'nn.Embedding', 'nn.Linear', 'nn.LayerNorm',
            'nn.Conv1d', 'nn.Conv2d', 'nn.Conv3d', 'nn.Dropout',
            'nn.ReLU', 'nn.GELU', 'nn.Identity', 'nn.ModuleList', 'nn.ModuleDict',
            'Module', 'Embedding', 'Linear', 'LayerNorm',
            'Conv1d', 'Conv2d', 'Conv3d', 'Dropout', 'ReLU', 'GELU', 'Identity',
            'ModuleList', 'ModuleDict',
        }
        SKIP_INHERIT = TORCH_BUILTINS | {
            'PreTrainedModel', 'GenerationMixin', 'BackboneMixin',
            'GradientCheckpointingLayer', 'object', 'IntEnum',
            'ModelOutput', 'BaseModelOutput', 'ProcessorMixin',
            'BaseImageProcessor', 'autograd.Function', 'Function',
        }

        folder_camel = ''.join(p.title() for p in base_folder.split('_'))

        # Generate rows
        rows_for_folder = []
        for c in kept_classes:
            cname = c['name']
            files = c.get('kb_nano_files', [])
            if files:
                paths = ', '.join(fmt_kb_path(p) for p in files)
                if len(files) > 1:
                    paths = r'\{' + paths + r'\}'
                bases = c.get('bases', [])
                # Add inheritance hint ONLY for cross-arch HF parents (not nn.X / boilerplate)
                cross_arch = [b for b in bases
                              if b not in SKIP_INHERIT
                              and not b.startswith(folder_camel)
                              and b not in own_classes]
                if cross_arch:
                    paths += r' \quad (inherits ' + r'\texttt{\seqsplit{' + le(cross_arch[0]) + '}})'
                mapping = paths
            else:
                # For folders sourced from modular_*.py: ALWAYS prefer the
                # modeling_*.py file for AST submodule extraction, since
                # modular only carries the inheritance diff (e.g. an override
                # of __init__ that only mentions overridden norms, not the
                # inherited attention/MLP). The full expanded class lives in
                # modeling_<folder>.py.
                class_src = c.get('_from_modeling')
                if class_src:
                    resolve_path = Path(class_src)
                elif src_file and src_file.startswith('modular_'):
                    modeling_candidate = HF / base_folder / f"modeling_{base_folder}.py"
                    resolve_path = modeling_candidate if modeling_candidate.exists() else hf_path
                else:
                    resolve_path = hf_path

                def _resolve_submodules(path_primary):
                    subs = get_init_submodules(path_primary, cname) if path_primary else []
                    if subs:
                        return subs
                    # Last fallback: try the original src_file
                    if hf_path and hf_path != path_primary:
                        return get_init_submodules(hf_path, cname)
                    return subs

                # BERT-style *Attention pattern: a class named *Attention whose
                # __init__ has exactly self.self = SelfAttention + self.output =
                # SelfOutput maps cleanly to L2/encoder_attention.py:EncoderAttention.
                bert_attn_match = False
                if cname.endswith('Attention') and not cname.endswith(('SelfAttention', 'CrossAttention')):
                    subs = _resolve_submodules(resolve_path)
                    attrs = {a for a, _ in subs}
                    if 'self' in attrs and 'output' in attrs:
                        sub_classes = {a: cls for a, cls in subs}
                        if (sub_classes.get('self', '').endswith('SelfAttention')
                                and sub_classes.get('output', '').endswith('SelfOutput')):
                            mapping = fmt_kb_path('tasks/baseline/L2/encoder_attention.py')
                            bert_attn_match = True

                if bert_attn_match:
                    pass  # mapping already set
                # Two cases: (a) genuine wiring class; (b) compute class with no kb-nano file.
                elif is_genuine_wiring_class(cname):
                    submodules = _resolve_submodules(resolve_path)
                    mapping = fmt_wiring(submodules, own_classes, own_class_kernels, act_kernel=act_kernel)
                else:
                    # Compute class without kb-nano match: decompose to L1 ops.
                    decomp = decompose_compute_class(cname)
                    if decomp:
                        paths = ', '.join(fmt_kb_path('tasks/baseline/' + p) for p in decomp)
                        if len(decomp) > 1:
                            paths = r'\{' + paths + r'\}'
                        # Add caveat that no exact L2 match exists
                        mapping = paths + r' \quad \textit{(compose; no exact L2 match)}'
                    else:
                        # Truly unmatchable: fall back to wiring
                        submodules = _resolve_submodules(resolve_path)
                        mapping = fmt_wiring(submodules, own_classes, own_class_kernels, act_kernel=act_kernel)
            rows_for_folder.append((cname, mapping))

        # Drop pure-wiring rows: a row whose mapping has no kb-nano kernel paths
        # (only HF class wiring references, or empty "composes (wiring)").
        # These add no coverage information — every leaf kernel they would
        # transitively reach is already mapped on its own row.
        def _has_kernel(mapping_str: str) -> bool:
            return bool(re.search(r'\\texttt\{L\d/\}', mapping_str))
        rows_for_folder = [(n, m) for (n, m) in rows_for_folder
                           if _has_kernel(m) or n.startswith('task heads')]

        # Add one collapsed row for task heads
        if collapsed_heads:
            n = len(collapsed_heads)
            head_names = sorted(set(h[1] for h in collapsed_heads))
            head_suffix_list = ', '.join(head_names[:5])
            if len(head_names) > 5:
                head_suffix_list += f' (+{len(head_names)-5} more)'
            label = f'task heads ({n})'
            example_class = collapsed_heads[0][0]
            base_model_name = example_class.replace(collapsed_heads[0][1], 'Model')
            mapping = (r'\textit{compose:} \texttt{\seqsplit{'
                       + le(base_model_name)
                       + r'}} + ' + fmt_kb_path('tasks/baseline/L1/linear.py')
                       + r' (per-task head) \quad \textit{[' + le(head_suffix_list) + ']}')
            rows_for_folder.append((label, mapping))

        n = len(rows_for_folder)
        if n == 0:
            continue
        n_folders += 1
        n_rows += n

        folder_label = short_modeling_label(base_folder, src_file) if not sub else \
                       r'\texttt{\seqsplit{' + le(base_folder + '/' + sub) + '}}'

        # Render rows
        lines.append(f'%' + '=' * 70)
        lines.append(f'% {folder_key} ({status})')
        lines.append(f'%' + '=' * 70)
        if n == 1:
            cname, mapping = rows_for_folder[0]
            cname_fmt = texttt(cname) if not cname.startswith('task heads') else r'\textit{' + cname + '}'
            lines.append(f'  {folder_label} & {marker} & {cname_fmt} & {mapping} \\\\')
        else:
            for i, (cname, mapping) in enumerate(rows_for_folder):
                cname_fmt = texttt(cname) if not cname.startswith('task heads') else r'\textit{' + cname + '}'
                if i == 0:
                    fc = '\\multirow{' + str(n) + '}{*}{' + folder_label + '}'
                    sc = '\\multirow{' + str(n) + '}{*}{' + marker + '}'
                else:
                    fc = ''
                    sc = ''
                lines.append(f'  {fc} & {sc} & {cname_fmt} & {mapping} \\\\')
        lines.append(r'\midrule')

    lines.append(r'\end{longtable}')
    lines.append(r'}')
    lines.append(r'\end{document}')

    OUT_TEX.write_text('\n'.join(lines) + '\n')
    print(f'Wrote {OUT_TEX}')
    print(f'  Folders rendered: {n_folders}')
    print(f'  Total rows: {n_rows}')


if __name__ == '__main__':
    main()
