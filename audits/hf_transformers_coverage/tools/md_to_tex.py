"""Convert agent-produced manual_audit_shard_*.md files into the final
appendix LaTeX longtable.

Markdown format per folder (one or more per file):

    ## <folder>
    - **src**: ...
    - **hidden_act**: ...
    - **status**: composable | kb_nano_l4 | partial | unsupported
    - **classes**:
      - **`ClassName`** [compute|wiring] [inherits `Parent`]: <mapping>
    - **task heads (N)**: ForX, ForY — base + linear (per-task)

Output: same 4-col longtable as before:
    HF folder | Status | HF class | FastKernels mapping
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path('/home/olu/kb_nano')
SHARD_DIR = REPO / 'audits/hf_transformers_coverage/tools'
OUT_TEX = REPO / 'audits/hf_transformers_coverage/hf_coverage_rows.tex'

STATUS_MARKER = {
    'kb_nano_l4': r'$\bullet$',
    'composable': r'\cmark',
    'partial': r'\textbf{P}',
    'unsupported': r'\xmark',
    'not_inference_required': '--',
}


def le(s: str) -> str:
    """Escape LaTeX special chars for safe insertion into a tex document."""
    out = []
    for ch in s:
        if ch == '\\':
            out.append(r'\textbackslash{}')
        elif ch == '_':
            out.append(r'\_')
        elif ch == '&':
            out.append(r'\&')
        elif ch == '%':
            out.append(r'\%')
        elif ch == '#':
            out.append(r'\#')
        elif ch == '$':
            out.append(r'\$')
        elif ch == '^':
            out.append(r'\textasciicircum{}')
        elif ch == '~':
            out.append(r'\textasciitilde{}')
        elif ch == '{':
            out.append(r'\{')
        elif ch == '}':
            out.append(r'\}')
        else:
            out.append(ch)
    return ''.join(out)


def texttt(s: str) -> str:
    """Class names: short → \\texttt, long → \\seqsplit."""
    esc = le(s)
    if len(s) > 14:
        return r'\texttt{\seqsplit{' + esc + '}}'
    return r'\texttt{' + esc + '}'


def fmt_kb_path(path: str) -> str:
    """L2/foo.py -> \\texttt{L2/}\\allowbreak\\texttt{foo.py}"""
    if '/' not in path:
        return r'\texttt{' + le(path) + '}'
    layer, fname = path.split('/', 1)
    return r'\texttt{' + le(layer) + r'/}\allowbreak\texttt{' + le(fname) + '}'


_NN_TO_KB_NANO = {
    'nn.Linear': 'L1/linear.py',
    'nn.Embedding': 'L1/embedding.py',
    'nn.LayerNorm': 'L1/layer_norm.py',
    'nn.Conv1d': 'L1/conv1d.py',
    'nn.Conv2d': 'L1/conv2d.py',
    'nn.Conv3d': 'L1/conv3d.py',
    'nn.GroupNorm': 'L1/group_norm.py',
    'nn.BatchNorm2d': 'L1/batch_norm2d.py',
    'nn.BatchNorm1d': 'L1/batch_norm1d.py',
    'nn.MaxPool2d': 'L1/max_pool2d.py',
    'nn.AvgPool2d': 'L1/avg_pool2d.py',
    'nn.AvgPool1d': 'L1/avg_pool1d.py',
    'nn.AdaptiveAvgPool2d': 'L1/adaptive_avg_pool2d.py',
    'nn.AdaptiveAvgPool1d': 'L1/adaptive_avg_pool1d.py',
    'nn.ConvTranspose1d': 'L1/conv_transpose1d.py',
    'nn.ConvTranspose2d': 'L1/conv_transpose2d.py',
    'nn.ConvTranspose3d': 'L1/conv_transpose3d.py',
    'nn.Upsample': 'L1/interpolate.py',
    'nn.MultiheadAttention': 'L1/dense_attention.py',
    'nn.GELU': 'L1/gelu.py',
    'nn.SiLU': 'L1/silu.py',
    'nn.ReLU': 'L1/relu.py',
    'nn.Tanh': 'L1/tanh.py',
    'nn.Sigmoid': 'L1/sigmoid.py',
    'nn.Softmax': 'L1/softmax.py',
    'LayerNorm': 'L1/layer_norm.py',
    'Linear': 'L1/linear.py',
    'Embedding': 'L1/embedding.py',
}

# Torch primitives that are no-op for inference / not a kernel — skip
_NN_SKIP = {
    'nn.Dropout', 'nn.Identity', 'nn.ModuleList', 'nn.ModuleDict',
    'nn.Sequential', 'nn.Parameter', 'nn.ParameterList',
    'nn.ZeroPad2d', 'nn.ZeroPad1d', 'nn.ConstantPad2d',
    'Dropout', 'Identity', 'ModuleList', 'ModuleDict',
    'Sequential', 'Parameter', 'ZeroPad2d',
}


def fmt_mapping(raw: str, own_class_kernels: dict | None = None) -> str:
    """Convert agent's mapping syntax to LaTeX. Resolves same-folder wiring
    references to their kernel paths via own_class_kernels lookup."""
    if not raw:
        return r'\textit{wiring}'
    own_class_kernels = own_class_kernels or {}

    raw = raw.strip()
    is_wiring = raw.lower().startswith('wires')
    body = raw[len('wires'):].strip() if is_wiring else raw
    direct = ''
    if '; direct ' in body:
        body, direct = body.split('; direct ', 1)

    parts = []

    if is_wiring:
        # Parse refs along with optional (xN) / (×N) multiplicity annotations.
        # The (xN) may appear EITHER outside the backticks (`Cls` (×N)) OR
        # inside them (`Cls (×N)`) — agents are inconsistent. Capture the
        # whole backtick contents, then strip any trailing parenthetical.
        ref_pattern = re.compile(
            r'`([^`]+)`'
            r'(?:\s*\([xX×]\s*(\d+)[^)]*\))?'
        )
        for m in ref_pattern.finditer(body):
            inner = m.group(1).strip()
            outer_count = int(m.group(2)) if m.group(2) else None
            # Look for inline (×N) inside the backticks
            inline_m = re.match(r'^([A-Za-z][A-Za-z0-9_/.]*)\s*\([xX×]\s*(\d+)[^)]*\)\s*$', inner)
            if inline_m:
                raw_ref = inline_m.group(1)
                count = int(inline_m.group(2))
            else:
                # Strip any trailing parenthetical (e.g. "Cls (note)")
                no_paren = re.match(r'^([A-Za-z][A-Za-z0-9_/.]*)', inner)
                raw_ref = no_paren.group(1) if no_paren else inner
                count = outer_count or 1
            if raw_ref in _NN_SKIP:
                continue
            if raw_ref in _NN_TO_KB_NANO:
                for _ in range(count):
                    parts.append(fmt_kb_path(_NN_TO_KB_NANO[raw_ref]))
                continue
            if raw_ref.endswith('.py') and '/' in raw_ref:
                for _ in range(count):
                    parts.append(fmt_kb_path(raw_ref))
            elif raw_ref in own_class_kernels and own_class_kernels[raw_ref]:
                for _ in range(count):
                    for kp in own_class_kernels[raw_ref]:
                        parts.append(fmt_kb_path(kp))
            else:
                # Unresolvable ref: cross-arch class or agent missed a row.
                # Show the class name in italics without "(wiring)" suffix —
                # the reader knows it's an opaque dependency.
                for _ in range(count):
                    parts.append(r'\texttt{\seqsplit{' + le(raw_ref) + r'}}')
    else:
        # Compute class: extract all kb-nano paths from the mapping body.
        # Be robust to agents prefixing pieces with class-name notes
        # (e.g. "q/k/v/o `L1/linear.py` + ...").
        path_pat = re.compile(
            r'(?:`)?([A-Za-z][A-Za-z0-9]*/[A-Za-z0-9_]+\.py)(?:`)?'
            r'(?:\s*[`)]?\s*\([xX×]\s*(\d+)[^)]*\))?'
        )
        for m in path_pat.finditer(body):
            path = m.group(1)
            count = int(m.group(2)) if m.group(2) else 1
            for _ in range(count):
                parts.append(fmt_kb_path(path))

    if direct:
        # Match `path/to.py` optionally followed by (xN) / (×N) multiplicity
        direct_pat = re.compile(
            r'`([A-Za-z][A-Za-z0-9_/.]*\.py)`'
            r'(?:\s*\([xX×]\s*(\d+)[^)]*\))?'
        )
        for m in direct_pat.finditer(direct):
            raw_ref = m.group(1)
            count = int(m.group(2)) if m.group(2) else 1
            for _ in range(count):
                parts.append(fmt_kb_path(raw_ref))

    if not parts:
        return r'\textit{wiring}'

    # Dedupe with multiplicities
    counts = {}
    order = []
    for p in parts:
        if p not in counts:
            counts[p] = 0
            order.append(p)
        counts[p] += 1
    rendered = []
    for p in order:
        n = counts[p]
        rendered.append(p)  # unique kernels only — drop multiplicity

    return ' + '.join(rendered)


def parse_md(text: str) -> list[dict]:
    """Parse markdown shard file → list of {folder, status, classes:[...], task_heads:[...]}."""
    folders = []
    current = None
    for line in text.split('\n'):
        if line.startswith('## '):
            if current:
                folders.append(current)
            current = {
                'folder': line[3:].strip(),
                'status': 'composable',
                'classes': [],
                'task_heads': [],
                'missing_reasons': [],
            }
            continue
        if not current:
            continue
        m = re.match(r'^- \*\*status\*\*:\s*(\w+)', line)
        if m:
            current['status'] = m.group(1)
            continue
        m = re.match(r'^\s*-\s*\*\*`([^`]+)`\*\*\s*(\[[^\]]+\])?\s*(\[inherits[^\]]+\])?:?\s*(.*)$', line)
        if m:
            cname = m.group(1)
            type_tag = m.group(2) or ''
            inherit_tag = m.group(3) or ''
            mapping = m.group(4).strip()
            current['classes'].append({
                'name': cname,
                'is_wiring': '[wiring' in type_tag.lower(),
                'inherits': re.search(r'inherits\s*`([^`]+)`', inherit_tag),
                'mapping': mapping,
            })
            # Extract missing-primitive reasons for non-composable folders
            # Priority signal: explicit "no kb-nano kernel" > vague "no L2 match"
            priority = 0
            if re.search(r'no kb-nano kernel|no kb-nano equivalent|requires.*library|external.*library|custom CUDA|new kernel|FFT|LSH|wkv', mapping, re.I):
                priority = 3
            elif re.search(r'no exact L\d|no L\d match|missing', mapping, re.I):
                priority = 1
            # Class-name signal: attention/mixer/scan/SSM are typically the
            # load-bearing missing primitive
            if re.search(r'(Attention|Mixer|SSM|Scan)$', cname):
                priority += 2
            elif re.search(r'(Layer|Block|Mixer)$', cname):
                priority += 1
            if priority > 0:
                short = mapping[:140].replace('`', '').strip()
                current['missing_reasons'].append((priority, cname, short))
            continue
        m = re.match(r'^- \*\*task heads \((\d+)\)\*\*:\s*(.+?)\s*[—-]', line)
        if m:
            current['task_heads'] = {
                'count': int(m.group(1)),
                'list': m.group(2).strip(),
            }
    if current:
        folders.append(current)
    return folders


def render_tex(all_folders: list[dict]) -> str:
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
                 r'\textit{A + B + C} -- the class is implemented by the listed kb-nano files combined '
                 r'(no exact single match). \textit{n$\times$file.py} indicates the same kernel is invoked '
                 r'multiple times within the class.')
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
    for folder in sorted(all_folders, key=lambda f: f['folder']):
        # Build own-class kernel lookup: extract kb-nano paths from each
        # compute class's mapping. Used to resolve wiring refs within the folder.
        own_class_kernels = {}
        # Pass 1: compute classes — extract paths directly from mapping
        for c in folder['classes']:
            if c['mapping'].lower().startswith('wires'):
                continue
            paths = re.findall(r'([A-Za-z][A-Za-z0-9]*/[A-Za-z0-9_]+\.py)', c['mapping'])
            seen = set()
            uniq = []
            for p in paths:
                if p not in seen:
                    seen.add(p)
                    uniq.append(p)
            # Include even classes with no kb-nano paths (e.g., agent noted
            # "no exact L2 match"). These render as the class name when
            # referenced as wiring, with no (wiring) suffix.
            own_class_kernels[c['name']] = uniq

        # Pass 2: iteratively resolve wiring chains. For each wiring class,
        # expand its `wires X, Y; direct L1/Z.py` refs using already-known
        # own_class_kernels. Keep iterating until fixed point.
        max_iter = 5
        for _ in range(max_iter):
            changed = False
            for c in folder['classes']:
                if c['name'] in own_class_kernels:
                    continue
                if not c['mapping'].lower().startswith('wires'):
                    continue
                body = c['mapping'][len('wires'):].strip()
                direct = ''
                if '; direct ' in body:
                    body, direct = body.split('; direct ', 1)
                resolved_paths = []
                all_resolved = True
                for m in re.finditer(r'`([^`]+)`', body):
                    inner = m.group(1).strip()
                    nm = re.match(r'^([A-Za-z][A-Za-z0-9_/.]*)', inner)
                    ref = nm.group(1) if nm else inner
                    if ref in _NN_SKIP:
                        continue
                    if ref in _NN_TO_KB_NANO:
                        resolved_paths.append(_NN_TO_KB_NANO[ref])
                    elif ref.endswith('.py') and '/' in ref:
                        resolved_paths.append(ref)
                    elif ref in own_class_kernels:
                        resolved_paths.extend(own_class_kernels[ref])
                    else:
                        all_resolved = False
                        break
                if not all_resolved:
                    continue
                for p in re.findall(r'([A-Za-z][A-Za-z0-9]*/[A-Za-z0-9_]+\.py)', direct):
                    resolved_paths.append(p)
                if resolved_paths:
                    seen = set()
                    uniq = []
                    for p in resolved_paths:
                        if p not in seen:
                            seen.add(p)
                            uniq.append(p)
                    own_class_kernels[c['name']] = uniq
                    changed = True
            if not changed:
                break

        # Compact mode: drop wiring classes (DecoderLayer/Block/Model/ForXxx
        # etc.) since their info is just the union of compute-class kernels in
        # the same folder. Keep only:
        #   - compute classes with their own direct kernel mapping
        #   - one Model/ForCausalLM/ForConditionalGeneration row per folder
        #     (kept as "structural anchor"; transitively resolved kernels)
        # This reduces row count ~60-70%.
        # Aggressive compaction:
        # 1. Drop ALL wiring classes (Layer/Block/Model/ForXxx etc.)
        # 2. Collapse classes that just inherit unchanged from a parent
        #    (e.g. "MistralRMSNorm(LlamaRMSNorm)") into a single representative
        # 3. Dedupe by kernel-set: if N compute classes in the same folder
        #    map to the same kernel(s), keep only the first
        WIRING_DROP_SUFFIXES = (
            'DecoderLayer', 'EncoderLayer', 'Layer', 'Block', 'Stage',
            'Encoder', 'Decoder', 'Transformer', 'LayerGroup', 'Stack',
            'Pooler', 'PredictionHeadTransform', 'Model',
            'ForCausalLM', 'ForConditionalGeneration', 'ForMaskedLM',
        )
        # Compute the FULL set of kernels referenced anywhere in this folder
        # (compute + wiring). We must ensure every one of these appears in
        # at least one kept row.
        full_kernel_set = set()
        for c in folder['classes']:
            full_kernel_set.update(re.findall(r'L\d/[a-zA-Z0-9_]+\.py', c['mapping']))
            for ref in re.findall(r'`([A-Za-z][A-Za-z0-9_]*)`', c['mapping']):
                if ref in own_class_kernels:
                    full_kernel_set.update(own_class_kernels[ref])

        kept = []
        kept_kernel_set = set()
        seen_kernel_signatures = set()

        def _class_kernels(c):
            paths = set(re.findall(r'L\d/[a-zA-Z0-9_]+\.py', c['mapping']))
            for ref in re.findall(r'`([A-Za-z][A-Za-z0-9_]*)`', c['mapping']):
                if ref in own_class_kernels:
                    paths.update(own_class_kernels[ref])
            return paths

        # First pass: compute classes only, dedupe identical signatures
        for c in folder['classes']:
            cname = c['name']
            mapping = c['mapping']
            if any(cname.endswith(suf) for suf in WIRING_DROP_SUFFIXES):
                continue
            if re.search(r'For[A-Z]\w+$', cname) or cname.endswith('LMHeadModel'):
                continue
            mapping_tex = fmt_mapping(mapping, own_class_kernels)
            if r'\texttt{L' not in mapping_tex:
                continue
            class_kernels = _class_kernels(c)
            sig = tuple(sorted(class_kernels))
            if sig and sig in seen_kernel_signatures:
                continue
            seen_kernel_signatures.add(sig)
            kept.append((cname, c.get('inherits'), mapping_tex))
            kept_kernel_set.update(class_kernels)

        # Second pass: if any kernel is missing from kept rows, find the
        # smallest class (compute or wiring) that introduces those missing
        # kernels and add it back.
        missing = full_kernel_set - kept_kernel_set
        while missing:
            best = None
            best_overlap = 0
            for c in folder['classes']:
                cname = c['name']
                if (cname, c.get('inherits'), None) in [(k[0], k[1], None) for k in kept]:
                    continue
                ck = _class_kernels(c)
                overlap = len(ck & missing)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = c
            if best is None:
                break
            mapping_tex = fmt_mapping(best['mapping'], own_class_kernels)
            if r'\texttt{L' in mapping_tex:
                kept.append((best['name'], best.get('inherits'), mapping_tex))
                ck = _class_kernels(best)
                kept_kernel_set.update(ck)
                missing -= ck
            else:
                break
        # Add task-heads row
        if folder.get('task_heads'):
            th = folder['task_heads']
            label = f'task heads ({th["count"]})'
            base_name = ''.join(p.title() for p in folder['folder'].split('_')) + 'Model'
            mapping = (r'\texttt{\seqsplit{' + le(base_name) + r'}} \textit{(wiring)} + '
                       + fmt_kb_path('L1/linear.py')
                       + r' (per-task head) \quad \textit{[' + le(th['list']) + r']}')
            kept.append((label, None, mapping))
        if not kept:
            # For partial/unsupported folders without any compute class
            # mappings, still render so the status is visible. Synthesize a
            # placeholder row from the folder rationale or a generic note.
            if folder['status'] in ('partial', 'unsupported') and folder.get('missing_reasons'):
                pass  # the (missing) row prepended below will be the only row
            elif folder['status'] in ('partial', 'unsupported'):
                # Fallback note when no missing_reasons were captured
                kept.append(('--missing--', None, r'\textit{(see folder rationale; no compute classes mapped)}'))
            else:
                # Composable wrapper with no compute classes: still render one row
                kept.append((folder['folder'], None, r'\textit{(wiring-only wrapper; delegates to other models)}'))
        n_folders += 1
        n_rows += len(kept)
        # If non-composable, prepend a "missing primitive" annotation row.
        if folder['status'] in ('partial', 'unsupported') and folder.get('missing_reasons'):
            # Pick the highest-priority missing reason
            sorted_reasons = sorted(folder['missing_reasons'], key=lambda x: -x[0])
            _, top_class, top_reason = sorted_reasons[0]
            note = (r'\textit{Missing primitive:} \texttt{' + le(top_class) + r'} -- '
                    + le(top_reason))
            kept.insert(0, ('--missing--', None, note))
        n = len(kept)
        marker = STATUS_MARKER.get(folder['status'], r'\cmark')
        folder_label = texttt(folder['folder'])
        lines.append('%' + '=' * 70)
        lines.append(f'% {folder["folder"]} ({folder["status"]})')
        lines.append('%' + '=' * 70)
        for i, (cname, inherit_match, mapping_tex) in enumerate(kept):
            if cname == '--missing--':
                cname_tex = r'\textit{(missing)}'
            elif cname.startswith('task heads'):
                cname_tex = r'\textit{' + cname + '}'
            else:
                cname_tex = texttt(cname)
            extra = ''
            if inherit_match and cname != '--missing--' and not cname.startswith('task heads'):
                parent = inherit_match.group(1)
                extra = r' \quad (inherits \texttt{\seqsplit{' + le(parent) + '}})'
            mapping_tex = mapping_tex + extra
            # Repeat folder + status on every row (avoids multirow page-break
            # overlap; longtable handles per-row breaks cleanly).
            if i == 0:
                lines.append(f'  {folder_label} & {marker} & {cname_tex} & {mapping_tex} \\\\')
            else:
                lines.append(f'   &  & {cname_tex} & {mapping_tex} \\\\')
        lines.append(r'\midrule')

    lines.append(r'\end{longtable}')
    lines.append(r'}')
    lines.append(r'\end{document}')
    print(f'Folders rendered: {n_folders}')
    print(f'Total rows: {n_rows}')
    return '\n'.join(lines)


def load_reclassifications() -> dict:
    out = {}
    # Initial reclassification (looser paper definition for partial→composable)
    for letter in ('A', 'B'):
        p = SHARD_DIR / f'reclassify_{letter}.md'
        if not p.exists():
            continue
        for line in p.read_text().split('\n'):
            m = re.match(r'^([a-z][a-z0-9_]*):\s*(composable|partial|unsupported|kb_nano_l4)\s*[—-]', line)
            if m:
                out[m.group(1)] = m.group(2)
    # Critical re-audit (final verdict, overrides above for the 28 critical folders)
    for letter in ('A', 'B'):
        p = SHARD_DIR / f'critical_reaudit_{letter}.md'
        if not p.exists():
            continue
        for line in p.read_text().split('\n'):
            m = re.match(r'^([a-z][a-z0-9_]*):\s*(composable|partial|unsupported|kb_nano_l4)\s*[—-]', line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def main():
    all_folders = []
    for i in range(1, 18):
        p = SHARD_DIR / f'manual_audit_shard_{i:02d}.md'
        if not p.exists():
            print(f'Skipping {p.name} (not yet written)')
            continue
        folders = parse_md(p.read_text())
        print(f'  {p.name}: {len(folders)} folders parsed')
        all_folders.extend(folders)

    # Apply paper-definition reclassifications (looser criterion)
    reclass = load_reclassifications()
    n_reclassified = 0
    for f in all_folders:
        new_status = reclass.get(f['folder'])
        if new_status and new_status != f['status']:
            f['status'] = new_status
            n_reclassified += 1
    print(f'Reclassified {n_reclassified} folders to looser paper-definition status')

    tex = render_tex(all_folders)
    OUT_TEX.write_text(tex)
    print(f'Wrote {OUT_TEX}')


if __name__ == '__main__':
    main()
