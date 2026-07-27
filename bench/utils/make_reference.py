#!/usr/bin/env python3
"""Generate self-contained pure-PyTorch semantic references.

``tasks/reference/L{1..4}/<op>.py`` files are *single-file, self-contained*
pure-PyTorch mirrors of the production baselines: no relative imports, no
external kernel libraries (flash-attn, FlashInfer, DeepGEMM, FLA, vLLM CUDA
ops, custom CUDA), and the target class defined **last** so that
``infra.kernel_swapper._find_module_class`` resolves it.

This tool builds those files for higher levels by flattening a baseline's
dependency graph, substituting the pure-PyTorch reference for every dependency
that has one, and emitting each dependency's body exactly once under an
``# Inlined from <path>`` marker -- the same layout as the hand-written
references already in the tree.

Usage:
    python -m fastkernels.bench.utils.make_reference --level 3 --target llama_decoder
    python -m fastkernels.bench.utils.make_reference --level 3 --all
    python -m fastkernels.bench.utils.make_reference --level 3 --target llama_decoder --check

``--check`` regenerates to stdout instead of writing, and ``--verify`` compares
a regenerated file against the committed one by defined-symbol set (used to
validate the tool against the hand-written L1/L2 corpus).

How a dependency's "own body" is recovered
------------------------------------------
A reference file may itself contain ``# Inlined from`` blocks.  Splitting on
those markers is not sufficient, because by convention the file's own
definitions follow the *last* marker's content in the same segment.  So for
each segment we drop the statements whose defined name belongs to that
segment's source file and keep the rest.  Applied uniformly, this recovers
exactly the definitions each file owns.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
KB_ROOT = _HERE.parents[2]
BASELINE = KB_ROOT / "tasks" / "baseline"
REFERENCE = KB_ROOT / "tasks" / "reference"

MARKER = "# Inlined from "

# Modules that must be **imported, not inlined**: they own process-global state
# that the benchmark harness writes into.  ``infra/context.py`` holds the
# forward-context singleton (`_CONTEXT`) that the runner sets via
# ``set_context`` and that every attention module reads via ``get_context``.
# Inlining it gives the reference a *private* singleton which nothing ever
# populates, so `ctx.slot_mapping` stays None and `store_kvcache` dies with
# "'>=' not supported between instances of 'NoneType' and 'int'".
#
# This does not weaken self-containment: the rule is "no external *kernel*
# libraries", and infra/context.py is FastKernels' own plumbing, not a kernel.
IMPORT_NOT_INLINE: dict[str, str] = {
    "infra/context.py": "fastkernels.infra.context",
}

# Libraries a semantic reference must never depend on.
FORBIDDEN_ROOTS = {
    "flash_attn", "flashinfer", "deep_gemm", "fla", "vllm", "sgl_kernel",
    "sglang", "causal_conv1d", "mamba_ssm", "triton", "apex", "xformers",
    "transformer_engine", "torch_scatter", "spconv", "gsplat", "nerfacc",
    "einops",
}


# --------------------------------------------------------------------------
# source-level helpers
# --------------------------------------------------------------------------

def _read(path: Path) -> str:
    return path.read_text()


def _lines(src: str) -> list[str]:
    return src.splitlines()


def _top_level_names(tree: ast.Module) -> set[str]:
    """Names bound by top-level statements (classes, functions, assignments)."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    names.add(tgt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


_DEFINED_CACHE: dict[Path, set[str]] = {}
_DEFINED_VISITING: set[Path] = set()


def defined_names(path: Path) -> set[str]:
    """Top-level names a file defines, ignoring its inlined blocks.

    For a file with markers this is computed recursively: a name defined in
    one of the file's marker sources is *not* owned by this file.  Memoized,
    with a cycle guard -- marker graphs in the hand-written corpus are not
    guaranteed acyclic.
    """
    if not path.is_file():
        return set()
    path = path.resolve()
    if path in _DEFINED_CACHE:
        return _DEFINED_CACHE[path]
    if path in _DEFINED_VISITING:
        # Cycle: treat this file as owning everything it textually defines.
        return _top_level_names(ast.parse(_read(path), filename=str(path)))

    _DEFINED_VISITING.add(path)
    try:
        src = _read(path)
        tree = ast.parse(src, filename=str(path))
        textual = _top_level_names(tree)

        # Ownership is anchored to the baseline: a reference file owns exactly
        # what its baseline counterpart defines, plus private helpers it adds
        # that no inlined block provides.  Without this anchor, reference files
        # that cross-inline each other (several yolov10 modules do) subtract
        # each other's classes and nobody ends up owning them.
        anchored: set[str] = set()
        if path.is_relative_to(REFERENCE):
            counterpart = BASELINE / path.relative_to(REFERENCE)
            if counterpart.is_file():
                anchored = _top_level_names(
                    ast.parse(_read(counterpart), filename=str(counterpart))
                )

        own = set(textual)
        for dep_path in _marker_sources(src):
            resolved = _resolve_marker(dep_path)
            if resolved is not None and resolved.resolve() != path:
                own -= defined_names(resolved)
        own |= (textual & anchored)
    finally:
        _DEFINED_VISITING.discard(path)

    _DEFINED_CACHE[path] = own
    return own


def _marker_sources(src: str) -> list[str]:
    """Paths named by ``# Inlined from <path>`` markers, in file order."""
    out = []
    for line in _lines(src):
        if line.startswith(MARKER):
            out.append(line[len(MARKER):].strip())
    return out


def _resolve_marker(rel: str) -> Path | None:
    p = KB_ROOT / rel
    return p if p.is_file() else None


def _segment_statements(path: Path) -> list[tuple[ast.stmt, str | None]]:
    """Pair each top-level statement with the marker source it sits under.

    ``None`` means the statement is in the pre-marker preamble.
    """
    src = _read(path)
    lines = _lines(src)
    tree = ast.parse(src, filename=str(path))

    # line number (1-based) -> marker source active from that line on
    marker_at: dict[int, str] = {}
    for i, line in enumerate(lines, start=1):
        if line.startswith(MARKER):
            marker_at[i] = line[len(MARKER):].strip()

    def active_marker(lineno: int) -> str | None:
        cur = None
        for ml in sorted(marker_at):
            if ml <= lineno:
                cur = marker_at[ml]
            else:
                break
        return cur

    return [(node, active_marker(node.lineno)) for node in tree.body]


def own_statements(path: Path) -> list[tuple[frozenset[str], str]]:
    """Statements ``path`` owns, as ``(bound_names, source_text)`` pairs.

    Drops the module docstring, ``from __future__`` imports, relative imports,
    and every statement that belongs to one of the file's inlined blocks.
    Returning per-statement pairs lets the generator dedupe definitions that
    several reference files each carry a copy of.
    """
    src = _read(path)
    lines = _lines(src)
    pieces: list[tuple[frozenset[str], str]] = []

    # A generated reference carries its *own* definitions under a marker that
    # points at its baseline counterpart.  That marker must not be treated as
    # "belongs to a dependency", or regenerating a file would drop its target
    # class and the tool would not be idempotent.
    self_baseline: Path | None = None
    if path.is_relative_to(REFERENCE):
        cand = BASELINE / path.relative_to(REFERENCE)
        if cand.is_file():
            self_baseline = cand.resolve()

    for node, marker in _segment_statements(path):
        # module docstring
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str) and node.lineno <= 3):
            continue
        if isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            if (node.level or 0) > 0:
                # Relative import: the dependency is inlined instead.  An
                # aliased import still needs its alias, or every use of the
                # alias becomes a NameError -- flux_attention imports
                # ``T5LayerNorm as FP32RMSNorm`` and refers to it four times.
                aliases = [
                    (a.asname, a.name) for a in node.names
                    if a.asname and a.asname != a.name
                ]
                for asname, name in aliases:
                    pieces.append((
                        frozenset({asname}),
                        f"{asname} = {name}  # alias from inlined dependency",
                    ))
                continue
        # statement owned by an inlined block?
        if marker is not None:
            dep = _resolve_marker(marker)
            if (dep is not None and dep != path
                    and dep.resolve() != self_baseline):
                bound = _top_level_names(ast.Module(body=[node], type_ignores=[]))
                if bound and bound <= defined_names(dep):
                    continue
        # ``lineno`` points at the ``class``/``def`` keyword, *after* any
        # decorators.  Slicing from there silently drops ``@dataclass`` -- which
        # turned 90 inlined config/output classes into plain classes that reject
        # keyword construction ("CausalLMOutputWithPast() takes no arguments").
        start = node.lineno - 1
        # Walk back over the decorator lines immediately above the definition.
        while start > 0 and lines[start - 1].lstrip().startswith("@"):
            start -= 1
        end = node.end_lineno
        text = "\n".join(lines[start:end]).rstrip()
        if not text.strip():
            continue
        bound = frozenset(
            _top_level_names(ast.Module(body=[node], type_ignores=[]))
        )
        pieces.append((bound, text))

    return pieces


def own_body(path: Path) -> str:
    """``own_statements`` joined back into a source block."""
    return "\n\n\n".join(text for _, text in own_statements(path))


# --------------------------------------------------------------------------
# dependency graph
# --------------------------------------------------------------------------

def _relative_import_targets(path: Path) -> list[Path]:
    """Files pulled in by ``path``'s relative imports (mirrors kernel_swapper)."""
    src = _read(path)
    tree = ast.parse(src, filename=str(path))
    out: list[Path] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        level = node.level or 0
        if level == 0:
            continue
        base = path.parent
        for _ in range(level - 1):
            base = base.parent
        cand = base / (node.module.replace(".", "/") + ".py")
        if cand.is_file():
            out.append(cand)
            continue
        pkg = base / node.module.replace(".", "/")
        for alias in node.names:
            sub = pkg / (alias.name + ".py")
            if sub.is_file():
                out.append(sub)
    return out


def _import_module_for(path: Path) -> str | None:
    """Dotted module to import instead of inlining ``path``, if any."""
    try:
        rel = path.resolve().relative_to(KB_ROOT).as_posix()
    except ValueError:
        return None
    return IMPORT_NOT_INLINE.get(rel)


def _imported_names(target: Path) -> dict[str, set[str]]:
    """Names each import-not-inline module must supply for ``target``'s closure.

    Walks the baseline closure's relative imports; whatever the production code
    pulls out of the stateful module is what the reference needs too.
    """
    wanted: dict[str, set[str]] = {}
    seen: set[Path] = set()

    def walk(fp: Path) -> None:
        fp = fp.resolve()
        if fp in seen or not fp.is_file():
            return
        seen.add(fp)
        try:
            tree = ast.parse(_read(fp), filename=str(fp))
        except SyntaxError:
            return
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if (node.level or 0) == 0:
                continue
            base = fp.parent
            for _ in range(node.level - 1):
                base = base.parent
            cand = base / (node.module.replace(".", "/") + ".py")
            if not cand.is_file():
                pkg = base / node.module.replace(".", "/")
                for alias in node.names:
                    sub = pkg / (alias.name + ".py")
                    if sub.is_file():
                        walk(sub)
                continue
            mod = _import_module_for(cand)
            if mod is not None:
                wanted.setdefault(mod, set()).update(
                    a.name for a in node.names
                )
            else:
                walk(cand)

    walk(target)
    return wanted


def _is_generated(ref_path: Path) -> bool:
    """True if ``ref_path`` was produced by this tool.

    Generated files carry a marker pointing at their own baseline; hand-written
    ones never do.
    """
    if not ref_path.is_file() or not ref_path.is_relative_to(REFERENCE):
        return False
    self_baseline = BASELINE / ref_path.relative_to(REFERENCE)
    needle = MARKER + self_baseline.relative_to(KB_ROOT).as_posix()
    return needle in _read(ref_path)


def emit_source_for(baseline_path: Path) -> Path:
    """Reference counterpart of a baseline file, if one exists."""
    try:
        rel = baseline_path.relative_to(BASELINE)
    except ValueError:
        return baseline_path  # infra/* and friends: use as-is
    ref = REFERENCE / rel
    return ref if ref.is_file() else baseline_path


def build_plan(target_baseline: Path) -> list[Path]:
    """Topologically ordered emit sources for ``target_baseline``'s closure.

    The target itself is excluded; callers append its own body last.
    """
    order: list[Path] = []
    visiting: set[Path] = set()
    done: set[Path] = set()

    # The target's own body is appended by the caller.  Exclude both its
    # baseline and its (possibly pre-existing) reference from the plan so a
    # dependency that inlines the target cannot emit it a second time --
    # several yolov10 references cross-inline each other this way.
    excluded = {target_baseline.resolve()}
    self_reference = emit_source_for(target_baseline)
    if self_reference != target_baseline:
        excluded.add(self_reference.resolve())

    def prerequisites(emit: Path) -> list[Path]:
        """Emit sources ``emit`` needs before it."""
        reqs: list[Path] = []
        # reference files declare their deps via markers
        if emit.is_relative_to(REFERENCE):
            for m in _marker_sources(_read(emit)):
                r = _resolve_marker(m)
                if r is not None and r != emit:
                    reqs.append(r)
        # baseline (and infra) files declare theirs via relative imports
        for dep in _relative_import_targets(emit):
            reqs.append(emit_source_for(dep))
        return reqs

    def visit(emit: Path) -> None:
        emit = emit.resolve()
        if emit in excluded or emit in done:
            return
        if _import_module_for(emit) is not None:
            return  # imported instead of inlined
        if emit in visiting:
            return  # import cycle: first visit wins
        visiting.add(emit)
        for req in prerequisites(emit):
            visit(req)
        visiting.discard(emit)
        done.add(emit)
        order.append(emit)

    for dep in _relative_import_targets(target_baseline):
        visit(emit_source_for(dep))
    # a reference for the target itself may pre-exist with useful markers
    self_ref = emit_source_for(target_baseline)
    if self_ref != target_baseline and self_ref.is_file():
        for m in _marker_sources(_read(self_ref)):
            r = _resolve_marker(m)
            if r is not None:
                visit(r)
    return order


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

def _docstring_of(path: Path) -> str:
    tree = ast.parse(_read(path), filename=str(path))
    doc = ast.get_docstring(tree, clean=False)
    return doc or ""


def _same_definition(a: str, b: str) -> bool:
    """True if two snippets define the same top-level name(s)."""
    try:
        na = _top_level_names(ast.parse(a))
        nb = _top_level_names(ast.parse(b))
    except SyntaxError:
        return False
    return bool(na) and na == nb


def generate(level: int, name: str) -> tuple[str, list[str]]:
    """Return (source, warnings) for the reference of ``L{level}/{name}``."""
    target = BASELINE / f"L{level}" / f"{name}.py"
    if not target.is_file():
        raise FileNotFoundError(target)

    warnings: list[str] = []
    plan = build_plan(target)

    doc = _docstring_of(target)
    header = f'"""{doc}"""' if doc else ""
    parts: list[str] = []
    if header:
        parts.append(header)
    parts.append("from __future__ import annotations")

    # Stateful framework modules are imported so the reference shares the
    # harness's forward-context singleton instead of owning a private copy.
    for mod, names in sorted(_imported_names(target).items()):
        if not names:
            continue
        listed = ", ".join(sorted(names))
        parts.append(
            f"# Imported (not inlined): {mod} owns process-global state that\n"
            f"# the benchmark harness writes into.\n"
            f"from {mod} import {listed}"
        )

    # name -> (origin, source text) of the definition already emitted
    seen: dict[str, tuple[str, str]] = {}

    def emit_block(path: Path, is_target: bool = False) -> None:
        rel = path.relative_to(KB_ROOT).as_posix()
        kept: list[str] = []
        for bound, text in own_statements(path):
            # A dependency's ``__all__`` is a module export list: meaningless
            # once flattened, and every dep carrying one would collide with the
            # others.  The target's own ``__all__`` is kept.
            if bound == frozenset({"__all__"}) and not is_target:
                continue
            # Only module-level imports threaten self-containment.  Imports
            # nested in a function under ``try/except ImportError`` (e.g.
            # infra/context.py's Blackwell backend probe) are fine and are
            # present in the hand-written corpus too.
            for _stmt in ast.parse(text).body:
                mods = []
                if isinstance(_stmt, ast.Import):
                    mods = [a.name for a in _stmt.names]
                elif isinstance(_stmt, ast.ImportFrom) and _stmt.module:
                    mods = [_stmt.module]
                for _m in mods:
                    if _m.split(".")[0] in FORBIDDEN_ROOTS:
                        warnings.append(
                            f"FORBIDDEN module-level import '{_m}' from {rel} "
                            f"-- this dependency needs a pure-PyTorch reference"
                        )
            if bound:
                dup = sorted(n for n in bound if n in seen)
                if dup:
                    prev_origin, prev_text = seen[dup[0]]
                    if prev_text == text:
                        continue  # byte-identical copy: emit once
                    # Same definition reached through two paths, differing only
                    # in incidental formatting (e.g. one copy carrying its
                    # decorator).  Keep the *longer* text -- it is the complete
                    # one -- and never emit the name twice, which would shadow
                    # the decorated version with a bare one.
                    if _same_definition(prev_text, text):
                        if len(text) > len(prev_text):
                            for n in bound:
                                seen[n] = (rel, text)
                            kept.append(text)
                        continue
                    warnings.append(
                        f"name collision: {dup} defined differently in "
                        f"{prev_origin} and {rel}; keeping {rel}"
                    )
                for n in bound:
                    seen[n] = (rel, text)
            kept.append(text)
        if kept:
            parts.append(f"{MARKER}{rel}\n\n\n" + "\n\n\n".join(kept))

    for emit in plan:
        emit_block(emit)

    # Prefer a committed reference for the target's own body: if the target's
    # baseline uses an external kernel in its *own* logic, only a hand-written
    # pure-PyTorch body can replace it -- flattening cannot invent one.  When
    # no reference exists yet (the L3/L4 case) the baseline body is emitted and
    # the FORBIDDEN-import check below flags exactly those targets.
    target_body_source = emit_source_for(target)
    if target_body_source != target and _is_generated(target_body_source):
        # A previously *generated* reference is not authoritative: regenerating
        # from it would compound any staleness in what it inlined.  Only a
        # hand-written reference can override the baseline body.
        target_body_source = target
    emit_block(target_body_source, is_target=True)

    source = "\n\n\n".join(parts).rstrip() + "\n"

    # Self-containment: no module-level relative imports may survive.
    # (Forbidden kernel-library imports are reported per-origin in emit_block.)
    tree = ast.parse(source)
    if any(isinstance(n, ast.ImportFrom) and (n.level or 0) > 0
           for n in tree.body):
        warnings.append("relative import survives: file is not self-contained")

    # the target class must be resolvable as the final top-level class
    classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    if not classes:
        warnings.append("no top-level class in generated reference")
    else:
        baseline_classes = [
            n.name for n in ast.parse(_read(target)).body
            if isinstance(n, ast.ClassDef)
        ]
        if baseline_classes and classes[-1] != baseline_classes[-1]:
            warnings.append(
                f"target class mismatch: baseline ends with "
                f"'{baseline_classes[-1]}', generated ends with '{classes[-1]}'"
            )
    return source, warnings


def verify(level: int, name: str) -> list[str]:
    """Compare a regenerated reference against the committed one by symbols."""
    committed = REFERENCE / f"L{level}" / f"{name}.py"
    if not committed.is_file():
        return [f"no committed reference at {committed}"]
    gen_src, warns = generate(level, name)
    gen_names = _top_level_names(ast.parse(gen_src))
    ref_names = _top_level_names(ast.parse(_read(committed)))
    msgs = list(warns)
    missing = ref_names - gen_names
    extra = gen_names - ref_names
    if missing:
        msgs.append(f"missing vs committed: {sorted(missing)}")
    if extra:
        msgs.append(f"extra vs committed: {sorted(extra)}")
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--level", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--target", type=str, default=None)
    ap.add_argument("--all", action="store_true",
                    help="generate for every baseline op at this level")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated subset of target names")
    ap.add_argument("--check", action="store_true",
                    help="print to stdout instead of writing")
    ap.add_argument("--verify", action="store_true",
                    help="compare against the committed reference by symbols")
    args = ap.parse_args()

    lvl_dir = BASELINE / f"L{args.level}"
    if args.target:
        names = [args.target]
    elif args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
    elif args.all:
        names = sorted(
            f[:-3] for f in os.listdir(lvl_dir)
            if f.endswith(".py") and f != "__init__.py"
        )
    else:
        ap.error("one of --target / --only / --all is required")

    out_dir = REFERENCE / f"L{args.level}"
    out_dir.mkdir(parents=True, exist_ok=True)
    init = out_dir / "__init__.py"
    if not init.exists():
        init.write_text("")

    n_ok = n_warn = n_fail = 0
    for name in names:
        try:
            if args.verify:
                msgs = verify(args.level, name)
                status = "OK" if not msgs else "DIFF"
                print(f"[{status}] L{args.level}/{name}")
                for m in msgs:
                    print(f"        {m}")
                n_ok += 1 if not msgs else 0
                n_warn += 1 if msgs else 0
                continue
            src, warns = generate(args.level, name)
            if args.check:
                print(f"##### L{args.level}/{name} #####")
                print(src)
            else:
                (out_dir / f"{name}.py").write_text(src)
            if warns:
                n_warn += 1
                print(f"[WARN] L{args.level}/{name}")
                for w in warns:
                    print(f"        {w}")
            else:
                n_ok += 1
                print(f"[ OK ] L{args.level}/{name}")
        except Exception as exc:
            n_fail += 1
            print(f"[FAIL] L{args.level}/{name}: {type(exc).__name__}: {exc}")

    print(f"\nclean {n_ok}   with-warnings {n_warn}   failed {n_fail}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
