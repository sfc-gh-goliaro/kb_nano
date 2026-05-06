"""Build kb_nano_operator_catalog.csv and hf_model_inventory.csv.

Run from the repo root:
    python audits/hf_transformers_coverage/tools/build_inventories.py

Inputs:
- kb-nano L1/L2/L3 source files at tasks/baseline/L{1,2,3}/*.py (current branch).
- HF transformers source at /tmp/hf_transformers_pinned/src/transformers/models.

Outputs (under audits/hf_transformers_coverage/):
- kb_nano_operator_catalog.csv
- hf_model_inventory.csv
"""

from __future__ import annotations

import ast
import csv
import sys
from pathlib import Path

REPO = Path("/home/olu/kb_nano")
HF = Path("/tmp/hf_transformers_pinned/src/transformers/models")
OUT = REPO / "audits/hf_transformers_coverage"


def module_doc_first_line(tree: ast.Module) -> str:
    doc = ast.get_docstring(tree) or ""
    return doc.strip().split("\n", 1)[0].strip()


def extract_kb_nano_classes(path: Path) -> list[tuple[str, str, list[str]]]:
    """Return list of (class_name, parent_string, docstring_first_line) for top-level classes."""
    src = path.read_text()
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:
        return [("<SYNTAX_ERROR>", str(e), "")]
    out = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            parents = [ast.unparse(b) for b in node.bases]
            doc = ast.get_docstring(node) or ""
            doc_first = doc.strip().split("\n", 1)[0].strip()
            out.append((node.name, ";".join(parents), doc_first))
    return out


def build_kb_nano_catalog():
    rows = []
    for layer in ("L1", "L2", "L3"):
        d = REPO / "tasks/baseline" / layer
        for p in sorted(d.glob("*.py")):
            if p.name == "__init__.py":
                continue
            rel = p.relative_to(REPO)
            tree_doc = ""
            try:
                tree_doc = module_doc_first_line(ast.parse(p.read_text()))
            except Exception:
                pass
            classes = extract_kb_nano_classes(p)
            if not classes:
                rows.append({
                    "layer": layer,
                    "file": str(rel),
                    "class_name": "",
                    "parents": "",
                    "module_doc_first_line": tree_doc,
                    "class_doc_first_line": "",
                })
            else:
                for cname, parents, cdoc in classes:
                    rows.append({
                        "layer": layer,
                        "file": str(rel),
                        "class_name": cname,
                        "parents": parents,
                        "module_doc_first_line": tree_doc,
                        "class_doc_first_line": cdoc,
                    })
    out_path = OUT / "kb_nano_operator_catalog.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"kb_nano_operator_catalog.csv: {len(rows)} rows -> {out_path}")
    by_layer = {}
    for r in rows:
        by_layer.setdefault(r["layer"], 0)
        by_layer[r["layer"]] += 1
    print(f"  by layer: {by_layer}")


def build_hf_inventory():
    rows = []
    for d in sorted(HF.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if name.startswith("__"):
            continue
        modeling_files = sorted(p.name for p in d.glob("modeling_*.py"))
        modular_files = sorted(p.name for p in d.glob("modular_*.py"))
        configuration_files = sorted(p.name for p in d.glob("configuration_*.py"))
        tokenization_files = sorted(p.name for p in d.glob("tokenization_*.py"))
        processing_files = sorted(p.name for p in d.glob("processing_*.py"))
        image_processing_files = sorted(p.name for p in d.glob("image_processing_*.py"))
        feature_extraction_files = sorted(p.name for p in d.glob("feature_extraction_*.py"))
        # PyTorch-only modeling (exclude TF/Flax)
        pytorch_modeling = [
            f for f in modeling_files
            if not f.startswith("modeling_tf_") and not f.startswith("modeling_flax_")
        ]
        has_pt = bool(pytorch_modeling)
        has_modular = bool(modular_files)
        has_exact_pt_modeling = f"modeling_{name}.py" in pytorch_modeling
        rows.append({
            "folder": name,
            "n_pytorch_modeling": len(pytorch_modeling),
            "pytorch_modeling_files": ";".join(pytorch_modeling),
            "n_modular": len(modular_files),
            "modular_files": ";".join(modular_files),
            "n_tokenization": len(tokenization_files),
            "n_processing": len(processing_files),
            "n_image_processing": len(image_processing_files),
            "n_feature_extraction": len(feature_extraction_files),
            "has_pt_modeling": has_pt,
            "has_modular": has_modular,
            "has_exact_pt_modeling": has_exact_pt_modeling,
            "is_modular_only": has_modular and not has_pt,
            "is_no_modeling": (not has_pt) and (not has_modular),
            "all_modeling_variants": ";".join(modeling_files),  # incl tf/flax
        })
    out_path = OUT / "hf_model_inventory.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_total = len(rows)
    n_pt = sum(1 for r in rows if r["has_pt_modeling"])
    n_exact = sum(1 for r in rows if r["has_exact_pt_modeling"])
    n_multi = sum(1 for r in rows if r["n_pytorch_modeling"] > 1)
    n_modular_only = sum(1 for r in rows if r["is_modular_only"])
    n_no_modeling = sum(1 for r in rows if r["is_no_modeling"])
    n_modular_total = sum(1 for r in rows if r["has_modular"])
    print(f"hf_model_inventory.csv: {n_total} rows -> {out_path}")
    print(f"  total folders                            : {n_total}")
    print(f"  has any PyTorch modeling_*.py            : {n_pt}")
    print(f"  has exact modeling_<folder>.py           : {n_exact}")
    print(f"  has multiple PyTorch modeling files      : {n_multi}")
    print(f"  has any modular_*.py                     : {n_modular_total}")
    print(f"  modular-only (no PyTorch modeling)       : {n_modular_only}")
    print(f"  no modeling at all (no PT, no modular)   : {n_no_modeling}")
    # Also count distinct PyTorch modeling files (preferred denominator)
    n_distinct_pt_files = sum(r["n_pytorch_modeling"] for r in rows)
    print(f"  distinct PyTorch modeling files (sum)    : {n_distinct_pt_files}")


if __name__ == "__main__":
    build_kb_nano_catalog()
    build_hf_inventory()
