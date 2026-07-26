"""Expert baseline for kb_rms_norm — the kb production baseline itself.

This file is INFRA, not an agent solution: it is packed into `expert_baseline.json`
at the child's root, where bench_utils profiles it to produce the score's
denominator. The delegation ban that applies to `solution/kernel.py` does NOT apply
here — the whole point of this file is to measure the incumbent kb kernel.

It imports through the `fastkernels` package name rather than a bare
`tasks.baseline...` path on purpose: the entrypoint binds `fastkernels` in
sys.modules before importing the candidate, so this resolves to the very module
object the entrypoint already loaded as the baseline. The expert and the
correctness oracle are then the same class — no second copy of the module, no
second JIT load of the CUDA extension.
"""

from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm as RMSNorm  # noqa: F401
