"""Global flag controlling whether modules dispatch to native PyTorch
implementations (for torch.compile tracing) or to custom Triton/CUDA kernels.

Set via `set_compile_mode(True)` in the engine before torch.compile.
Modules check `is_compile_mode()` in their forward paths.
"""

_COMPILE_MODE = False


def set_compile_mode(enabled: bool) -> None:
    global _COMPILE_MODE
    _COMPILE_MODE = enabled


def is_compile_mode() -> bool:
    return _COMPILE_MODE
