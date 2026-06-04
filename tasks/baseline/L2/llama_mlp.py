"""Llama SwiGLU MLP block: gate_up_proj -> SiluAndMul -> down_proj.

Also used by DeepSeek V3's "shared expert": pass ``reduce_results=False``
and override ``intermediate_size`` with ``moe_intermediate_size *
n_shared_experts``.  See ``L2/deepseek_moe.py``.
"""

from __future__ import annotations

import torch.nn as nn

from .parallel_linear import MergedColumnParallelLinear, RowParallelLinear
from ..L1.silu_and_mul import SiluAndMul, SiluAndMulWithClamp


class LlamaMLP(nn.Module):
    def __init__(self, config, quant_config: dict | None = None,
                 hidden_size: int | None = None,
                 intermediate_size: int | None = None,
                 reduce_results: bool = True,
                 swiglu_limit: float | None = None):
        super().__init__()
        h = hidden_size if hidden_size is not None else config.hidden_size
        i = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            h, [i] * 2,
            quant_config=quant_config,
        )
        self.down_proj = RowParallelLinear(
            i, h,
            quant_config=quant_config,
            reduce_results=reduce_results,
        )
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x):
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        return self.down_proj(x)
