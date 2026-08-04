        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            start = pid * BLOCK
            offs = start + tl.arange(0, BLOCK)
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            y = tl.load(y_ptr + offs, mask=mask, other=0.0)
            z = x + y
            tl.store(out_ptr + offs, z, mask=mask)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                if x.is_cuda and y.is_cuda:
                    assert x.dtype == y.dtype, "Dtypes must match"
                    assert x.device == y.device, "Devices must match"
                    out = torch.empty_like(x)
                    n = out.numel()
                    BLOCK = 1024
                    grid = (triton.cdiv(n, BLOCK),)
                    add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
                    return out
                else:
                    return x + y
        