import torch
import torch._dynamo.testing
from torch.testing._internal import common_utils
from torch._higher_order_ops.triton_kernel_wrap import AccessPatternAnalysis, identify_access_patterns
import torch._inductor.utils
import torch._inductor.test_case
from torch.library import triton_op, wrap_triton

from torch.testing._internal.triton_utils import *  # noqa: F403
import triton
import triton.language as tl

import torch.nn as nn

def compare_read_write_patterns(fn):
    @requires_gpu
    def test_fn(self):
        from torch._higher_order_ops.triton_kernel_wrap import analyze_access_patterns
        import sympy

        kernel, inputs, grid, tma_descriptor_metadata, reads, writes = fn()
        access_pattern = identify_access_patterns(kernel, inputs, tma_descriptor_metadata, grid)

        def canonicalize(expr):
            if expr is None:
                return None
            return str(sympy.simplify(sympy.sympify(str(expr))))

        self.assertEqual(len(access_pattern.reads), len(reads))
        for actual, expected in zip(access_pattern.reads, reads):
            actual_ptr = canonicalize(actual.ptr_expr)
            expected_ptr = canonicalize(expected.ptr_expr)
            self.assertEqual(actual_ptr, expected_ptr)

            actual_mask = canonicalize(actual.mask_expr)
            expected_mask = canonicalize(expected.mask_expr)
            self.assertEqual(actual_mask, expected_mask)

        self.assertEqual(len(access_pattern.writes), len(writes))
        for actual, expected in zip(access_pattern.writes, writes):
            actual_ptr = canonicalize(actual.ptr_expr)
            expected_ptr = canonicalize(expected.ptr_expr)
            self.assertEqual(actual_ptr, expected_ptr)

            actual_mask = canonicalize(actual.mask_expr)
            expected_mask = canonicalize(expected.mask_expr)
            self.assertEqual(actual_mask, expected_mask)

    return test_fn


class MutationTests(torch._inductor.test_case.TestCase):

    @compare_read_write_patterns
    def test_custom_gelu():
        import sympy
        from torch._higher_order_ops.triton_kernel_wrap import (
            ReadPattern,
            WritePattern,
        )

        @triton.jit
        def gelu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)

            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)

            sigmoid_input = 1.702 * x
            sigmoid_val = 1.0 / (1.0 + tl.exp(-sigmoid_input))
            gelu_result = x * sigmoid_val

            tl.store(output_ptr + offsets, gelu_result, mask=mask)

        t = torch.randn(64, 1024)
        BLOCK_SIZE = 1024
        GRID = (triton.cdiv(t.numel(), BLOCK_SIZE),)

        return (
            gelu_kernel,
            {
                "x_ptr": t,
                "output_ptr": t,
                "n_elements": t.numel(),
                "BLOCK_SIZE": BLOCK_SIZE
            },
            GRID,
            {},
            [ReadPattern(ptr_expr=sympy.Symbol("p0") + 1024*sympy.Symbol("d0") + sympy.Symbol("d15"),
                         mask_expr=1024*sympy.Symbol("d0") + sympy.Symbol("d15") < 65536)
            ],
            [WritePattern(ptr_expr=sympy.Symbol("p1") + 1024 * sympy.Symbol("d0") + sympy.Symbol("d15"),
                          mask_expr=1024*sympy.Symbol("d0") + sympy.Symbol("d15") < 65536,
                          loc=13,
                          operand_name='output_ptr')
            ])



@triton_op("mine::gelu", mutates_args={}, attempt_fusion=True)
def custom_gelu_triton(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    GRID = (triton.cdiv(n_elements, BLOCK_SIZE),)
    wrap_triton(gelu_kernel)[GRID](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return output

@triton.jit
def gelu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    sigmoid_input = 1.702 * x
    sigmoid_val = 1.0 / (1.0 + tl.exp(-sigmoid_input))
    gelu_result = x * sigmoid_val

    tl.store(output_ptr + offsets, gelu_result, mask=mask)

class FusionBlockedMLP(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.linear1 = nn.Linear(dim, 4*dim)
        self.linear2 = nn.Linear(4*dim, dim)

    def forward(self, x):
        x = self.linear1(x)
        x = custom_gelu_triton(x)
        x = x * 0.5
        x = self.linear2(x)
        return x

class PointwiseChain(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * 0.5
        x = x + 0.1
        x = custom_gelu_triton(x)
        x = x * 2.0
        x = x - 0.5
        x = custom_gelu_triton(x)
        x = x + 0.3
        return x


class AccuracyTests(torch._inductor.test_case.TestCase):

    def test_fusion_blocked_mlp(self):
        device = torch.device("cuda")
        model = FusionBlockedMLP().to(device).eval()
        x = torch.randn(64, 1024, device=device)

        compiled_out = None
        model_out = model(x)
        with torch.no_grad():
            with torch._inductor.utils.fresh_inductor_cache():
                torch.compiler.reset()
                compiled = torch.compile(model)
                compiled_out = compiled(x)

        self.assertEqual(model_out, compiled_out)

    def test_pointwise_chain(self):
        device = torch.device("cuda")
        model = PointwiseChain().to(device).eval()
        x = torch.randn(64, 1024, device=device)

        compiled_out = None
        model_out = model(x)
        with torch.no_grad():
            with torch._inductor.utils.fresh_inductor_cache():
                torch.compiler.reset()
                compiled = torch.compile(model)
                compiled_out = compiled(x)

        self.assertEqual(model_out, compiled_out)

# common_utils.instantiate_parametrized_tests(MutationTests)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
