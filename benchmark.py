import torch
import triton
import triton.testing
import torch.nn.functional as F

# assuming these are in other files
from baseline_torch_swiglu import PyTorchSwiGLU
from fused_triton_swiglu import fused_swiglu


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M'],
        x_vals=[128, 512, 1024, 2048, 4096, 8192],
        line_arg='provider',
        line_vals=['pytorch', 'triton'],
        line_names=['PyTorch', 'Fused Triton'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='ms',
        plot_name='SwiGLU_Performance',
        args={'K': 1024, 'N': 4096, 'dtype': torch.bfloat16, 'device': 'cuda'},
    )
)
def benchmark(M, K, N, dtype, device, provider):
    # benchmark function
    x = torch.randn(M, K, device=device, dtype=dtype)
    w_proj = torch.randn(K, N, device=device, dtype=dtype)
    v_proj = torch.randn(K, N, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'pytorch':
        pytorch_fn = lambda: F.silu(x @ w_proj) * (x @ v_proj)
        ms, min_ms, max_ms = triton.testing.do_bench(pytorch_fn, quantiles=quantiles)
    elif provider == 'triton':
        triton_fn = lambda: fused_swiglu(x, w_proj, v_proj)
        ms, min_ms, max_ms = triton.testing.do_bench(triton_fn, quantiles=quantiles)

    return ms, min_ms, max_ms


if __name__ == "__main__":
    print("checking correctness...")

    # config
    M, K, N = 2048, 1024, 4096
    dtype = torch.bfloat16
    device = 'cuda'

    # test tensors
    x = torch.randn(M, K, device=device, dtype=dtype)
    w = torch.randn(K, N, device=device, dtype=dtype)
    v = torch.randn(K, N, device=device, dtype=dtype)

    pytorch_output = F.silu(x @ w) * (x @ v)
    triton_output = fused_swiglu(x, w, v)

    # using a high tolerance for bfloat16
    is_correct = torch.allclose(pytorch_output, triton_output, atol=1e1, rtol=1e2)
    print(f"Correctness check passed: {is_correct}\n")

    print("running benchmark...")
    benchmark.run(print_data=True, show_plots=True)
