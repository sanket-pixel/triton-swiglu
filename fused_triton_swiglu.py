import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_swiglu_kernel(
        x_ptr, w_proj_ptr, v_proj_ptr, output_ptr,
        B, M, N, K,
        stride_xb, stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_vk, stride_vn,
        stride_ob, stride_om, stride_on,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)  # batch

    # grouped scheduling
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + (pid_b * stride_xb + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_proj_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    v_ptrs = v_proj_ptr + (offs_k[:, None] * stride_vk + offs_n[None, :] * stride_vn)

    # accumulators
    acc_w = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_v = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        mask_k = offs_k < k_remaining

        x_block = tl.load(x_ptrs, mask=mask_k[None, :], other=0.0)
        w_block = tl.load(w_ptrs, mask=mask_k[:, None], other=0.0)
        v_block = tl.load(v_ptrs, mask=mask_k[:, None], other=0.0)

        acc_w += tl.dot(x_block, w_block)
        acc_v += tl.dot(x_block, v_block)

        # advance pointers
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk
        v_ptrs += BLOCK_SIZE_K * stride_vk

    silu_acc_w = acc_w * tl.sigmoid(acc_w)
    output_block = silu_acc_w * acc_v

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = output_ptr + (pid_b * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)

    mask_output = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, output_block, mask=mask_output)


def fused_swiglu(x: torch.Tensor, w: torch.Tensor, v: torch.Tensor):
    # handle 2d case
    if x.dim() == 2:
        x = x.unsqueeze(0)

    B, M, K = x.shape
    _, N = w.shape

    output = torch.empty((B, M, N), device=x.device, dtype=x.dtype)

    # ensure contiguous inputs
    x = x.contiguous()
    w = w.contiguous()
    v = v.contiguous()

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        B
    )

    fused_swiglu_kernel[grid](
        x, w, v, output,
        B, M, N, K,
        x.stride(0), x.stride(1), x.stride(2),
        w.stride(0), w.stride(1),
        v.stride(0), v.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
    )

    return output.squeeze(0) if B == 1 else output