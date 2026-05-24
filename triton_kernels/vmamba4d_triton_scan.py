"""Triton recurrent scan primitive for the VMamba4D CUDA baseline."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU/TPU import guard
    triton = None
    tl = None

if triton is None:  # pragma: no cover - CUDA-only module
    raise ImportError("triton_kernels.vmamba4d_triton_scan requires Triton.")


def _require_cuda(retain: torch.Tensor, update: torch.Tensor) -> None:
    if not retain.is_cuda or not update.is_cuda:
        raise RuntimeError("VMamba4D Triton scan requires CUDA tensors.")


@triton.jit
def _selective_scan_fwd_kernel(
    RETAIN_PTR,
    UPDATE_PTR,
    OUT_PTR,
    B: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    d_block = tl.program_id(1)
    d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d < D

    h = tl.zeros((BLOCK_D,), dtype=tl.float32)
    base_b = b * L * D
    for t in range(L):
        offs = base_b + t * D + d
        retain = tl.load(RETAIN_PTR + offs, mask=mask, other=0.0).to(tl.float32)
        update = tl.load(UPDATE_PTR + offs, mask=mask, other=0.0).to(tl.float32)
        h = retain * h + update
        tl.store(OUT_PTR + offs, h, mask=mask)


@triton.jit
def _selective_scan_bwd_kernel(
    GRAD_OUT_PTR,
    RETAIN_PTR,
    OUT_PTR,
    GRAD_RETAIN_PTR,
    GRAD_UPDATE_PTR,
    B: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    d_block = tl.program_id(1)
    d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d < D

    g_next = tl.zeros((BLOCK_D,), dtype=tl.float32)
    base_b = b * L * D
    for t_rev in range(L):
        t = L - 1 - t_rev
        offs = base_b + t * D + d
        grad_y = tl.load(GRAD_OUT_PTR + offs, mask=mask, other=0.0).to(tl.float32)
        retain = tl.load(RETAIN_PTR + offs, mask=mask, other=0.0).to(tl.float32)
        g_h = grad_y + g_next

        h_prev = tl.zeros((BLOCK_D,), dtype=tl.float32)
        if t > 0:
            h_prev = tl.load(OUT_PTR + base_b + (t - 1) * D + d, mask=mask, other=0.0).to(tl.float32)

        tl.store(GRAD_RETAIN_PTR + offs, g_h * h_prev, mask=mask)
        tl.store(GRAD_UPDATE_PTR + offs, g_h, mask=mask)
        g_next = g_h * retain


class VMambaSelectiveScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, retain: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        _require_cuda(retain, update)
        retain = retain.contiguous()
        update = update.contiguous()
        bsz, length, d_dim = retain.shape
        if update.shape != retain.shape:
            raise ValueError(f"update shape {tuple(update.shape)} must match retain {tuple(retain.shape)}")

        out = torch.empty_like(update)
        block_d = 128
        grid = (bsz, triton.cdiv(d_dim, block_d))
        _selective_scan_fwd_kernel[grid](
            retain,
            update,
            out,
            B=bsz,
            L=length,
            D=d_dim,
            BLOCK_D=block_d,
            num_warps=4,
        )
        ctx.save_for_backward(retain, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        retain, out = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_retain = torch.empty_like(retain)
        grad_update = torch.empty_like(out)
        bsz, length, d_dim = retain.shape
        block_d = 128
        grid = (bsz, triton.cdiv(d_dim, block_d))
        _selective_scan_bwd_kernel[grid](
            grad_output,
            retain,
            out,
            grad_retain,
            grad_update,
            B=bsz,
            L=length,
            D=d_dim,
            BLOCK_D=block_d,
            num_warps=4,
        )
        return grad_retain, grad_update


def selective_scan_cuda(retain: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    return VMambaSelectiveScanFunction.apply(retain, update)
