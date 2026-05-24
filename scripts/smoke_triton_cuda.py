#!/usr/bin/env python3
"""CUDA smoke tests and microbenchmarks for the Triton scan kernels."""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.continuous_spatial_mamba import cs_mamba_forward_reference
from models.continuous_spatial_mamba import ContinuousSpatialSSM
from models.vmamba_4d import CrossScanSS2D
from triton_kernels.csma_autograd import cs_scan
from triton_kernels.csma_triton_scan import cs_scan_forward_cuda, grid_neighbor_index
from triton_kernels.vmamba4d_triton_scan import selective_scan_cuda


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> float:
    max_abs = (actual - expected).abs().max().item()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    print(f"{name}: max_abs={max_abs:.3e}")
    return max_abs


def cuda_time_ms(fn, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def make_csma_inputs(batch: int, d_model: int, d_state: int, h_grid: int, w_grid: int, k_steps: int):
    del k_steps
    device = "cuda"
    n_tokens = h_grid * w_grid
    torch.manual_seed(123)
    h0 = torch.randn(batch, n_tokens, d_model, d_state, device=device, dtype=torch.float32) * 0.1
    x = torch.randn(batch, n_tokens, d_model, device=device, dtype=torch.float32) * 0.1
    delta_s = torch.rand(batch, n_tokens, d_model, device=device, dtype=torch.float32) * 0.15
    delta_d = torch.rand(batch, n_tokens, d_model, device=device, dtype=torch.float32) * 0.15
    A = -torch.rand(d_model, d_state, device=device, dtype=torch.float32)
    B_mat = torch.randn(batch, n_tokens, d_state, device=device, dtype=torch.float32) * 0.1
    D_phys = torch.rand(1, d_model, 1, 1, device=device, dtype=torch.float32) * 0.5
    return h0, x, delta_s, delta_d, A, B_mat, D_phys


def csma_correctness(args) -> None:
    print("\nCS-Mamba V1 Triton scan correctness")
    h0, x, ds, dd, A, bm, dp = make_csma_inputs(
        args.batch, args.d_model, args.d_state, args.h_grid, args.w_grid, args.k_steps
    )
    grad_seed = torch.randn_like(h0)

    ref_inputs = [t.detach().clone().requires_grad_(True) for t in (h0, x, ds, dd, A, bm, dp)]
    h_ref, _ = cs_mamba_forward_reference(*ref_inputs, args.k_steps, args.h_grid, args.w_grid)
    (h_ref * grad_seed).sum().backward()
    ref_grads = [t.grad.detach().clone() if t.grad is not None else None for t in ref_inputs]

    tri_inputs = [t.detach().clone().requires_grad_(True) for t in (h0, x, ds, dd, A, bm, dp)]
    h_tri = cs_scan(*tri_inputs, args.k_steps, args.h_grid, args.w_grid)
    (h_tri * grad_seed).sum().backward()
    tri_grads = [t.grad.detach().clone() if t.grad is not None else None for t in tri_inputs]

    assert_close("forward", h_tri, h_ref, atol=args.atol, rtol=args.rtol)
    for name, tri_grad, ref_grad in [
        ("grad_h0", tri_grads[0], ref_grads[0]),
        ("grad_delta_s", tri_grads[2], ref_grads[2]),
        ("grad_delta_d", tri_grads[3], ref_grads[3]),
        ("grad_A", tri_grads[4], ref_grads[4]),
        ("grad_D_phys", tri_grads[6], ref_grads[6]),
    ]:
        assert_close(name, tri_grad, ref_grad, atol=args.grad_atol, rtol=args.grad_rtol)

    neighbors = grid_neighbor_index(args.h_grid, args.w_grid, h0.device)
    assert tuple(neighbors.shape) == (args.h_grid * args.w_grid, 4)
    assert int(neighbors[0, 0].item()) == 0
    assert int(neighbors[0, 2].item()) == 0
    assert int(neighbors[-1, 1].item()) == args.h_grid * args.w_grid - 1
    assert int(neighbors[-1, 3].item()) == args.h_grid * args.w_grid - 1
    h_tri_custom, _ = cs_scan_forward_cuda(
        h0.detach(),
        ds.detach(),
        dd.detach(),
        A.detach(),
        dp.detach(),
        args.k_steps,
        args.h_grid,
        args.w_grid,
        neighbor_index=neighbors,
    )
    assert_close("forward_custom_neighbor_index", h_tri_custom, h_ref.detach(), atol=args.atol, rtol=args.rtol)


def vmamba_reference_scan(retain: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    states = []
    h = torch.zeros_like(update[:, 0])
    for t in range(update.shape[1]):
        h = retain[:, t] * h + update[:, t]
        states.append(h)
    return torch.stack(states, dim=1)


def vmamba_correctness(args) -> None:
    print("\nVMamba4D Triton selective-scan correctness")
    torch.manual_seed(456)
    retain = (0.20 + 0.75 * torch.rand(args.scan_batch, args.scan_len, args.scan_dim, device="cuda")).requires_grad_(True)
    update = (torch.randn(args.scan_batch, args.scan_len, args.scan_dim, device="cuda") * 0.1).requires_grad_(True)
    grad_seed = torch.randn_like(update)

    retain_ref = retain.detach().clone().requires_grad_(True)
    update_ref = update.detach().clone().requires_grad_(True)
    y_ref = vmamba_reference_scan(retain_ref, update_ref)
    (y_ref * grad_seed).sum().backward()

    retain_tri = retain.detach().clone().requires_grad_(True)
    update_tri = update.detach().clone().requires_grad_(True)
    y_tri = selective_scan_cuda(retain_tri, update_tri)
    (y_tri * grad_seed).sum().backward()

    assert_close("forward", y_tri, y_ref, atol=args.atol, rtol=args.rtol)
    assert_close("grad_retain", retain_tri.grad, retain_ref.grad, atol=args.grad_atol, rtol=args.grad_rtol)
    assert_close("grad_update", update_tri.grad, update_ref.grad, atol=args.grad_atol, rtol=args.grad_rtol)


def module_smoke() -> None:
    print("\nModule integration smoke test")
    torch.manual_seed(999)

    csma = ContinuousSpatialSSM(d_model=32, d_state=8).cuda().train()
    x = torch.randn(2, 16, 64, device="cuda")
    with torch.no_grad():
        y_ref = csma(x, K_steps=3, use_triton=False)
        y_tri = csma(x, K_steps=3, use_triton=True)
    assert_close("ContinuousSpatialSSM forward", y_tri, y_ref, atol=3e-4, rtol=3e-4)
    opt = torch.optim.AdamW(csma.parameters(), lr=1e-4)
    opt.zero_grad(set_to_none=True)
    csma(x, K_steps=3, use_triton=True).square().mean().backward()
    opt.step()
    print("ContinuousSpatialSSM optimizer step: ok")

    ss2d = CrossScanSS2D(d_model=32).cuda().train()
    seq = torch.randn(2, 16, 64, device="cuda")
    with torch.no_grad():
        ss2d.use_triton = False
        y_ref = ss2d(seq)
        ss2d.use_triton = True
        y_tri = ss2d(seq)
    assert_close("CrossScanSS2D forward", y_tri, y_ref, atol=3e-4, rtol=3e-4)
    opt = torch.optim.AdamW(ss2d.parameters(), lr=1e-4)
    opt.zero_grad(set_to_none=True)
    ss2d(seq).square().mean().backward()
    opt.step()
    print("CrossScanSS2D optimizer step: ok")


def benchmark_csma(args) -> None:
    print("\nCS-Mamba V1 scan benchmark")
    h0, x, ds, dd, A, bm, dp = make_csma_inputs(
        args.bench_batch, args.bench_d_model, args.d_state, args.h_grid, args.w_grid, args.k_steps
    )

    def ref_fwd():
        with torch.no_grad():
            cs_mamba_forward_reference(h0, x, ds, dd, A, bm, dp, args.k_steps, args.h_grid, args.w_grid)

    def tri_fwd():
        with torch.no_grad():
            cs_scan(h0, x, ds, dd, A, bm, dp, args.k_steps, args.h_grid, args.w_grid)

    ref_ms = cuda_time_ms(ref_fwd, iters=args.iters)
    tri_ms = cuda_time_ms(tri_fwd, iters=args.iters)
    print(f"forward PyTorch: {ref_ms:.3f} ms")
    print(f"forward Triton : {tri_ms:.3f} ms ({ref_ms / tri_ms:.2f}x)")

    def ref_bwd():
        h0_r, x_r, ds_r, dd_r, A_r, bm_r, dp_r = [
            t.detach().clone().requires_grad_(True) for t in (h0, x, ds, dd, A, bm, dp)
        ]
        out, _ = cs_mamba_forward_reference(h0_r, x_r, ds_r, dd_r, A_r, bm_r, dp_r, args.k_steps, args.h_grid, args.w_grid)
        out.square().mean().backward()

    def tri_bwd():
        h0_r, x_r, ds_r, dd_r, A_r, bm_r, dp_r = [
            t.detach().clone().requires_grad_(True) for t in (h0, x, ds, dd, A, bm, dp)
        ]
        out = cs_scan(h0_r, x_r, ds_r, dd_r, A_r, bm_r, dp_r, args.k_steps, args.h_grid, args.w_grid)
        out.square().mean().backward()

    ref_bwd_ms = cuda_time_ms(ref_bwd, warmup=max(2, args.warmup // 2), iters=max(5, args.iters // 5))
    tri_bwd_ms = cuda_time_ms(tri_bwd, warmup=max(2, args.warmup // 2), iters=max(5, args.iters // 5))
    print(f"fwd+bwd PyTorch: {ref_bwd_ms:.3f} ms")
    print(f"fwd+bwd Triton : {tri_bwd_ms:.3f} ms ({ref_bwd_ms / tri_bwd_ms:.2f}x)")


def benchmark_vmamba(args) -> None:
    print("\nVMamba4D selective-scan benchmark")
    torch.manual_seed(789)
    retain = 0.20 + 0.75 * torch.rand(args.scan_batch, args.scan_len, args.scan_dim, device="cuda")
    update = torch.randn(args.scan_batch, args.scan_len, args.scan_dim, device="cuda") * 0.1

    def ref_fwd():
        with torch.no_grad():
            vmamba_reference_scan(retain, update)

    def tri_fwd():
        with torch.no_grad():
            selective_scan_cuda(retain, update)

    ref_ms = cuda_time_ms(ref_fwd, iters=args.iters)
    tri_ms = cuda_time_ms(tri_fwd, iters=args.iters)
    print(f"forward PyTorch: {ref_ms:.3f} ms")
    print(f"forward Triton : {tri_ms:.3f} ms ({ref_ms / tri_ms:.2f}x)")

    def ref_bwd():
        retain_r = retain.detach().clone().requires_grad_(True)
        update_r = update.detach().clone().requires_grad_(True)
        vmamba_reference_scan(retain_r, update_r).square().mean().backward()

    def tri_bwd():
        retain_r = retain.detach().clone().requires_grad_(True)
        update_r = update.detach().clone().requires_grad_(True)
        selective_scan_cuda(retain_r, update_r).square().mean().backward()

    ref_bwd_ms = cuda_time_ms(ref_bwd, warmup=max(2, args.warmup // 2), iters=max(5, args.iters // 5))
    tri_bwd_ms = cuda_time_ms(tri_bwd, warmup=max(2, args.warmup // 2), iters=max(5, args.iters // 5))
    print(f"fwd+bwd PyTorch: {ref_bwd_ms:.3f} ms")
    print(f"fwd+bwd Triton : {tri_bwd_ms:.3f} ms ({ref_bwd_ms / tri_bwd_ms:.2f}x)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--h_grid", type=int, default=14)
    parser.add_argument("--w_grid", type=int, default=14)
    parser.add_argument("--k_steps", type=int, default=4)
    parser.add_argument("--bench_batch", type=int, default=4)
    parser.add_argument("--bench_d_model", type=int, default=256)
    parser.add_argument("--scan_batch", type=int, default=128)
    parser.add_argument("--scan_len", type=int, default=14)
    parser.add_argument("--scan_dim", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--grad_atol", type=float, default=8e-4)
    parser.add_argument("--grad_rtol", type=float, default=8e-4)
    parser.add_argument("--correctness_only", action="store_true")
    parser.add_argument("--skip_module_smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    print(f"torch {torch.__version__}")
    print(f"cuda {torch.cuda.get_device_name(0)}")
    print(f"benchmark started {time.strftime('%Y-%m-%d %H:%M:%S')}")

    csma_correctness(args)
    vmamba_correctness(args)
    if not args.skip_module_smoke:
        module_smoke()
    if not args.correctness_only:
        benchmark_csma(args)
        benchmark_vmamba(args)


if __name__ == "__main__":
    main()
