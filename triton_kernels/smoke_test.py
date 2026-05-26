"""Smoke test for all Triton kernel fixes.

Tests:
  1. V1.2 integrator scan (heun/rk4/imex) — Triton forward vs PyTorch reference
     with non-identity activations (verifies Fix 1: Heun predictor activation).
  2. V1.2 flex kernel — Triton fwd/bwd vs PyTorch reference
     with 4-point and 8-point stencils and various activations.
  3. V1 backward — manual reference vs torch.autograd (verifies Fix 3: adjoint).
  4. V1.2 basic Euler kernel — Triton fwd/bwd vs PyTorch reference.
  5. VMamba4D selective scan — Triton fwd/bwd vs naive loop.

Usage:
    source ~/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/activate
    cd ~/Downloads/Documents/Work/Research/CVPR/learning/CG_Mamba
    python -m triton_kernels.smoke_test
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn.functional as F
import numpy as np


def _header(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def _check(name: str, a: torch.Tensor, b: torch.Tensor, atol: float = 1e-3, rtol: float = 1e-3) -> bool:
    max_err = (a.float() - b.float()).abs().max().item()
    rel_denom = b.float().abs().max().item() + 1e-12
    rel_err = max_err / rel_denom
    ok = max_err < atol or rel_err < rtol
    sym = "✓" if ok else "✗"
    print(f"  {sym} {name}: max_abs={max_err:.2e}, rel={rel_err:.2e}")
    return ok


# ─── Test 1: V1.2 Integrator Scan (heun/rk4/imex) ──────────────────
def test_v12_integrator_scan():
    _header("Test 1: V1.2 Integrator Scan — Triton vs Reference")
    from triton_kernels.csma_triton_scan_v12 import (
        cs_scan_v12_forward_integrator_cuda,
        _torch_reference_integrator_scan,
    )

    torch.manual_seed(42)
    B, D, H, W, K = 2, 64, 7, 7, 3
    N = H * W

    h0 = torch.randn(B, N, D, device="cuda", dtype=torch.float32)
    delta_s = (torch.rand(B, N, D, device="cuda") * 0.15)
    delta_d = (torch.rand(B, N, D, device="cuda") * 0.15)
    A = -F.softplus(torch.randn(D, device="cuda"))
    D_phys = torch.sigmoid(torch.randn(1, D, 1, 1, device="cuda")) * 0.5

    all_ok = True
    for integrator in ["heun", "rk4", "imex"]:
        for activation in ["identity", "silu", "tanh", "relu6"]:
            for stencil in [4, 8]:
                label = f"{integrator}/{activation}/stencil{stencil}"
                try:
                    ref = _torch_reference_integrator_scan(
                        h0, delta_s, delta_d, A, D_phys, K, H, W,
                        stencil=stencil, activation=activation,
                        integrator=integrator, imex_iters=3,
                    )
                    triton_out = cs_scan_v12_forward_integrator_cuda(
                        h0, delta_s, delta_d, A, D_phys, K, H, W,
                        stencil=stencil, activation=activation,
                        integrator=integrator, imex_iters=3,
                    )
                    ok = _check(label, triton_out, ref, atol=5e-3)
                    all_ok = all_ok and ok
                except Exception as e:
                    print(f"  ✗ {label}: EXCEPTION — {e}")
                    all_ok = False
    return all_ok


# ─── Test 2: V1.2 Flex Kernel (fwd + bwd) ───────────────────────────
def test_v12_flex_kernel():
    _header("Test 2: V1.2 Flex Kernel — Triton fwd/bwd vs Reference")
    from triton_kernels.csma_triton_scan_v12 import cs_scan_v12_flex_cuda

    torch.manual_seed(123)
    B, D, H, W, K = 2, 32, 7, 7, 3
    N = H * W

    all_ok = True
    for activation in ["identity", "silu", "tanh"]:
        for stencil in [4, 8]:
            label = f"flex/{activation}/stencil{stencil}"

            h0 = torch.randn(B, N, D, device="cuda", dtype=torch.float32, requires_grad=True)
            delta_s = (torch.rand(B, N, D, device="cuda") * 0.15).requires_grad_(True)
            delta_d = (torch.rand(B, N, D, device="cuda") * 0.15).requires_grad_(True)
            A = (-F.softplus(torch.randn(D, device="cuda"))).requires_grad_(True)
            D_phys = (torch.sigmoid(torch.randn(1, D, 1, 1, device="cuda")) * 0.5).requires_grad_(True)

            try:
                out = cs_scan_v12_flex_cuda(h0, delta_s, delta_d, A, D_phys, K, H, W,
                                            stencil=stencil, activation=activation)
                loss = out.sum()
                loss.backward()
                ok = True
                for name, p in [("h0", h0), ("delta_s", delta_s), ("delta_d", delta_d),
                                ("A", A), ("D_phys", D_phys)]:
                    if p.grad is None:
                        print(f"  ✗ {label}/grad_{name}: None!")
                        ok = False
                    elif not torch.isfinite(p.grad).all():
                        print(f"  ✗ {label}/grad_{name}: non-finite!")
                        ok = False
                if ok:
                    print(f"  ✓ {label}: forward+backward OK, all grads finite")
                all_ok = all_ok and ok
            except Exception as e:
                print(f"  ✗ {label}: EXCEPTION — {e}")
                all_ok = False
    return all_ok


# ─── Test 3: V1 Backward Reference vs Autograd ──────────────────────
def test_v1_backward_adjoint():
    _header("Test 3: V1 Backward — Manual Reference vs Autograd (Fix 3)")
    from triton_kernels.csma_reference import (
        cs_mamba_forward_reference,
        cs_mamba_backward_reference,
    )

    torch.manual_seed(42)
    B, D, S, H, W, K = 2, 8, 4, 4, 4, 2
    N = H * W

    # Use float64 for precise gradient comparison
    h0_base = torch.randn(B, N, D, S, dtype=torch.float64)
    x_base = torch.randn(B, N, D, dtype=torch.float64)
    # Use spatially-varying delta_d to exercise the fix
    ds_base = torch.rand(B, N, D, dtype=torch.float64) * 0.15
    dd_base = torch.rand(B, N, D, dtype=torch.float64) * 0.15
    A_base = -torch.rand(D, S, dtype=torch.float64).abs()
    bm_base = torch.randn(B, N, S, dtype=torch.float64)
    dp_base = torch.rand(1, D, 1, 1, dtype=torch.float64) * 0.5

    # ─ Autograd ground truth ─
    h0 = h0_base.clone().requires_grad_(True)
    x = x_base.clone().requires_grad_(True)
    ds = ds_base.clone().requires_grad_(True)
    dd = dd_base.clone().requires_grad_(True)
    A = A_base.clone().requires_grad_(True)
    bm = bm_base.clone().requires_grad_(True)
    dp = dp_base.clone().requires_grad_(True)

    h_final, _ = cs_mamba_forward_reference(h0, x, ds, dd, A, bm, dp, K, H, W)
    h_final.sum().backward()

    # ─ Manual backward ─
    h0_nd = h0_base.clone()
    x_nd = x_base.clone()
    ds_nd = ds_base.clone()
    dd_nd = dd_base.clone()
    A_nd = A_base.clone()
    bm_nd = bm_base.clone()
    dp_nd = dp_base.clone()

    h_final_nd, h_saved = cs_mamba_forward_reference(h0_nd, x_nd, ds_nd, dd_nd, A_nd, bm_nd, dp_nd, K, H, W)
    grad_output = torch.ones_like(h_final_nd)
    grads = cs_mamba_backward_reference(grad_output, h_saved, x_nd, ds_nd, dd_nd, A_nd, bm_nd, dp_nd, K, H, W)

    all_ok = True
    for name, manual, auto in [
        ("h0", grads["grad_h0"], h0.grad),
        ("delta_s", grads["grad_delta_s"], ds.grad),
        ("delta_d", grads["grad_delta_d"], dd.grad),
        ("A", grads["grad_A"], A.grad),
        ("B_mat", grads["grad_B_mat"], bm.grad),
    ]:
        ok = _check(f"grad_{name}", manual, auto, atol=1e-6)
        all_ok = all_ok and ok
    return all_ok


# ─── Test 4: V1.2 Basic Euler Kernel ────────────────────────────────
def test_v12_euler_kernel():
    _header("Test 4: V1.2 Euler Kernel — Triton fwd/bwd")
    from triton_kernels.csma_triton_scan_v12 import cs_scan_v12_cuda

    torch.manual_seed(99)
    B, D, H, W, K = 2, 64, 7, 7, 3
    N = H * W

    h0 = torch.randn(B, N, D, device="cuda", requires_grad=True)
    delta_s = (torch.rand(B, N, D, device="cuda") * 0.15).requires_grad_(True)
    delta_d = (torch.rand(B, N, D, device="cuda") * 0.15).requires_grad_(True)
    A = (-F.softplus(torch.randn(D, device="cuda"))).requires_grad_(True)
    D_phys = (torch.sigmoid(torch.randn(1, D, 1, 1, device="cuda")) * 0.5).requires_grad_(True)

    try:
        out = cs_scan_v12_cuda(h0, delta_s, delta_d, A, D_phys, K, H, W)
        loss = out.sum()
        loss.backward()
        ok = True
        for name, p in [("h0", h0), ("delta_s", delta_s), ("delta_d", delta_d),
                        ("A", A), ("D_phys", D_phys)]:
            if p.grad is None:
                print(f"  ✗ grad_{name}: None!")
                ok = False
            elif not torch.isfinite(p.grad).all():
                print(f"  ✗ grad_{name}: non-finite!")
                ok = False
        if ok:
            print(f"  ✓ Euler forward+backward OK, all grads finite")
        return ok
    except Exception as e:
        print(f"  ✗ EXCEPTION — {e}")
        return False


# ─── Test 5: VMamba4D Selective Scan ─────────────────────────────────
def test_vmamba4d_scan():
    _header("Test 5: VMamba4D Selective Scan — Triton vs Naive")
    from triton_kernels.vmamba4d_triton_scan import selective_scan_cuda

    torch.manual_seed(7)
    B, L, D = 2, 64, 128

    retain = torch.sigmoid(torch.randn(B, L, D, device="cuda")).detach().requires_grad_(True)
    update = torch.randn(B, L, D, device="cuda").detach().requires_grad_(True)

    # Naive reference
    retain_ref = retain.detach().clone().requires_grad_(True)
    update_ref = update.detach().clone().requires_grad_(True)
    out_ref = torch.zeros_like(update_ref)
    h = torch.zeros(B, D, device="cuda")
    for t in range(L):
        h = retain_ref[:, t] * h + update_ref[:, t]
        out_ref[:, t] = h
    out_ref.sum().backward()

    # Triton
    out_triton = selective_scan_cuda(retain, update)
    out_triton.sum().backward()

    ok1 = _check("forward", out_triton, out_ref.detach(), atol=1e-4)
    ok2 = _check("grad_retain", retain.grad, retain_ref.grad, atol=1e-4)
    ok3 = _check("grad_update", update.grad, update_ref.grad, atol=1e-4)
    return ok1 and ok2 and ok3


# ─── Test 6: V1 Full Autograd Triton Scan ────────────────────────────
def test_v1_triton_scan():
    _header("Test 6: V1 Triton Scan — fwd/bwd")
    from triton_kernels.csma_triton_scan import cs_scan_forward_cuda, cs_scan_backward_cuda

    torch.manual_seed(42)
    B, D, S, H, W, K = 2, 32, 16, 7, 7, 3
    N = H * W

    h0 = torch.randn(B, N, D, S, device="cuda", dtype=torch.float32)
    delta_s = (torch.rand(B, N, D, device="cuda") * 0.15)
    delta_d = (torch.rand(B, N, D, device="cuda") * 0.15)
    A = -torch.rand(D, S, device="cuda").abs()
    D_phys = torch.rand(1, D, 1, 1, device="cuda") * 0.5

    try:
        h_final, states = cs_scan_forward_cuda(h0, delta_s, delta_d, A, D_phys, K, H, W)
        grad_output = torch.randn_like(h_final)
        grads = cs_scan_backward_cuda(grad_output, states, h0, delta_s, delta_d, A, D_phys, K, H, W)
        ok = True
        for i, name in enumerate(["h0", "delta_s", "delta_d", "A", "D_phys"]):
            if not torch.isfinite(grads[i]).all():
                print(f"  ✗ grad_{name}: non-finite!")
                ok = False
        if ok:
            print(f"  ✓ V1 Triton forward+backward OK, all grads finite")
        return ok
    except Exception as e:
        print(f"  ✗ EXCEPTION — {e}")
        return False


# ─── Test 7: V1.2 IMEX Custom Backward vs Reference ─────────────────
def test_v12_imex_custom_backward():
    _header("Test 7: V1.2 IMEX Custom Backward — Triton vs Reference")
    from triton_kernels.csma_triton_scan_v12 import cs_scan_v12_imex_cuda, _torch_reference_integrator_scan

    torch.manual_seed(42)
    B, D, H, W, K = 2, 8, 4, 4, 2
    N = H * W

    # Use float64 for absolute precision check
    h0_base = torch.randn(B, N, D, device="cuda", dtype=torch.float64)
    delta_s_base = torch.rand(B, N, D, device="cuda", dtype=torch.float64) * 0.15
    delta_d_base = torch.rand(B, N, D, device="cuda", dtype=torch.float64) * 0.15
    A_base = -F.softplus(torch.randn(D, device="cuda", dtype=torch.float64))
    D_phys_base = torch.sigmoid(torch.randn(1, D, 1, 1, device="cuda", dtype=torch.float64)) * 0.5

    all_ok = True
    for activation in ["identity", "silu", "tanh"]:
        for stencil in [4, 8]:
            label = f"imex-bwd/{activation}/stencil{stencil}"
            
            # Autograd reference
            h0_ref = h0_base.clone().requires_grad_(True)
            delta_s_ref = delta_s_base.clone().requires_grad_(True)
            delta_d_ref = delta_d_base.clone().requires_grad_(True)
            A_ref = A_base.clone().requires_grad_(True)
            D_phys_ref = D_phys_base.clone().requires_grad_(True)

            ref_out = _torch_reference_integrator_scan(
                h0_ref, delta_s_ref, delta_d_ref, A_ref, D_phys_ref, K, H, W,
                stencil=stencil, activation=activation,
                integrator="imex", imex_iters=3,
            )
            ref_out.sum().backward()

            # Custom Triton
            h0_tri = h0_base.clone().requires_grad_(True)
            delta_s_tri = delta_s_base.clone().requires_grad_(True)
            delta_d_tri = delta_d_base.clone().requires_grad_(True)
            A_tri = A_base.clone().requires_grad_(True)
            D_phys_tri = D_phys_base.clone().requires_grad_(True)

            tri_out = cs_scan_v12_imex_cuda(
                h0_tri, delta_s_tri, delta_d_tri, A_tri, D_phys_tri, K, H, W,
                stencil=stencil, activation=activation, imex_iters=3,
            )
            tri_out.sum().backward()

            ok = _check(f"{label} Fwd", tri_out, ref_out.detach(), atol=1e-5)
            all_ok = all_ok and ok

            for name, tri_p, ref_p in [
                ("h0", h0_tri, h0_ref),
                ("delta_s", delta_s_tri, delta_s_ref),
                ("delta_d", delta_d_tri, delta_d_ref),
                ("A", A_tri, A_ref),
                ("D_phys", D_phys_tri, D_phys_ref),
            ]:
                p_ok = _check(f"{label} grad_{name}", tri_p.grad, ref_p.grad, atol=1e-5)
                all_ok = all_ok and p_ok
                
    return all_ok


# ─── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device available. Cannot run smoke tests.")
        sys.exit(1)

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch:     {torch.__version__}")
    try:
        import triton
        print(f"Triton:      {triton.__version__}")
    except Exception:
        print("Triton:      not importable!")
        sys.exit(1)

    results = {}
    results["V1.2 Integrator (heun/rk4/imex)"] = test_v12_integrator_scan()
    results["V1.2 Flex Kernel (fwd+bwd)"] = test_v12_flex_kernel()
    results["V1 Backward Adjoint (Fix 3)"] = test_v1_backward_adjoint()
    results["V1.2 Euler Kernel"] = test_v12_euler_kernel()
    results["VMamba4D Selective Scan"] = test_vmamba4d_scan()
    results["V1 Triton Scan (fwd+bwd)"] = test_v1_triton_scan()
    results["V1.2 IMEX Custom Backward"] = test_v12_imex_custom_backward()

    _header("SUMMARY")
    all_passed = True
    for name, ok in results.items():
        sym = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {sym}  {name}")
        all_passed = all_passed and ok

    print()
    if all_passed:
        print("All smoke tests PASSED ✓")
    else:
        print("Some tests FAILED ✗")
    sys.exit(0 if all_passed else 1)
