"""Benchmark script comparing CS-Mamba Triton kernels vs pure PyTorch reference.

Measures:
  1. V1 Forward & Backward (Triton vs Autograd Reference)
  2. V1.2 Forward & Backward (Triton vs Autograd Reference)

Usage:
    source ~/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/activate
    cd ~/Downloads/Documents/Work/Research/CVPR/learning/CG_Mamba
    python -m triton_kernels.benchmark
"""

import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Precision/Profiling setup
torch.backends.cudnn.benchmark = True

def benchmark_op(func, args, kwargs, name, warmup=50, iters=200):
    # Warmup
    for _ in range(warmup):
        out = func(*args, **kwargs)
        if isinstance(out, tuple):
            loss = out[0].sum()
        else:
            loss = out.sum()
        loss.backward(retain_graph=True)
    
    torch.cuda.synchronize()
    
    # Timing Forward
    start_evt_fwd = torch.cuda.Event(enable_timing=True)
    end_evt_fwd = torch.cuda.Event(enable_timing=True)
    
    start_evt_fwd.record()
    for _ in range(iters):
        out = func(*args, **kwargs)
    end_evt_fwd.record()
    
    torch.cuda.synchronize()
    fwd_time = start_evt_fwd.elapsed_time(end_evt_fwd) / iters
    
    # Get last out for backward
    out = func(*args, **kwargs)
    if isinstance(out, tuple):
        out_tensor = out[0]
    else:
        out_tensor = out
    
    # Reset grad
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.requires_grad:
            if arg.grad is not None:
                arg.grad.zero_()
                
    grad_output = torch.ones_like(out_tensor)
    
    # Timing Backward
    start_evt_bwd = torch.cuda.Event(enable_timing=True)
    end_evt_bwd = torch.cuda.Event(enable_timing=True)
    
    start_evt_bwd.record()
    for _ in range(iters):
        out_tensor.backward(grad_output, retain_graph=True)
    end_evt_bwd.record()
    
    torch.cuda.synchronize()
    bwd_time = start_evt_bwd.elapsed_time(end_evt_bwd) / iters
    
    return fwd_time, bwd_time

def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # Settings matching typical training sizes
    B, D, H, W, K = 8, 128, 14, 14, 3
    N = H * W
    
    # ==========================================
    # Benchmark V1 (4D State)
    # ==========================================
    print("\nBenchmarking V1 (4D State - S=16)...")
    from triton_kernels.csma_autograd import CSScanFunction
    from triton_kernels.csma_triton_scan import cs_scan_cuda
    
    S = 16
    h0 = torch.randn(B, N, D, S, device="cuda", dtype=torch.float32, requires_grad=True)
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float32, requires_grad=True)
    ds = (torch.rand(B, N, D, device="cuda", dtype=torch.float32) * 0.15).requires_grad_(True)
    dd = (torch.rand(B, N, D, device="cuda", dtype=torch.float32) * 0.15).requires_grad_(True)
    A = (-F.softplus(torch.randn(D, S, device="cuda", dtype=torch.float32))).requires_grad_(True)
    bm = torch.randn(B, N, S, device="cuda", dtype=torch.float32, requires_grad=True)
    dp = (torch.rand(1, D, 1, 1, device="cuda", dtype=torch.float32) * 0.5).requires_grad_(True)
    
    def v1_ref(h0, x, ds, dd, A, bm, dp):
        return CSScanFunction.apply(h0, x, ds, dd, A, bm, dp, K, H, W)
        
    def v1_triton(h0, x, ds, dd, A, bm, dp):
        return cs_scan_cuda(h0, x, ds, dd, A, bm, dp, K, H, W)
        
    v1_ref_fwd, v1_ref_bwd = benchmark_op(v1_ref, (h0, x, ds, dd, A, bm, dp), {}, "V1 Ref")
    v1_tri_fwd, v1_tri_bwd = benchmark_op(v1_triton, (h0, x, ds, dd, A, bm, dp), {}, "V1 Triton")
    
    # ==========================================
    # Benchmark V1.2 (3D State)
    # ==========================================
    print("Benchmarking V1.2 (3D State)...")
    from triton_kernels.csma_triton_scan_v12 import cs_scan_v12_cuda, cs_scan_v12_flex_cuda
    from triton_kernels.csma_triton_scan_v12 import _torch_reference_integrator_scan
    
    # Clear cache/re-init
    h0_v12 = torch.randn(B, N, D, device="cuda", dtype=torch.float32, requires_grad=True)
    ds_v12 = (torch.rand(B, N, D, device="cuda", dtype=torch.float32) * 0.15).requires_grad_(True)
    dd_v12 = (torch.rand(B, N, D, device="cuda", dtype=torch.float32) * 0.15).requires_grad_(True)
    A_v12 = (-F.softplus(torch.randn(D, device="cuda", dtype=torch.float32))).requires_grad_(True)
    dp_v12 = (torch.rand(1, D, 1, 1, device="cuda", dtype=torch.float32) * 0.5).requires_grad_(True)
    
    def v12_ref(h0, ds, dd, A, dp):
        # Emulate reference step loop
        return _torch_reference_integrator_scan(
            h0, ds, dd, A, dp, K, H, W,
            stencil=4, activation="identity",
            integrator="heun", imex_iters=3
        )
        
    def v12_triton(h0, ds, dd, A, dp):
        return cs_scan_v12_cuda(h0, ds, dd, A, dp, K, H, W)
        
    v12_ref_fwd, v12_ref_bwd = benchmark_op(v12_ref, (h0_v12, ds_v12, dd_v12, A_v12, dp_v12), {}, "V1.2 Ref")
    v12_tri_fwd, v12_tri_bwd = benchmark_op(v12_triton, (h0_v12, ds_v12, dd_v12, A_v12, dp_v12), {}, "V1.2 Triton")
    
    # ==========================================
    # Benchmark V1.2 IMEX (3D State)
    # ==========================================
    print("Benchmarking V1.2 IMEX...")
    from triton_kernels.csma_triton_scan_v12 import cs_scan_v12_imex_cuda
    
    h0_imex = torch.randn(B, N, D, device="cuda", dtype=torch.float32, requires_grad=True)
    ds_imex = (torch.rand(B, N, D, device="cuda", dtype=torch.float32) * 0.15).requires_grad_(True)
    dd_imex = (torch.rand(B, N, D, device="cuda", dtype=torch.float32) * 0.15).requires_grad_(True)
    A_imex = (-F.softplus(torch.randn(D, device="cuda", dtype=torch.float32))).requires_grad_(True)
    dp_imex = (torch.rand(1, D, 1, 1, device="cuda", dtype=torch.float32) * 0.5).requires_grad_(True)
    
    def imex_ref(h0, ds, dd, A, dp):
        return _torch_reference_integrator_scan(
            h0, ds, dd, A, dp, K, H, W,
            stencil=4, activation="silu",
            integrator="imex", imex_iters=3
        )
        
    def imex_triton(h0, ds, dd, A, dp):
        return cs_scan_v12_imex_cuda(
            h0, ds, dd, A, dp, K, H, W,
            stencil=4, activation="silu", imex_iters=3
        )
        
    imex_ref_fwd, imex_ref_bwd = benchmark_op(imex_ref, (h0_imex, ds_imex, dd_imex, A_imex, dp_imex), {}, "IMEX Ref")
    imex_tri_fwd, imex_tri_bwd = benchmark_op(imex_triton, (h0_imex, ds_imex, dd_imex, A_imex, dp_imex), {}, "IMEX Triton")
    
    print("\n" + "="*70)
    print("  CS-MAMBA KERNEL SPEED BENCHMARK SUMMARY")
    print("="*70)
    print(f"| Model / Direction | PyTorch Reference | Triton Kernel | Speedup |")
    print(f"|-------------------|-------------------|---------------|---------|")
    print(f"| **V1 (4D)** Fwd   | {v1_ref_fwd:14.3f} ms | {v1_tri_fwd:10.3f} ms | {v1_ref_fwd/v1_tri_fwd:6.2f}x |")
    print(f"| **V1 (4D)** Bwd   | {v1_ref_bwd:14.3f} ms | {v1_tri_bwd:10.3f} ms | {v1_ref_bwd/v1_tri_bwd:6.2f}x |")
    print(f"| **V1.2 (3D)** Fwd | {v12_ref_fwd:14.3f} ms | {v12_tri_fwd:10.3f} ms | {v12_ref_fwd/v12_tri_fwd:6.2f}x |")
    print(f"| **V1.2 (3D)** Bwd | {v12_ref_bwd:14.3f} ms | {v12_tri_bwd:10.3f} ms | {v12_ref_bwd/v12_tri_bwd:6.2f}x |")
    print(f"| **IMEX** Fwd      | {imex_ref_fwd:14.3f} ms | {imex_tri_fwd:10.3f} ms | {imex_ref_fwd/imex_tri_fwd:6.2f}x |")
    print(f"| **IMEX** Bwd      | {imex_ref_bwd:14.3f} ms | {imex_tri_bwd:10.3f} ms | {imex_ref_bwd/imex_tri_bwd:6.2f}x |")
    print("="*70)
    
if __name__ == "__main__":
    main()
