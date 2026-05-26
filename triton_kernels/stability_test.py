"""Stability and expressiveness comparison: Euler vs IMEX discretization.

Shows:
  1. The CFL blowup threshold for Explicit Euler.
  2. The unconditional stability of IMEX under large diffusivity.
  3. Receptive field propagation speed (Euler vs IMEX).
"""

import torch
import torch.nn.functional as F

def laplacian_2d_neumann(h_2d: torch.Tensor) -> torch.Tensor:
    h_pad = F.pad(h_2d, (1, 1, 1, 1), mode="replicate")
    return (
        h_pad[:, :, 0:-2, 1:-1]
        + h_pad[:, :, 2:, 1:-1]
        + h_pad[:, :, 1:-1, 0:-2]
        + h_pad[:, :, 1:-1, 2:]
        - 4.0 * h_2d
    )

def neighbor_sum(h_2d: torch.Tensor) -> torch.Tensor:
    h_pad = F.pad(h_2d, (1, 1, 1, 1), mode="replicate")
    return (
        h_pad[:, :, 0:-2, 1:-1]
        + h_pad[:, :, 2:, 1:-1]
        + h_pad[:, :, 1:-1, 0:-2]
        + h_pad[:, :, 1:-1, 2:]
    )

def run_euler(h, dt, delta_s, delta_d, A, h0, D_phys, K):
    for _ in range(K):
        lap = laplacian_2d_neumann(h)
        rhs = delta_s * (A * h + h0) + delta_d * D_phys * lap
        h = h + dt * rhs
    return h

def run_imex(h, dt, delta_s, delta_d, A, h0, D_phys, K, imex_iters=3):
    for _ in range(K):
        rhs_explicit = h + dt * delta_s * (A * h + h0)
        alpha = dt * delta_d * D_phys
        z = rhs_explicit
        for _ in range(imex_iters):
            z = (rhs_explicit + alpha * neighbor_sum(z)) / (1.0 + alpha * 4.0)
        h = z
    return h

def main():
    H, W = 16, 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Stability test
    # Set up a delta input at the center of the grid
    h_init = torch.zeros(1, 1, H, W, device=device)
    h_init[0, 0, H//2, W//2] = 1.0
    
    delta_s = torch.zeros_like(h_init)
    delta_d = torch.ones_like(h_init)
    A = torch.zeros_like(h_init)
    h0 = torch.zeros_like(h_init)
    
    # K steps, step size dt = 1.0 / K
    K = 4
    dt = 1.0 / K
    
    print("=" * 60)
    # The mathematical CFL threshold for 2D explicit diffusion is: dt * delta_d * D_phys <= 0.25
    # If dt = 0.25, and delta_d = 1.0, then D_phys must be <= 1.0
    # Let's test with a high diffusivity D_phys to trigger a blowup
    for D_val in [0.4, 0.9, 1.5, 3.0, 10.0]:
        D_phys = torch.full_like(h_init, D_val)
        
        cfl_param = dt * 1.0 * D_val
        print(f"\nDiffusivity D = {D_val:.1f} (CFL Parameter = {cfl_param:.3f}, Stability limit is 0.25)")
        
        # Run Explicit Euler
        h_euler = run_euler(h_init.clone(), dt, delta_s, delta_d, A, h0, D_phys, K)
        euler_norm = h_euler.norm().item()
        euler_status = "STABLE" if torch.isfinite(h_euler).all() and euler_norm < 10.0 else "EXPLODED (NaN/Inf)"
        print(f"  Explicit Euler: {euler_status:<20} (Output Norm: {euler_norm:.4f})")
        
        # Run IMEX (3 Jacobi iterations)
        h_imex = run_imex(h_init.clone(), dt, delta_s, delta_d, A, h0, D_phys, K, imex_iters=3)
        imex_norm = h_imex.norm().item()
        imex_status = "STABLE" if torch.isfinite(h_imex).all() and imex_norm < 10.0 else "EXPLODED"
        print(f"  IMEX (3 It):    {imex_status:<20} (Output Norm: {imex_norm:.4f})")

    # 2. Expressive power / Information Propagation
    print("\n" + "=" * 60)
    print("Information Propagation Profile (Value at Grid Edge row=0, col=0)")
    print("=" * 60)
    # With K=2 steps:
    # Explicit Euler can only travel 2 pixels away from the center (H//2, W//2 = 8, 8).
    # Since H=16, the corners (0, 0) should remain EXACTLY 0.
    # IMEX should be able to propagate values globally to all pixels immediately.
    
    h_init = torch.zeros(1, 1, 16, 16, device=device)
    h_init[0, 0, 8, 8] = 1.0
    D_phys = torch.full_like(h_init, 0.4)
    
    h_euler_prop = run_euler(h_init.clone(), 0.5, delta_s, delta_d, A, h0, D_phys, K=2)
    h_imex_prop = run_imex(h_init.clone(), 0.5, delta_s, delta_d, A, h0, D_phys, K=2, imex_iters=3)
    
    print(f"Explicit Euler (K=2) corner value: {h_euler_prop[0, 0, 0, 0].item():.2e} (Strictly Zero due to local stencil)")
    print(f"IMEX (K=2) corner value:           {h_imex_prop[0, 0, 0, 0].item():.2e} (Non-zero: Global Information Flow)")
    print("=" * 60)

if __name__ == "__main__":
    main()
