"""
Triton CUDA implementation of the V1 continuous spatial Mamba recurrence.

The kernel matches models.continuous_spatial_mamba.cs_mamba_forward_reference:

    h_{k+1} = h_k * (1 + dt * delta_s * A)
              + dt * delta_s * h0
              + dt * delta_d * D_phys * laplacian(sum_s h_k)

The implementation keeps one Triton program per (batch, patch, channel tile).
For the ImageNet/Tiny-ImageNet patch grid used here (14x14 at 224/16), this is
small enough to keep the complete state-vector dimension in registers while
loading only the four spatial neighbors needed by the Neumann Laplacian.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - import guard for CPU/TPU environments
    triton = None
    tl = None

if triton is None:  # pragma: no cover - CUDA-only module
    raise ImportError("triton_kernels.csma_triton_scan requires Triton.")


def _require_triton_cuda(*tensors: torch.Tensor) -> None:
    if not all(t.is_cuda for t in tensors if isinstance(t, torch.Tensor)):
        raise RuntimeError("The Triton CS-Mamba scan requires CUDA tensors.")


_NEIGHBOR_INDEX_CACHE: dict[tuple[int, int, int], torch.Tensor] = {}


def grid_neighbor_index(H: int, W: int, device: torch.device | str) -> torch.Tensor:
    """Return flat-grid Neumann neighbor indices with columns [top, bottom, left, right]."""
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("grid_neighbor_index is only used by CUDA/Triton kernels.")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    key = (int(device_index), int(H), int(W))
    cached = _NEIGHBOR_INDEX_CACHE.get(key)
    if cached is not None and cached.device.index == device_index:
        return cached

    with torch.cuda.device(device_index):
        n_tokens = int(H) * int(W)
        idx = torch.arange(n_tokens, device=device, dtype=torch.int64)
        row = idx // int(W)
        col = idx - row * int(W)
        top = torch.where(row > 0, idx - int(W), idx)
        bottom = torch.where(row < int(H) - 1, idx + int(W), idx)
        left = torch.where(col > 0, idx - 1, idx)
        right = torch.where(col < int(W) - 1, idx + 1, idx)
        neighbors = torch.stack((top, bottom, left, right), dim=1).contiguous()
    _NEIGHBOR_INDEX_CACHE[key] = neighbors
    return neighbors


@triton.jit
def _csma_forward_step_kernel(
    H_PTR,
    H0_PTR,
    DS_PTR,
    DD_PTR,
    A_PTR,
    DPHYS_PTR,
    NEIGHBOR_PTR,
    OUT_PTR,
    SAVE_PTR,
    DT: tl.constexpr,
    STEP: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    d_block = tl.program_id(2)

    d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    s = tl.arange(0, BLOCK_S)
    mask_ds = (d[:, None] < D) & (s[None, :] < S)
    mask_d = d < D

    n_top = tl.load(NEIGHBOR_PTR + n * 4 + 0)
    n_bot = tl.load(NEIGHBOR_PTR + n * 4 + 1)
    n_left = tl.load(NEIGHBOR_PTR + n * 4 + 2)
    n_right = tl.load(NEIGHBOR_PTR + n * 4 + 3)

    base = ((b * N + n) * D + d[:, None]) * S + s[None, :]
    top_base = ((b * N + n_top) * D + d[:, None]) * S + s[None, :]
    bot_base = ((b * N + n_bot) * D + d[:, None]) * S + s[None, :]
    left_base = ((b * N + n_left) * D + d[:, None]) * S + s[None, :]
    right_base = ((b * N + n_right) * D + d[:, None]) * S + s[None, :]

    h = tl.load(H_PTR + base, mask=mask_ds, other=0.0).to(tl.float32)
    h0 = tl.load(H0_PTR + base, mask=mask_ds, other=0.0).to(tl.float32)
    top = tl.load(H_PTR + top_base, mask=mask_ds, other=0.0).to(tl.float32)
    bot = tl.load(H_PTR + bot_base, mask=mask_ds, other=0.0).to(tl.float32)
    left = tl.load(H_PTR + left_base, mask=mask_ds, other=0.0).to(tl.float32)
    right = tl.load(H_PTR + right_base, mask=mask_ds, other=0.0).to(tl.float32)

    q = tl.sum(h, axis=1)
    lap = tl.sum(top, axis=1) + tl.sum(bot, axis=1) + tl.sum(left, axis=1) + tl.sum(right, axis=1) - 4.0 * q

    bnd = (b * N + n) * D + d
    ds = tl.load(DS_PTR + bnd, mask=mask_d, other=0.0).to(tl.float32)
    dd = tl.load(DD_PTR + bnd, mask=mask_d, other=0.0).to(tl.float32)
    dphys = tl.load(DPHYS_PTR + d, mask=mask_d, other=0.0).to(tl.float32)
    a = tl.load(A_PTR + d[:, None] * S + s[None, :], mask=mask_ds, other=0.0).to(tl.float32)

    out = h * (1.0 + DT * ds[:, None] * a)
    out += DT * ds[:, None] * h0
    out += DT * dd[:, None] * dphys[:, None] * lap[:, None]

    tl.store(OUT_PTR + base, out, mask=mask_ds)
    tl.store(SAVE_PTR + STEP * B * N * D * S + base, h, mask=mask_ds)


@triton.jit
def _csma_backward_step_kernel(
    G_PTR,
    H_STATE_PTR,
    H0_PTR,
    DS_PTR,
    DD_PTR,
    A_PTR,
    DPHYS_PTR,
    NEIGHBOR_PTR,
    G_PREV_PTR,
    GH0_ACC_PTR,
    GDS_PTR,
    GDD_PTR,
    GA_PTR,
    GDPHYS_PTR,
    DT: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    d_block = tl.program_id(2)

    d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    s = tl.arange(0, BLOCK_S)
    mask_ds = (d[:, None] < D) & (s[None, :] < S)
    mask_d = d < D

    n_top = tl.load(NEIGHBOR_PTR + n * 4 + 0)
    n_bot = tl.load(NEIGHBOR_PTR + n * 4 + 1)
    n_left = tl.load(NEIGHBOR_PTR + n * 4 + 2)
    n_right = tl.load(NEIGHBOR_PTR + n * 4 + 3)

    base = ((b * N + n) * D + d[:, None]) * S + s[None, :]
    top_base = ((b * N + n_top) * D + d[:, None]) * S + s[None, :]
    bot_base = ((b * N + n_bot) * D + d[:, None]) * S + s[None, :]
    left_base = ((b * N + n_left) * D + d[:, None]) * S + s[None, :]
    right_base = ((b * N + n_right) * D + d[:, None]) * S + s[None, :]

    h = tl.load(H_STATE_PTR + base, mask=mask_ds, other=0.0).to(tl.float32)
    h0 = tl.load(H0_PTR + base, mask=mask_ds, other=0.0).to(tl.float32)
    h_top = tl.load(H_STATE_PTR + top_base, mask=mask_ds, other=0.0).to(tl.float32)
    h_bot = tl.load(H_STATE_PTR + bot_base, mask=mask_ds, other=0.0).to(tl.float32)
    h_left = tl.load(H_STATE_PTR + left_base, mask=mask_ds, other=0.0).to(tl.float32)
    h_right = tl.load(H_STATE_PTR + right_base, mask=mask_ds, other=0.0).to(tl.float32)

    g = tl.load(G_PTR + base, mask=mask_ds, other=0.0).to(tl.float32)
    g_top = tl.load(G_PTR + top_base, mask=mask_ds, other=0.0).to(tl.float32)
    g_bot = tl.load(G_PTR + bot_base, mask=mask_ds, other=0.0).to(tl.float32)
    g_left = tl.load(G_PTR + left_base, mask=mask_ds, other=0.0).to(tl.float32)
    g_right = tl.load(G_PTR + right_base, mask=mask_ds, other=0.0).to(tl.float32)

    q = tl.sum(h, axis=1)
    lap_h = (
        tl.sum(h_top, axis=1)
        + tl.sum(h_bot, axis=1)
        + tl.sum(h_left, axis=1)
        + tl.sum(h_right, axis=1)
        - 4.0 * q
    )

    bnd = (b * N + n) * D + d
    top_bnd = (b * N + n_top) * D + d
    bot_bnd = (b * N + n_bot) * D + d
    left_bnd = (b * N + n_left) * D + d
    right_bnd = (b * N + n_right) * D + d

    ds = tl.load(DS_PTR + bnd, mask=mask_d, other=0.0).to(tl.float32)
    dd = tl.load(DD_PTR + bnd, mask=mask_d, other=0.0).to(tl.float32)
    dd_top = tl.load(DD_PTR + top_bnd, mask=mask_d, other=0.0).to(tl.float32)
    dd_bot = tl.load(DD_PTR + bot_bnd, mask=mask_d, other=0.0).to(tl.float32)
    dd_left = tl.load(DD_PTR + left_bnd, mask=mask_d, other=0.0).to(tl.float32)
    dd_right = tl.load(DD_PTR + right_bnd, mask=mask_d, other=0.0).to(tl.float32)

    dphys = tl.load(DPHYS_PTR + d, mask=mask_d, other=0.0).to(tl.float32)
    a = tl.load(A_PTR + d[:, None] * S + s[None, :], mask=mask_ds, other=0.0).to(tl.float32)

    gsum = tl.sum(g, axis=1)
    r = DT * dd * dphys * gsum
    r_top = DT * dd_top * dphys * tl.sum(g_top, axis=1)
    r_bot = DT * dd_bot * dphys * tl.sum(g_bot, axis=1)
    r_left = DT * dd_left * dphys * tl.sum(g_left, axis=1)
    r_right = DT * dd_right * dphys * tl.sum(g_right, axis=1)
    lap_r = r_top + r_bot + r_left + r_right - 4.0 * r

    g_prev = g * (1.0 + DT * ds[:, None] * a) + lap_r[:, None]
    tl.store(G_PREV_PTR + base, g_prev, mask=mask_ds)

    gh0_add = DT * ds[:, None] * g
    old_gh0 = tl.load(GH0_ACC_PTR + base, mask=mask_ds, other=0.0).to(tl.float32)
    tl.store(GH0_ACC_PTR + base, old_gh0 + gh0_add, mask=mask_ds)

    grad_ds_add = DT * tl.sum(g * (a * h + h0), axis=1)
    grad_dd_add = DT * dphys * lap_h * gsum

    old_gds = tl.load(GDS_PTR + bnd, mask=mask_d, other=0.0).to(tl.float32)
    old_gdd = tl.load(GDD_PTR + bnd, mask=mask_d, other=0.0).to(tl.float32)
    tl.store(GDS_PTR + bnd, old_gds + grad_ds_add, mask=mask_d)
    tl.store(GDD_PTR + bnd, old_gdd + grad_dd_add, mask=mask_d)

    tl.atomic_add(GA_PTR + d[:, None] * S + s[None, :], DT * ds[:, None] * g * h, sem="relaxed", mask=mask_ds)
    tl.atomic_add(GDPHYS_PTR + d, DT * dd * lap_h * gsum, sem="relaxed", mask=mask_d)


@triton.jit
def _add_kernel(
    A_PTR,
    B_PTR,
    OUT_PTR,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    a = tl.load(A_PTR + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_PTR + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT_PTR + offsets, a + b, mask=mask)


def _block_s(s_dim: int) -> int:
    return max(16, triton.next_power_of_2(s_dim))


@triton.jit
def _csma_forward_s16_step_kernel(
    H_PTR,
    H0_PTR,
    DS_PTR,
    DD_PTR,
    A_PTR,
    DPHYS_PTR,
    NEIGHBOR_PTR,
    OUT_PTR,
    SAVE_PTR,
    DT: tl.constexpr,
    STEP: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    d_block = tl.program_id(2)

    d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    s = tl.arange(0, 16)
    mask_d = d < D
    mask_ds = (d[:, None] < D) & (s[None, :] < 16)

    n_top = tl.load(NEIGHBOR_PTR + n * 4 + 0)
    n_bot = tl.load(NEIGHBOR_PTR + n * 4 + 1)
    n_left = tl.load(NEIGHBOR_PTR + n * 4 + 2)
    n_right = tl.load(NEIGHBOR_PTR + n * 4 + 3)

    base = ((b * N + n) * D + d[:, None]) * 16 + s[None, :]
    top_base = ((b * N + n_top) * D + d[:, None]) * 16 + s[None, :]
    bot_base = ((b * N + n_bot) * D + d[:, None]) * 16 + s[None, :]
    left_base = ((b * N + n_left) * D + d[:, None]) * 16 + s[None, :]
    right_base = ((b * N + n_right) * D + d[:, None]) * 16 + s[None, :]

    h = tl.load(H_PTR + base, mask=mask_ds, other=0.0).to(tl.float32)
    h0 = tl.load(H0_PTR + base, mask=mask_ds, other=0.0).to(tl.float32)
    top = tl.load(H_PTR + top_base, mask=mask_ds, other=0.0).to(tl.float32)
    bot = tl.load(H_PTR + bot_base, mask=mask_ds, other=0.0).to(tl.float32)
    left = tl.load(H_PTR + left_base, mask=mask_ds, other=0.0).to(tl.float32)
    right = tl.load(H_PTR + right_base, mask=mask_ds, other=0.0).to(tl.float32)

    q = tl.sum(h, axis=1)
    lap = tl.sum(top, axis=1) + tl.sum(bot, axis=1) + tl.sum(left, axis=1) + tl.sum(right, axis=1) - 4.0 * q

    bnd = (b * N + n) * D + d
    ds = tl.load(DS_PTR + bnd, mask=mask_d, other=0.0).to(tl.float32)
    dd = tl.load(DD_PTR + bnd, mask=mask_d, other=0.0).to(tl.float32)
    dphys = tl.load(DPHYS_PTR + d, mask=mask_d, other=0.0).to(tl.float32)
    a = tl.load(A_PTR + d[:, None] * 16 + s[None, :], mask=mask_ds, other=0.0).to(tl.float32)

    out = h * (1.0 + DT * ds[:, None] * a)
    out += DT * ds[:, None] * h0
    out += DT * dd[:, None] * dphys[:, None] * lap[:, None]

    tl.store(OUT_PTR + base, out, mask=mask_ds)
    tl.store(SAVE_PTR + STEP * B * N * D * 16 + base, h, mask=mask_ds)


def cs_scan_forward_cuda(
    h0: torch.Tensor,
    delta_s: torch.Tensor,
    delta_d: torch.Tensor,
    A: torch.Tensor,
    D_phys: torch.Tensor,
    K: int,
    H: int,
    W: int,
    *,
    neighbor_index: torch.Tensor | None = None,
    block_d: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return final h and saved pre-step states for the custom backward."""
    _require_triton_cuda(h0, delta_s, delta_d, A, D_phys)
    if not h0.is_contiguous():
        h0 = h0.contiguous()
    delta_s = delta_s.contiguous()
    delta_d = delta_d.contiguous()
    A = A.contiguous()
    D_phys_flat = D_phys.contiguous().view(-1)

    bsz, n_tokens, d_dim, s_dim = h0.shape
    if n_tokens != H * W:
        raise ValueError(f"H*W={H * W} does not match N={n_tokens}.")
    if A.shape != (d_dim, s_dim):
        raise ValueError(f"A shape {tuple(A.shape)} does not match {(d_dim, s_dim)}.")
    if D_phys_flat.numel() != d_dim:
        raise ValueError(f"D_phys has {D_phys_flat.numel()} values, expected {d_dim}.")
    if neighbor_index is None:
        neighbor_index = grid_neighbor_index(H, W, h0.device)
    else:
        _require_triton_cuda(neighbor_index)
        neighbor_index = neighbor_index.contiguous()
        if tuple(neighbor_index.shape) != (n_tokens, 4):
            raise ValueError(f"neighbor_index shape {tuple(neighbor_index.shape)} must be {(n_tokens, 4)}.")

    h = h0.contiguous()
    work_a = torch.empty_like(h)
    work_b = torch.empty_like(h)
    states = torch.empty((K, bsz, n_tokens, d_dim, s_dim), device=h.device, dtype=h.dtype)
    grid = (bsz, n_tokens, triton.cdiv(d_dim, block_d))
    block_s = _block_s(s_dim)
    dt = 1.0 / float(K)

    for step in range(K):
        h_next = work_a if step % 2 == 0 else work_b
        _csma_forward_step_kernel[grid](
            h,
            h0,
            delta_s,
            delta_d,
            A,
            D_phys_flat,
            neighbor_index,
            h_next,
            states,
            DT=dt,
            STEP=step,
            B=bsz,
            N=n_tokens,
            D=d_dim,
            S=s_dim,
            BLOCK_D=block_d,
            BLOCK_S=block_s,
            num_warps=4,
        )
        h = h_next

    return h, states


def cs_scan_forward_s16_cuda(
    h0: torch.Tensor,
    delta_s: torch.Tensor,
    delta_d: torch.Tensor,
    A: torch.Tensor,
    D_phys: torch.Tensor,
    K: int,
    H: int,
    W: int,
    *,
    neighbor_index: torch.Tensor | None = None,
    block_d: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_triton_cuda(h0, delta_s, delta_d, A, D_phys)
    if not h0.is_contiguous():
        h0 = h0.contiguous()
    delta_s = delta_s.contiguous()
    delta_d = delta_d.contiguous()
    A = A.contiguous()
    D_phys_flat = D_phys.contiguous().view(-1)

    bsz, n_tokens, d_dim, s_dim = h0.shape
    if s_dim != 16:
        raise ValueError(f"cs_scan_forward_s16_cuda requires S=16, got S={s_dim}.")
    if n_tokens != H * W:
        raise ValueError(f"H*W={H * W} does not match N={n_tokens}.")
    if A.shape != (d_dim, 16):
        raise ValueError(f"A shape {tuple(A.shape)} does not match {(d_dim, 16)}.")
    if D_phys_flat.numel() != d_dim:
        raise ValueError(f"D_phys has {D_phys_flat.numel()} values, expected {d_dim}.")
    if neighbor_index is None:
        neighbor_index = grid_neighbor_index(H, W, h0.device)
    else:
        _require_triton_cuda(neighbor_index)
        neighbor_index = neighbor_index.contiguous()
        if tuple(neighbor_index.shape) != (n_tokens, 4):
            raise ValueError(f"neighbor_index shape {tuple(neighbor_index.shape)} must be {(n_tokens, 4)}.")

    h = h0.contiguous()
    work_a = torch.empty_like(h)
    work_b = torch.empty_like(h)
    states = torch.empty((K, bsz, n_tokens, d_dim, 16), device=h.device, dtype=h.dtype)
    grid = (bsz, n_tokens, triton.cdiv(d_dim, block_d))
    dt = 1.0 / float(K)

    for step in range(K):
        h_next = work_a if step % 2 == 0 else work_b
        _csma_forward_s16_step_kernel[grid](
            h,
            h0,
            delta_s,
            delta_d,
            A,
            D_phys_flat,
            neighbor_index,
            h_next,
            states,
            DT=dt,
            STEP=step,
            B=bsz,
            N=n_tokens,
            D=d_dim,
            BLOCK_D=block_d,
            num_warps=4,
        )
        h = h_next

    return h, states


def cs_scan_backward_cuda(
    grad_output: torch.Tensor,
    states: torch.Tensor,
    h0: torch.Tensor,
    delta_s: torch.Tensor,
    delta_d: torch.Tensor,
    A: torch.Tensor,
    D_phys: torch.Tensor,
    K: int,
    H: int,
    W: int,
    *,
    neighbor_index: torch.Tensor | None = None,
    block_d: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_triton_cuda(grad_output, states, h0, delta_s, delta_d, A, D_phys)
    grad_output = grad_output.contiguous()
    states = states.contiguous()
    h0 = h0.contiguous()
    delta_s = delta_s.contiguous()
    delta_d = delta_d.contiguous()
    A = A.contiguous()
    D_phys_flat = D_phys.contiguous().view(-1)

    bsz, n_tokens, d_dim, s_dim = h0.shape
    if neighbor_index is None:
        neighbor_index = grid_neighbor_index(H, W, h0.device)
    else:
        _require_triton_cuda(neighbor_index)
        neighbor_index = neighbor_index.contiguous()
        if tuple(neighbor_index.shape) != (n_tokens, 4):
            raise ValueError(f"neighbor_index shape {tuple(neighbor_index.shape)} must be {(n_tokens, 4)}.")

    g = grad_output
    g_prev = torch.empty_like(g)
    grad_h0_input = torch.zeros_like(h0)
    grad_delta_s = torch.zeros_like(delta_s)
    grad_delta_d = torch.zeros_like(delta_d)
    grad_A = torch.zeros_like(A)
    grad_D_phys_flat = torch.zeros_like(D_phys_flat)

    grid = (bsz, n_tokens, triton.cdiv(d_dim, block_d))
    block_s = _block_s(s_dim)
    dt = 1.0 / float(K)

    for step in range(K - 1, -1, -1):
        h_state = states[step]
        _csma_backward_step_kernel[grid](
            g,
            h_state,
            h0,
            delta_s,
            delta_d,
            A,
            D_phys_flat,
            neighbor_index,
            g_prev,
            grad_h0_input,
            grad_delta_s,
            grad_delta_d,
            grad_A,
            grad_D_phys_flat,
            DT=dt,
            B=bsz,
            N=n_tokens,
            D=d_dim,
            S=s_dim,
            BLOCK_D=block_d,
            BLOCK_S=block_s,
            num_warps=4,
        )
        g, g_prev = g_prev, g

    grad_h0 = torch.empty_like(h0)
    total = h0.numel()
    _add_kernel[(triton.cdiv(total, 1024),)](
        grad_h0_input,
        g,
        grad_h0,
        TOTAL=total,
        BLOCK=1024,
        num_warps=4,
    )
    grad_D_phys = grad_D_phys_flat.reshape_as(D_phys)
    return grad_h0, grad_delta_s, grad_delta_d, grad_A, grad_D_phys


class CSTritonScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h0, x, delta_s, delta_d, A, B_mat, D_phys, K, H, W):
        del x, B_mat
        use_s16 = h0.shape[-1] == 16 and os.environ.get("CS_MAMBA_USE_S16_FORWARD") == "1"
        if use_s16:
            h, states = cs_scan_forward_s16_cuda(h0, delta_s, delta_d, A, D_phys, int(K), int(H), int(W))
        else:
            h, states = cs_scan_forward_cuda(h0, delta_s, delta_d, A, D_phys, int(K), int(H), int(W))
        ctx.save_for_backward(states, h0, delta_s, delta_d, A, D_phys)
        ctx.K = int(K)
        ctx.H = int(H)
        ctx.W = int(W)
        return h

    @staticmethod
    def backward(ctx, grad_output):
        states, h0, delta_s, delta_d, A, D_phys = ctx.saved_tensors
        grad_h0, grad_delta_s, grad_delta_d, grad_A, grad_D_phys = cs_scan_backward_cuda(
            grad_output,
            states,
            h0,
            delta_s,
            delta_d,
            A,
            D_phys,
            ctx.K,
            ctx.H,
            ctx.W,
        )
        return (
            grad_h0,
            None,
            grad_delta_s,
            grad_delta_d,
            grad_A,
            None,
            grad_D_phys,
            None,
            None,
            None,
        )


def cs_scan_cuda(h0, x, delta_s, delta_d, A, B_mat, D_phys, K, H, W):
    return CSTritonScanFunction.apply(h0, x, delta_s, delta_d, A, B_mat, D_phys, K, H, W)
