"""Triton CUDA kernels for CS-Mamba V1.2 3D-state recurrence."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

if triton is None:  # pragma: no cover
    raise ImportError("triton_kernels.csma_triton_scan_v12 requires Triton.")

from triton_kernels.csma_triton_scan import grid_neighbor_index


def _require_triton_cuda(*tensors: torch.Tensor) -> None:
    if not all(t.is_cuda for t in tensors if isinstance(t, torch.Tensor)):
        raise RuntimeError("The Triton CS-Mamba V1.2 scan requires CUDA tensors.")


_ACT_TO_ID = {
    "identity": 0,
    "silu": 1,
    "tanh": 2,
    "relu6": 3,
    "relu": 4,
    "gelu": 5,
}


def _activation_id(name: str) -> int:
    try:
        return _ACT_TO_ID[name]
    except KeyError as exc:
        raise ValueError(f"Triton V1.2 scan does not support recurrence_nonlinearity={name!r}") from exc


@triton.jit
def _v12_apply_activation(x, ACT: tl.constexpr):
    if ACT == 0:
        return x
    if ACT == 1:
        return x * tl.sigmoid(x)
    if ACT == 2:
        return 2.0 * tl.sigmoid(2.0 * x) - 1.0
    if ACT == 3:
        return tl.minimum(tl.maximum(x, 0.0), 6.0)
    if ACT == 4:
        return tl.maximum(x, 0.0)
    if ACT == 5:
        return 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    return x


@triton.jit
def _v12_activation_grad(x, ACT: tl.constexpr):
    if ACT == 0:
        return x * 0.0 + 1.0
    if ACT == 1:
        sig = tl.sigmoid(x)
        return sig * (1.0 + x * (1.0 - sig))
    if ACT == 2:
        t = 2.0 * tl.sigmoid(2.0 * x) - 1.0
        return 1.0 - t * t
    if ACT == 3:
        return tl.where((x > 0.0) & (x < 6.0), 1.0, 0.0)
    if ACT == 4:
        return tl.where(x > 0.0, 1.0, 0.0)
    if ACT == 5:
        cdf = 0.5 * (1.0 + tl.erf(x * 0.7071067811865476))
        pdf_x = 0.3989422804014327 * x * tl.exp(-0.5 * x * x)
        return cdf + pdf_x
    return x * 0.0 + 1.0


@triton.jit
def _v12_forward_step_flex_kernel(
    H_PTR,
    H0_PTR,
    DS_PTR,
    DD_PTR,
    A_PTR,
    DPHYS_PTR,
    OUT_PTR,
    SAVE_PTR,
    PREACT_PTR,
    DT: tl.constexpr,
    STEP: tl.constexpr,
    B: tl.constexpr,
    HGRID: tl.constexpr,
    WGRID: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    STENCIL: tl.constexpr,
    ACT: tl.constexpr,
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    d_block = tl.program_id(2)

    d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d < D
    row = n // WGRID
    col = n - row * WGRID

    n_top = tl.where(row > 0, n - WGRID, n)
    n_bot = tl.where(row < HGRID - 1, n + WGRID, n)
    n_left = tl.where(col > 0, n - 1, n)
    n_right = tl.where(col < WGRID - 1, n + 1, n)
    n_tl = tl.where((row > 0) & (col > 0), n - WGRID - 1, tl.where(row > 0, n - WGRID, tl.where(col > 0, n - 1, n)))
    n_tr = tl.where((row > 0) & (col < WGRID - 1), n - WGRID + 1, tl.where(row > 0, n - WGRID, tl.where(col < WGRID - 1, n + 1, n)))
    n_bl = tl.where((row < HGRID - 1) & (col > 0), n + WGRID - 1, tl.where(row < HGRID - 1, n + WGRID, tl.where(col > 0, n - 1, n)))
    n_br = tl.where((row < HGRID - 1) & (col < WGRID - 1), n + WGRID + 1, tl.where(row < HGRID - 1, n + WGRID, tl.where(col < WGRID - 1, n + 1, n)))

    base = (b * N + n) * D + d
    h = tl.load(H_PTR + base, mask=mask, other=0.0).to(tl.float32)
    h0 = tl.load(H0_PTR + base, mask=mask, other=0.0).to(tl.float32)
    top = tl.load(H_PTR + (b * N + n_top) * D + d, mask=mask, other=0.0).to(tl.float32)
    bot = tl.load(H_PTR + (b * N + n_bot) * D + d, mask=mask, other=0.0).to(tl.float32)
    left = tl.load(H_PTR + (b * N + n_left) * D + d, mask=mask, other=0.0).to(tl.float32)
    right = tl.load(H_PTR + (b * N + n_right) * D + d, mask=mask, other=0.0).to(tl.float32)
    neighbor_sum = top + bot + left + right
    degree = 4.0
    if STENCIL == 8:
        tlv = tl.load(H_PTR + (b * N + n_tl) * D + d, mask=mask, other=0.0).to(tl.float32)
        trv = tl.load(H_PTR + (b * N + n_tr) * D + d, mask=mask, other=0.0).to(tl.float32)
        blv = tl.load(H_PTR + (b * N + n_bl) * D + d, mask=mask, other=0.0).to(tl.float32)
        brv = tl.load(H_PTR + (b * N + n_br) * D + d, mask=mask, other=0.0).to(tl.float32)
        neighbor_sum += tlv + trv + blv + brv
        degree = 8.0

    ds = tl.load(DS_PTR + base, mask=mask, other=0.0).to(tl.float32)
    dd = tl.load(DD_PTR + base, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(A_PTR + d, mask=mask, other=0.0).to(tl.float32)
    dphys = tl.load(DPHYS_PTR + d, mask=mask, other=0.0).to(tl.float32)

    lap = neighbor_sum - degree * h
    pre = h + DT * (ds * (a * h + h0) + dd * dphys * lap)
    out = _v12_apply_activation(pre, ACT)

    tl.store(OUT_PTR + base, out, mask=mask)
    tl.store(SAVE_PTR + STEP * B * N * D + base, h, mask=mask)
    tl.store(PREACT_PTR + STEP * B * N * D + base, pre, mask=mask)


@triton.jit
def _v12_backward_step_flex_kernel(
    G_PTR,
    H_STATE_PTR,
    PREACT_PTR,
    H0_PTR,
    DS_PTR,
    DD_PTR,
    A_PTR,
    DPHYS_PTR,
    G_PREV_PTR,
    GH0_ACC_PTR,
    GDS_PTR,
    GDD_PTR,
    GA_PTR,
    GDPHYS_PTR,
    DT: tl.constexpr,
    B: tl.constexpr,
    HGRID: tl.constexpr,
    WGRID: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    STENCIL: tl.constexpr,
    ACT: tl.constexpr,
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    d_block = tl.program_id(2)

    d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d < D
    row = n // WGRID
    col = n - row * WGRID

    n_top = tl.where(row > 0, n - WGRID, n)
    n_bot = tl.where(row < HGRID - 1, n + WGRID, n)
    n_left = tl.where(col > 0, n - 1, n)
    n_right = tl.where(col < WGRID - 1, n + 1, n)
    n_tl = tl.where((row > 0) & (col > 0), n - WGRID - 1, tl.where(row > 0, n - WGRID, tl.where(col > 0, n - 1, n)))
    n_tr = tl.where((row > 0) & (col < WGRID - 1), n - WGRID + 1, tl.where(row > 0, n - WGRID, tl.where(col < WGRID - 1, n + 1, n)))
    n_bl = tl.where((row < HGRID - 1) & (col > 0), n + WGRID - 1, tl.where(row < HGRID - 1, n + WGRID, tl.where(col > 0, n - 1, n)))
    n_br = tl.where((row < HGRID - 1) & (col < WGRID - 1), n + WGRID + 1, tl.where(row < HGRID - 1, n + WGRID, tl.where(col < WGRID - 1, n + 1, n)))

    base = (b * N + n) * D + d
    h = tl.load(H_STATE_PTR + base, mask=mask, other=0.0).to(tl.float32)
    h0 = tl.load(H0_PTR + base, mask=mask, other=0.0).to(tl.float32)
    h_top = tl.load(H_STATE_PTR + (b * N + n_top) * D + d, mask=mask, other=0.0).to(tl.float32)
    h_bot = tl.load(H_STATE_PTR + (b * N + n_bot) * D + d, mask=mask, other=0.0).to(tl.float32)
    h_left = tl.load(H_STATE_PTR + (b * N + n_left) * D + d, mask=mask, other=0.0).to(tl.float32)
    h_right = tl.load(H_STATE_PTR + (b * N + n_right) * D + d, mask=mask, other=0.0).to(tl.float32)
    h_neighbor_sum = h_top + h_bot + h_left + h_right

    g = tl.load(G_PTR + base, mask=mask, other=0.0).to(tl.float32)
    g_top = tl.load(G_PTR + (b * N + n_top) * D + d, mask=mask, other=0.0).to(tl.float32)
    g_bot = tl.load(G_PTR + (b * N + n_bot) * D + d, mask=mask, other=0.0).to(tl.float32)
    g_left = tl.load(G_PTR + (b * N + n_left) * D + d, mask=mask, other=0.0).to(tl.float32)
    g_right = tl.load(G_PTR + (b * N + n_right) * D + d, mask=mask, other=0.0).to(tl.float32)
    pre = tl.load(PREACT_PTR + base, mask=mask, other=0.0).to(tl.float32)
    pre_top = tl.load(PREACT_PTR + (b * N + n_top) * D + d, mask=mask, other=0.0).to(tl.float32)
    pre_bot = tl.load(PREACT_PTR + (b * N + n_bot) * D + d, mask=mask, other=0.0).to(tl.float32)
    pre_left = tl.load(PREACT_PTR + (b * N + n_left) * D + d, mask=mask, other=0.0).to(tl.float32)
    pre_right = tl.load(PREACT_PTR + (b * N + n_right) * D + d, mask=mask, other=0.0).to(tl.float32)

    gp = g * _v12_activation_grad(pre, ACT)
    gp_top = g_top * _v12_activation_grad(pre_top, ACT)
    gp_bot = g_bot * _v12_activation_grad(pre_bot, ACT)
    gp_left = g_left * _v12_activation_grad(pre_left, ACT)
    gp_right = g_right * _v12_activation_grad(pre_right, ACT)

    ds = tl.load(DS_PTR + base, mask=mask, other=0.0).to(tl.float32)
    dd = tl.load(DD_PTR + base, mask=mask, other=0.0).to(tl.float32)
    dd_top = tl.load(DD_PTR + (b * N + n_top) * D + d, mask=mask, other=0.0).to(tl.float32)
    dd_bot = tl.load(DD_PTR + (b * N + n_bot) * D + d, mask=mask, other=0.0).to(tl.float32)
    dd_left = tl.load(DD_PTR + (b * N + n_left) * D + d, mask=mask, other=0.0).to(tl.float32)
    dd_right = tl.load(DD_PTR + (b * N + n_right) * D + d, mask=mask, other=0.0).to(tl.float32)

    a = tl.load(A_PTR + d, mask=mask, other=0.0).to(tl.float32)
    dphys = tl.load(DPHYS_PTR + d, mask=mask, other=0.0).to(tl.float32)
    degree = 4.0

    if STENCIL == 8:
        h_tl = tl.load(H_STATE_PTR + (b * N + n_tl) * D + d, mask=mask, other=0.0).to(tl.float32)
        h_tr = tl.load(H_STATE_PTR + (b * N + n_tr) * D + d, mask=mask, other=0.0).to(tl.float32)
        h_bl = tl.load(H_STATE_PTR + (b * N + n_bl) * D + d, mask=mask, other=0.0).to(tl.float32)
        h_br = tl.load(H_STATE_PTR + (b * N + n_br) * D + d, mask=mask, other=0.0).to(tl.float32)
        h_neighbor_sum += h_tl + h_tr + h_bl + h_br

        g_tl = tl.load(G_PTR + (b * N + n_tl) * D + d, mask=mask, other=0.0).to(tl.float32)
        g_tr = tl.load(G_PTR + (b * N + n_tr) * D + d, mask=mask, other=0.0).to(tl.float32)
        g_bl = tl.load(G_PTR + (b * N + n_bl) * D + d, mask=mask, other=0.0).to(tl.float32)
        g_br = tl.load(G_PTR + (b * N + n_br) * D + d, mask=mask, other=0.0).to(tl.float32)
        pre_tl = tl.load(PREACT_PTR + (b * N + n_tl) * D + d, mask=mask, other=0.0).to(tl.float32)
        pre_tr = tl.load(PREACT_PTR + (b * N + n_tr) * D + d, mask=mask, other=0.0).to(tl.float32)
        pre_bl = tl.load(PREACT_PTR + (b * N + n_bl) * D + d, mask=mask, other=0.0).to(tl.float32)
        pre_br = tl.load(PREACT_PTR + (b * N + n_br) * D + d, mask=mask, other=0.0).to(tl.float32)
        gp_tl = g_tl * _v12_activation_grad(pre_tl, ACT)
        gp_tr = g_tr * _v12_activation_grad(pre_tr, ACT)
        gp_bl = g_bl * _v12_activation_grad(pre_bl, ACT)
        gp_br = g_br * _v12_activation_grad(pre_br, ACT)

        dd_tl = tl.load(DD_PTR + (b * N + n_tl) * D + d, mask=mask, other=0.0).to(tl.float32)
        dd_tr = tl.load(DD_PTR + (b * N + n_tr) * D + d, mask=mask, other=0.0).to(tl.float32)
        dd_bl = tl.load(DD_PTR + (b * N + n_bl) * D + d, mask=mask, other=0.0).to(tl.float32)
        dd_br = tl.load(DD_PTR + (b * N + n_br) * D + d, mask=mask, other=0.0).to(tl.float32)
        degree = 8.0

    lap_h = h_neighbor_sum - degree * h

    r = DT * dd * dphys * gp
    r_sum = DT * dphys * (
        dd_top * gp_top + dd_bot * gp_bot + dd_left * gp_left + dd_right * gp_right
    )
    if STENCIL == 8:
        r_sum += DT * dphys * (dd_tl * gp_tl + dd_tr * gp_tr + dd_bl * gp_bl + dd_br * gp_br)
    lap_r = r_sum - degree * r

    g_prev = gp * (1.0 + DT * ds * a) + lap_r
    tl.store(G_PREV_PTR + base, g_prev, mask=mask)

    old_gh0 = tl.load(GH0_ACC_PTR + base, mask=mask, other=0.0).to(tl.float32)
    tl.store(GH0_ACC_PTR + base, old_gh0 + DT * ds * gp, mask=mask)

    old_gds = tl.load(GDS_PTR + base, mask=mask, other=0.0).to(tl.float32)
    old_gdd = tl.load(GDD_PTR + base, mask=mask, other=0.0).to(tl.float32)
    tl.store(GDS_PTR + base, old_gds + DT * gp * (a * h + h0), mask=mask)
    tl.store(GDD_PTR + base, old_gdd + DT * gp * dphys * lap_h, mask=mask)

    tl.atomic_add(GA_PTR + d, DT * ds * gp * h, sem="relaxed", mask=mask)
    tl.atomic_add(GDPHYS_PTR + d, DT * dd * gp * lap_h, sem="relaxed", mask=mask)


@triton.jit
def _v12_forward_step_kernel(
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
    mask = d < D

    n_top = tl.load(NEIGHBOR_PTR + n * 4 + 0)
    n_bot = tl.load(NEIGHBOR_PTR + n * 4 + 1)
    n_left = tl.load(NEIGHBOR_PTR + n * 4 + 2)
    n_right = tl.load(NEIGHBOR_PTR + n * 4 + 3)

    base = (b * N + n) * D + d
    top_base = (b * N + n_top) * D + d
    bot_base = (b * N + n_bot) * D + d
    left_base = (b * N + n_left) * D + d
    right_base = (b * N + n_right) * D + d

    h = tl.load(H_PTR + base, mask=mask, other=0.0).to(tl.float32)
    h0 = tl.load(H0_PTR + base, mask=mask, other=0.0).to(tl.float32)
    top = tl.load(H_PTR + top_base, mask=mask, other=0.0).to(tl.float32)
    bot = tl.load(H_PTR + bot_base, mask=mask, other=0.0).to(tl.float32)
    left = tl.load(H_PTR + left_base, mask=mask, other=0.0).to(tl.float32)
    right = tl.load(H_PTR + right_base, mask=mask, other=0.0).to(tl.float32)

    ds = tl.load(DS_PTR + base, mask=mask, other=0.0).to(tl.float32)
    dd = tl.load(DD_PTR + base, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(A_PTR + d, mask=mask, other=0.0).to(tl.float32)
    dphys = tl.load(DPHYS_PTR + d, mask=mask, other=0.0).to(tl.float32)

    lap = top + bot + left + right - 4.0 * h
    out = h * (1.0 + DT * ds * a) + DT * ds * h0 + DT * dd * dphys * lap

    tl.store(OUT_PTR + base, out, mask=mask)
    tl.store(SAVE_PTR + STEP * B * N * D + base, h, mask=mask)


@triton.jit
def _v12_backward_step_kernel(
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
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    d_block = tl.program_id(2)

    d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d < D

    n_top = tl.load(NEIGHBOR_PTR + n * 4 + 0)
    n_bot = tl.load(NEIGHBOR_PTR + n * 4 + 1)
    n_left = tl.load(NEIGHBOR_PTR + n * 4 + 2)
    n_right = tl.load(NEIGHBOR_PTR + n * 4 + 3)

    base = (b * N + n) * D + d
    top_base = (b * N + n_top) * D + d
    bot_base = (b * N + n_bot) * D + d
    left_base = (b * N + n_left) * D + d
    right_base = (b * N + n_right) * D + d

    h = tl.load(H_STATE_PTR + base, mask=mask, other=0.0).to(tl.float32)
    h0 = tl.load(H0_PTR + base, mask=mask, other=0.0).to(tl.float32)
    h_top = tl.load(H_STATE_PTR + top_base, mask=mask, other=0.0).to(tl.float32)
    h_bot = tl.load(H_STATE_PTR + bot_base, mask=mask, other=0.0).to(tl.float32)
    h_left = tl.load(H_STATE_PTR + left_base, mask=mask, other=0.0).to(tl.float32)
    h_right = tl.load(H_STATE_PTR + right_base, mask=mask, other=0.0).to(tl.float32)

    g = tl.load(G_PTR + base, mask=mask, other=0.0).to(tl.float32)
    g_top = tl.load(G_PTR + top_base, mask=mask, other=0.0).to(tl.float32)
    g_bot = tl.load(G_PTR + bot_base, mask=mask, other=0.0).to(tl.float32)
    g_left = tl.load(G_PTR + left_base, mask=mask, other=0.0).to(tl.float32)
    g_right = tl.load(G_PTR + right_base, mask=mask, other=0.0).to(tl.float32)

    ds = tl.load(DS_PTR + base, mask=mask, other=0.0).to(tl.float32)
    dd = tl.load(DD_PTR + base, mask=mask, other=0.0).to(tl.float32)
    dd_top = tl.load(DD_PTR + top_base, mask=mask, other=0.0).to(tl.float32)
    dd_bot = tl.load(DD_PTR + bot_base, mask=mask, other=0.0).to(tl.float32)
    dd_left = tl.load(DD_PTR + left_base, mask=mask, other=0.0).to(tl.float32)
    dd_right = tl.load(DD_PTR + right_base, mask=mask, other=0.0).to(tl.float32)

    a = tl.load(A_PTR + d, mask=mask, other=0.0).to(tl.float32)
    dphys = tl.load(DPHYS_PTR + d, mask=mask, other=0.0).to(tl.float32)

    lap_h = h_top + h_bot + h_left + h_right - 4.0 * h
    r = DT * dd * dphys * g
    r_top = DT * dd_top * dphys * g_top
    r_bot = DT * dd_bot * dphys * g_bot
    r_left = DT * dd_left * dphys * g_left
    r_right = DT * dd_right * dphys * g_right
    lap_r = r_top + r_bot + r_left + r_right - 4.0 * r

    g_prev = g * (1.0 + DT * ds * a) + lap_r
    tl.store(G_PREV_PTR + base, g_prev, mask=mask)

    old_gh0 = tl.load(GH0_ACC_PTR + base, mask=mask, other=0.0).to(tl.float32)
    tl.store(GH0_ACC_PTR + base, old_gh0 + DT * ds * g, mask=mask)

    old_gds = tl.load(GDS_PTR + base, mask=mask, other=0.0).to(tl.float32)
    old_gdd = tl.load(GDD_PTR + base, mask=mask, other=0.0).to(tl.float32)
    tl.store(GDS_PTR + base, old_gds + DT * g * (a * h + h0), mask=mask)
    tl.store(GDD_PTR + base, old_gdd + DT * g * dphys * lap_h, mask=mask)

    tl.atomic_add(GA_PTR + d, DT * ds * g * h, sem="relaxed", mask=mask)
    tl.atomic_add(GDPHYS_PTR + d, DT * dd * g * lap_h, sem="relaxed", mask=mask)


@triton.jit
def _add_kernel(A_PTR, B_PTR, OUT_PTR, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    a = tl.load(A_PTR + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_PTR + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT_PTR + offsets, a + b, mask=mask)


def cs_scan_v12_forward_cuda(
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
    block_d: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_triton_cuda(h0, delta_s, delta_d, A, D_phys)
    h0 = h0.contiguous()
    delta_s = delta_s.contiguous()
    delta_d = delta_d.contiguous()
    A = A.contiguous()
    D_phys_flat = D_phys.contiguous().view(-1)

    bsz, n_tokens, d_dim = h0.shape
    if n_tokens != H * W:
        raise ValueError(f"H*W={H * W} does not match N={n_tokens}.")
    if A.numel() != d_dim:
        raise ValueError(f"A has {A.numel()} values, expected {d_dim}.")
    if D_phys_flat.numel() != d_dim:
        raise ValueError(f"D_phys has {D_phys_flat.numel()} values, expected {d_dim}.")
    if neighbor_index is None:
        neighbor_index = grid_neighbor_index(H, W, h0.device)
    else:
        _require_triton_cuda(neighbor_index)
        neighbor_index = neighbor_index.contiguous()
        if tuple(neighbor_index.shape) != (n_tokens, 4):
            raise ValueError(f"neighbor_index shape {tuple(neighbor_index.shape)} must be {(n_tokens, 4)}.")

    h = h0
    work_a = torch.empty_like(h)
    work_b = torch.empty_like(h)
    states = torch.empty((K, bsz, n_tokens, d_dim), device=h.device, dtype=h.dtype)
    grid = (bsz, n_tokens, triton.cdiv(d_dim, block_d))
    dt = 1.0 / float(K)

    for step in range(K):
        h_next = work_a if step % 2 == 0 else work_b
        _v12_forward_step_kernel[grid](
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


def cs_scan_v12_forward_flex_cuda(
    h0: torch.Tensor,
    delta_s: torch.Tensor,
    delta_d: torch.Tensor,
    A: torch.Tensor,
    D_phys: torch.Tensor,
    K: int,
    H: int,
    W: int,
    *,
    stencil: int = 4,
    activation: str = "identity",
    block_d: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_triton_cuda(h0, delta_s, delta_d, A, D_phys)
    if stencil not in {4, 8}:
        raise ValueError(f"Unsupported stencil={stencil}; expected 4 or 8.")
    h0 = h0.contiguous()
    delta_s = delta_s.contiguous()
    delta_d = delta_d.contiguous()
    A = A.contiguous()
    D_phys_flat = D_phys.contiguous().view(-1)

    bsz, n_tokens, d_dim = h0.shape
    if n_tokens != H * W:
        raise ValueError(f"H*W={H * W} does not match N={n_tokens}.")
    if A.numel() != d_dim:
        raise ValueError(f"A has {A.numel()} values, expected {d_dim}.")
    if D_phys_flat.numel() != d_dim:
        raise ValueError(f"D_phys has {D_phys_flat.numel()} values, expected {d_dim}.")

    h = h0
    work_a = torch.empty_like(h)
    work_b = torch.empty_like(h)
    states = torch.empty((K, bsz, n_tokens, d_dim), device=h.device, dtype=h.dtype)
    preacts = torch.empty_like(states)
    grid = (bsz, n_tokens, triton.cdiv(d_dim, block_d))
    dt = 1.0 / float(K)
    act_id = _activation_id(activation)

    for step in range(K):
        h_next = work_a if step % 2 == 0 else work_b
        _v12_forward_step_flex_kernel[grid](
            h,
            h0,
            delta_s,
            delta_d,
            A,
            D_phys_flat,
            h_next,
            states,
            preacts,
            DT=dt,
            STEP=step,
            B=bsz,
            HGRID=int(H),
            WGRID=int(W),
            N=n_tokens,
            D=d_dim,
            BLOCK_D=block_d,
            STENCIL=int(stencil),
            ACT=act_id,
            num_warps=4,
        )
        h = h_next

    return h, states, preacts


def cs_scan_v12_backward_cuda(
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
    block_d: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_triton_cuda(grad_output, states, h0, delta_s, delta_d, A, D_phys)
    grad_output = grad_output.contiguous()
    states = states.contiguous()
    h0 = h0.contiguous()
    delta_s = delta_s.contiguous()
    delta_d = delta_d.contiguous()
    A = A.contiguous()
    D_phys_flat = D_phys.contiguous().view(-1)

    bsz, n_tokens, d_dim = h0.shape
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
    dt = 1.0 / float(K)

    for step in range(K - 1, -1, -1):
        _v12_backward_step_kernel[grid](
            g,
            states[step],
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
            BLOCK_D=block_d,
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
    return grad_h0, grad_delta_s, grad_delta_d, grad_A, grad_D_phys_flat.reshape_as(D_phys)


def cs_scan_v12_backward_flex_cuda(
    grad_output: torch.Tensor,
    states: torch.Tensor,
    preacts: torch.Tensor,
    h0: torch.Tensor,
    delta_s: torch.Tensor,
    delta_d: torch.Tensor,
    A: torch.Tensor,
    D_phys: torch.Tensor,
    K: int,
    H: int,
    W: int,
    *,
    stencil: int = 4,
    activation: str = "identity",
    block_d: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_triton_cuda(grad_output, states, preacts, h0, delta_s, delta_d, A, D_phys)
    if stencil not in {4, 8}:
        raise ValueError(f"Unsupported stencil={stencil}; expected 4 or 8.")
    grad_output = grad_output.contiguous()
    states = states.contiguous()
    preacts = preacts.contiguous()
    h0 = h0.contiguous()
    delta_s = delta_s.contiguous()
    delta_d = delta_d.contiguous()
    A = A.contiguous()
    D_phys_flat = D_phys.contiguous().view(-1)

    bsz, n_tokens, d_dim = h0.shape
    g = grad_output
    g_prev = torch.empty_like(g)
    grad_h0_input = torch.zeros_like(h0)
    grad_delta_s = torch.zeros_like(delta_s)
    grad_delta_d = torch.zeros_like(delta_d)
    grad_A = torch.zeros_like(A)
    grad_D_phys_flat = torch.zeros_like(D_phys_flat)

    grid = (bsz, n_tokens, triton.cdiv(d_dim, block_d))
    dt = 1.0 / float(K)
    act_id = _activation_id(activation)

    for step in range(K - 1, -1, -1):
        _v12_backward_step_flex_kernel[grid](
            g,
            states[step],
            preacts[step],
            h0,
            delta_s,
            delta_d,
            A,
            D_phys_flat,
            g_prev,
            grad_h0_input,
            grad_delta_s,
            grad_delta_d,
            grad_A,
            grad_D_phys_flat,
            DT=dt,
            B=bsz,
            HGRID=int(H),
            WGRID=int(W),
            N=n_tokens,
            D=d_dim,
            BLOCK_D=block_d,
            STENCIL=int(stencil),
            ACT=act_id,
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
    return grad_h0, grad_delta_s, grad_delta_d, grad_A, grad_D_phys_flat.reshape_as(D_phys)


class CSScanV12FlexFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h0, delta_s, delta_d, A, D_phys, K, H, W, stencil, activation):
        h, states, preacts = cs_scan_v12_forward_flex_cuda(
            h0,
            delta_s,
            delta_d,
            A,
            D_phys,
            int(K),
            int(H),
            int(W),
            stencil=int(stencil),
            activation=str(activation),
        )
        ctx.save_for_backward(states, preacts, h0, delta_s, delta_d, A, D_phys)
        ctx.K = int(K)
        ctx.H = int(H)
        ctx.W = int(W)
        ctx.stencil = int(stencil)
        ctx.activation = str(activation)
        return h

    @staticmethod
    def backward(ctx, grad_output):
        states, preacts, h0, delta_s, delta_d, A, D_phys = ctx.saved_tensors
        grad_h0, grad_delta_s, grad_delta_d, grad_A, grad_D_phys = cs_scan_v12_backward_flex_cuda(
            grad_output,
            states,
            preacts,
            h0,
            delta_s,
            delta_d,
            A,
            D_phys,
            ctx.K,
            ctx.H,
            ctx.W,
            stencil=ctx.stencil,
            activation=ctx.activation,
        )
        return grad_h0, grad_delta_s, grad_delta_d, grad_A, grad_D_phys, None, None, None, None, None


class CSScanV12Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h0, delta_s, delta_d, A, D_phys, K, H, W):
        h, states = cs_scan_v12_forward_cuda(h0, delta_s, delta_d, A, D_phys, int(K), int(H), int(W))
        ctx.save_for_backward(states, h0, delta_s, delta_d, A, D_phys)
        ctx.K = int(K)
        ctx.H = int(H)
        ctx.W = int(W)
        return h

    @staticmethod
    def backward(ctx, grad_output):
        states, h0, delta_s, delta_d, A, D_phys = ctx.saved_tensors
        grad_h0, grad_delta_s, grad_delta_d, grad_A, grad_D_phys = cs_scan_v12_backward_cuda(
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
        return grad_h0, grad_delta_s, grad_delta_d, grad_A, grad_D_phys, None, None, None


def cs_scan_v12_cuda(h0, delta_s, delta_d, A, D_phys, K, H, W):
    return CSScanV12Function.apply(h0, delta_s, delta_d, A, D_phys, K, H, W)


def cs_scan_v12_flex_cuda(h0, delta_s, delta_d, A, D_phys, K, H, W, stencil=4, activation="identity"):
    return CSScanV12FlexFunction.apply(h0, delta_s, delta_d, A, D_phys, K, H, W, int(stencil), str(activation))
