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
    "lstm": 2,
    "relu6": 3,
    "relu": 4,
    "gelu": 5,
}


def _activation_id(name: str) -> int:
    try:
        return _ACT_TO_ID[name]
    except KeyError as exc:
        raise ValueError(f"Triton V1.2 scan does not support recurrence_nonlinearity={name!r}") from exc


_INTEGRATOR_TO_ID = {
    "heun": 1,
    "rk4": 2,
    "imex": 3,
}


def _integrator_id(name: str) -> int:
    try:
        return _INTEGRATOR_TO_ID[name]
    except KeyError as exc:
        raise ValueError(f"Triton V1.2 staged scan does not support integrator={name!r}") from exc


def _torch_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    import torch.nn.functional as F

    if activation == "identity":
        return x
    if activation == "silu":
        return F.silu(x)
    if activation in {"tanh", "lstm"}:
        return torch.tanh(x)
    if activation == "gelu":
        return F.gelu(x)
    if activation == "relu6":
        return F.relu6(x)
    if activation == "relu":
        return F.relu(x)
    raise RuntimeError(f"Unsupported activation={activation!r}")


def _torch_laplacian(h_2d: torch.Tensor, stencil: int) -> torch.Tensor:
    import torch.nn.functional as F

    h_pad = F.pad(h_2d, (1, 1, 1, 1), mode="replicate")
    lap = (
        h_pad[:, :, 0:-2, 1:-1]
        + h_pad[:, :, 2:, 1:-1]
        + h_pad[:, :, 1:-1, 0:-2]
        + h_pad[:, :, 1:-1, 2:]
    )
    if stencil == 8:
        lap = (
            lap
            + h_pad[:, :, 0:-2, 0:-2]
            + h_pad[:, :, 0:-2, 2:]
            + h_pad[:, :, 2:, 0:-2]
            + h_pad[:, :, 2:, 2:]
            - 8.0 * h_2d
        )
    else:
        lap = lap - 4.0 * h_2d
    return lap


def _torch_reference_integrator_scan(
    h0: torch.Tensor,
    delta_s: torch.Tensor,
    delta_d: torch.Tensor,
    A: torch.Tensor,
    D_phys: torch.Tensor,
    K: int,
    H: int,
    W: int,
    *,
    stencil: int,
    activation: str,
    integrator: str,
    imex_iters: int,
) -> torch.Tensor:
    import torch.nn.functional as F

    h = h0.transpose(1, 2).unflatten(2, (H, W))
    h0_spatial = h
    delta_s_spatial = delta_s.transpose(1, 2).unflatten(2, (H, W))
    delta_d_spatial = delta_d.transpose(1, 2).unflatten(2, (H, W))
    a = A.view(1, -1, 1, 1)
    dphys = D_phys.contiguous().view(1, -1, 1, 1)
    dt = 1.0 / float(K)

    def rhs(z: torch.Tensor) -> torch.Tensor:
        return delta_s_spatial * (a * z + h0_spatial) + delta_d_spatial * dphys * _torch_laplacian(z, stencil)

    for _ in range(K):
        if integrator == "heun":
            k1 = rhs(h)
            pred = _torch_activation(h + dt * k1, activation)
            k2 = rhs(pred)
            h_next = h + 0.5 * dt * (k1 + k2)
        elif integrator == "rk4":
            k1 = rhs(h)
            k2 = rhs(h + 0.5 * dt * k1)
            k3 = rhs(h + 0.5 * dt * k2)
            k4 = rhs(h + dt * k3)
            h_next = h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        elif integrator == "imex":
            rhs_explicit = h + dt * delta_s_spatial * (a * h + h0_spatial)
            alpha = dt * delta_d_spatial * dphys
            z = rhs_explicit
            degree = 8.0 if stencil == 8 else 4.0
            for _ in range(imex_iters):
                h_pad = F.pad(z, (1, 1, 1, 1), mode="replicate")
                neighbor_sum = (
                    h_pad[:, :, 0:-2, 1:-1]
                    + h_pad[:, :, 2:, 1:-1]
                    + h_pad[:, :, 1:-1, 0:-2]
                    + h_pad[:, :, 1:-1, 2:]
                )
                if stencil == 8:
                    neighbor_sum = (
                        neighbor_sum
                        + h_pad[:, :, 0:-2, 0:-2]
                        + h_pad[:, :, 0:-2, 2:]
                        + h_pad[:, :, 2:, 0:-2]
                        + h_pad[:, :, 2:, 2:]
                    )
                z = (rhs_explicit + alpha * neighbor_sum) / (1.0 + alpha * degree)
            h_next = z
        else:
            raise RuntimeError(f"Unsupported integrator={integrator!r}")
        h = _torch_activation(h_next, activation)
    return h.flatten(2).transpose(1, 2)


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
def _v12_stage_kernel(
    BASE_PTR,
    EVAL_PTR,
    H0_PTR,
    DS_PTR,
    DD_PTR,
    A_PTR,
    DPHYS_PTR,
    OUT_PTR,
    DT_SCALE: tl.constexpr,
    B: tl.constexpr,
    HGRID: tl.constexpr,
    WGRID: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    STENCIL: tl.constexpr,
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
    base_h = tl.load(BASE_PTR + base, mask=mask, other=0.0).to(tl.float32)
    h = tl.load(EVAL_PTR + base, mask=mask, other=0.0).to(tl.float32)
    h0 = tl.load(H0_PTR + base, mask=mask, other=0.0).to(tl.float32)
    neighbor_sum = (
        tl.load(EVAL_PTR + (b * N + n_top) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(EVAL_PTR + (b * N + n_bot) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(EVAL_PTR + (b * N + n_left) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(EVAL_PTR + (b * N + n_right) * D + d, mask=mask, other=0.0).to(tl.float32)
    )
    degree = 4.0
    if STENCIL == 8:
        neighbor_sum += (
            tl.load(EVAL_PTR + (b * N + n_tl) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(EVAL_PTR + (b * N + n_tr) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(EVAL_PTR + (b * N + n_bl) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(EVAL_PTR + (b * N + n_br) * D + d, mask=mask, other=0.0).to(tl.float32)
        )
        degree = 8.0

    ds = tl.load(DS_PTR + base, mask=mask, other=0.0).to(tl.float32)
    dd = tl.load(DD_PTR + base, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(A_PTR + d, mask=mask, other=0.0).to(tl.float32)
    dphys = tl.load(DPHYS_PTR + d, mask=mask, other=0.0).to(tl.float32)

    lap = neighbor_sum - degree * h
    rhs = ds * (a * h + h0) + dd * dphys * lap
    tl.store(OUT_PTR + base, base_h + DT_SCALE * rhs, mask=mask)


@triton.jit
def _v12_heun_finish_kernel(
    H_PTR,
    PRED_PTR,
    H0_PTR,
    DS_PTR,
    DD_PTR,
    A_PTR,
    DPHYS_PTR,
    OUT_PTR,
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
    h = tl.load(H_PTR + base, mask=mask, other=0.0).to(tl.float32)
    pred = tl.load(PRED_PTR + base, mask=mask, other=0.0).to(tl.float32)
    h0 = tl.load(H0_PTR + base, mask=mask, other=0.0).to(tl.float32)

    h_sum = (
        tl.load(H_PTR + (b * N + n_top) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(H_PTR + (b * N + n_bot) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(H_PTR + (b * N + n_left) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(H_PTR + (b * N + n_right) * D + d, mask=mask, other=0.0).to(tl.float32)
    )
    p_sum = (
        tl.load(PRED_PTR + (b * N + n_top) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(PRED_PTR + (b * N + n_bot) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(PRED_PTR + (b * N + n_left) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(PRED_PTR + (b * N + n_right) * D + d, mask=mask, other=0.0).to(tl.float32)
    )
    degree = 4.0
    if STENCIL == 8:
        h_sum += (
            tl.load(H_PTR + (b * N + n_tl) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(H_PTR + (b * N + n_tr) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(H_PTR + (b * N + n_bl) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(H_PTR + (b * N + n_br) * D + d, mask=mask, other=0.0).to(tl.float32)
        )
        p_sum += (
            tl.load(PRED_PTR + (b * N + n_tl) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(PRED_PTR + (b * N + n_tr) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(PRED_PTR + (b * N + n_bl) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(PRED_PTR + (b * N + n_br) * D + d, mask=mask, other=0.0).to(tl.float32)
        )
        degree = 8.0

    ds = tl.load(DS_PTR + base, mask=mask, other=0.0).to(tl.float32)
    dd = tl.load(DD_PTR + base, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(A_PTR + d, mask=mask, other=0.0).to(tl.float32)
    dphys = tl.load(DPHYS_PTR + d, mask=mask, other=0.0).to(tl.float32)
    k1 = ds * (a * h + h0) + dd * dphys * (h_sum - degree * h)
    k2 = ds * (a * pred + h0) + dd * dphys * (p_sum - degree * pred)
    pre = h + 0.5 * DT * (k1 + k2)
    tl.store(OUT_PTR + base, _v12_apply_activation(pre, ACT), mask=mask)


@triton.jit
def _v12_rk4_finish_kernel(
    H_PTR,
    H2_PTR,
    H3_PTR,
    H4_PTR,
    H0_PTR,
    DS_PTR,
    DD_PTR,
    A_PTR,
    DPHYS_PTR,
    OUT_PTR,
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
    h = tl.load(H_PTR + base, mask=mask, other=0.0).to(tl.float32)
    h2 = tl.load(H2_PTR + base, mask=mask, other=0.0).to(tl.float32)
    h3 = tl.load(H3_PTR + base, mask=mask, other=0.0).to(tl.float32)
    h4 = tl.load(H4_PTR + base, mask=mask, other=0.0).to(tl.float32)
    h0 = tl.load(H0_PTR + base, mask=mask, other=0.0).to(tl.float32)

    h_sum = tl.load(H_PTR + (b * N + n_top) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H_PTR + (b * N + n_bot) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H_PTR + (b * N + n_left) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H_PTR + (b * N + n_right) * D + d, mask=mask, other=0.0).to(tl.float32)
    h2_sum = tl.load(H2_PTR + (b * N + n_top) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H2_PTR + (b * N + n_bot) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H2_PTR + (b * N + n_left) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H2_PTR + (b * N + n_right) * D + d, mask=mask, other=0.0).to(tl.float32)
    h3_sum = tl.load(H3_PTR + (b * N + n_top) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H3_PTR + (b * N + n_bot) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H3_PTR + (b * N + n_left) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H3_PTR + (b * N + n_right) * D + d, mask=mask, other=0.0).to(tl.float32)
    h4_sum = tl.load(H4_PTR + (b * N + n_top) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H4_PTR + (b * N + n_bot) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H4_PTR + (b * N + n_left) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H4_PTR + (b * N + n_right) * D + d, mask=mask, other=0.0).to(tl.float32)
    degree = 4.0
    if STENCIL == 8:
        h_sum += tl.load(H_PTR + (b * N + n_tl) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H_PTR + (b * N + n_tr) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H_PTR + (b * N + n_bl) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H_PTR + (b * N + n_br) * D + d, mask=mask, other=0.0).to(tl.float32)
        h2_sum += tl.load(H2_PTR + (b * N + n_tl) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H2_PTR + (b * N + n_tr) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H2_PTR + (b * N + n_bl) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H2_PTR + (b * N + n_br) * D + d, mask=mask, other=0.0).to(tl.float32)
        h3_sum += tl.load(H3_PTR + (b * N + n_tl) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H3_PTR + (b * N + n_tr) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H3_PTR + (b * N + n_bl) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H3_PTR + (b * N + n_br) * D + d, mask=mask, other=0.0).to(tl.float32)
        h4_sum += tl.load(H4_PTR + (b * N + n_tl) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H4_PTR + (b * N + n_tr) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H4_PTR + (b * N + n_bl) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H4_PTR + (b * N + n_br) * D + d, mask=mask, other=0.0).to(tl.float32)
        degree = 8.0

    ds = tl.load(DS_PTR + base, mask=mask, other=0.0).to(tl.float32)
    dd = tl.load(DD_PTR + base, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(A_PTR + d, mask=mask, other=0.0).to(tl.float32)
    dphys = tl.load(DPHYS_PTR + d, mask=mask, other=0.0).to(tl.float32)
    k1 = ds * (a * h + h0) + dd * dphys * (h_sum - degree * h)
    k2 = ds * (a * h2 + h0) + dd * dphys * (h2_sum - degree * h2)
    k3 = ds * (a * h3 + h0) + dd * dphys * (h3_sum - degree * h3)
    k4 = ds * (a * h4 + h0) + dd * dphys * (h4_sum - degree * h4)
    pre = h + (DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    tl.store(OUT_PTR + base, _v12_apply_activation(pre, ACT), mask=mask)


@triton.jit
def _v12_rk4_finish_preact_kernel(
    H_PTR,
    H2_PTR,
    H3_PTR,
    H4_PTR,
    H0_PTR,
    DS_PTR,
    DD_PTR,
    A_PTR,
    DPHYS_PTR,
    OUT_PTR,
    DT: tl.constexpr,
    B: tl.constexpr,
    HGRID: tl.constexpr,
    WGRID: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    STENCIL: tl.constexpr,
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
    h2 = tl.load(H2_PTR + base, mask=mask, other=0.0).to(tl.float32)
    h3 = tl.load(H3_PTR + base, mask=mask, other=0.0).to(tl.float32)
    h4 = tl.load(H4_PTR + base, mask=mask, other=0.0).to(tl.float32)
    h0 = tl.load(H0_PTR + base, mask=mask, other=0.0).to(tl.float32)

    h_sum = tl.load(H_PTR + (b * N + n_top) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H_PTR + (b * N + n_bot) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H_PTR + (b * N + n_left) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H_PTR + (b * N + n_right) * D + d, mask=mask, other=0.0).to(tl.float32)
    h2_sum = tl.load(H2_PTR + (b * N + n_top) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H2_PTR + (b * N + n_bot) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H2_PTR + (b * N + n_left) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H2_PTR + (b * N + n_right) * D + d, mask=mask, other=0.0).to(tl.float32)
    h3_sum = tl.load(H3_PTR + (b * N + n_top) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H3_PTR + (b * N + n_bot) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H3_PTR + (b * N + n_left) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H3_PTR + (b * N + n_right) * D + d, mask=mask, other=0.0).to(tl.float32)
    h4_sum = tl.load(H4_PTR + (b * N + n_top) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H4_PTR + (b * N + n_bot) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H4_PTR + (b * N + n_left) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H4_PTR + (b * N + n_right) * D + d, mask=mask, other=0.0).to(tl.float32)
    degree = 4.0
    if STENCIL == 8:
        h_sum += tl.load(H_PTR + (b * N + n_tl) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H_PTR + (b * N + n_tr) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H_PTR + (b * N + n_bl) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H_PTR + (b * N + n_br) * D + d, mask=mask, other=0.0).to(tl.float32)
        h2_sum += tl.load(H2_PTR + (b * N + n_tl) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H2_PTR + (b * N + n_tr) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H2_PTR + (b * N + n_bl) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H2_PTR + (b * N + n_br) * D + d, mask=mask, other=0.0).to(tl.float32)
        h3_sum += tl.load(H3_PTR + (b * N + n_tl) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H3_PTR + (b * N + n_tr) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H3_PTR + (b * N + n_bl) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H3_PTR + (b * N + n_br) * D + d, mask=mask, other=0.0).to(tl.float32)
        h4_sum += tl.load(H4_PTR + (b * N + n_tl) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H4_PTR + (b * N + n_tr) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H4_PTR + (b * N + n_bl) * D + d, mask=mask, other=0.0).to(tl.float32) + tl.load(H4_PTR + (b * N + n_br) * D + d, mask=mask, other=0.0).to(tl.float32)
        degree = 8.0

    ds = tl.load(DS_PTR + base, mask=mask, other=0.0).to(tl.float32)
    dd = tl.load(DD_PTR + base, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(A_PTR + d, mask=mask, other=0.0).to(tl.float32)
    dphys = tl.load(DPHYS_PTR + d, mask=mask, other=0.0).to(tl.float32)
    k1 = ds * (a * h + h0) + dd * dphys * (h_sum - degree * h)
    k2 = ds * (a * h2 + h0) + dd * dphys * (h2_sum - degree * h2)
    k3 = ds * (a * h3 + h0) + dd * dphys * (h3_sum - degree * h3)
    k4 = ds * (a * h4 + h0) + dd * dphys * (h4_sum - degree * h4)
    pre = h + (DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    tl.store(OUT_PTR + base, pre, mask=mask)


@triton.jit
def _v12_imex_rhs_kernel(
    H_PTR,
    H0_PTR,
    DS_PTR,
    A_PTR,
    OUT_PTR,
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
    base = (b * N + n) * D + d
    h = tl.load(H_PTR + base, mask=mask, other=0.0).to(tl.float32)
    h0 = tl.load(H0_PTR + base, mask=mask, other=0.0).to(tl.float32)
    ds = tl.load(DS_PTR + base, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(A_PTR + d, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT_PTR + base, h + DT * ds * (a * h + h0), mask=mask)


@triton.jit
def _v12_imex_jacobi_kernel(
    RHS_PTR,
    Z_PTR,
    DD_PTR,
    DPHYS_PTR,
    OUT_PTR,
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
    rhs = tl.load(RHS_PTR + base, mask=mask, other=0.0).to(tl.float32)
    neighbor_sum = (
        tl.load(Z_PTR + (b * N + n_top) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(Z_PTR + (b * N + n_bot) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(Z_PTR + (b * N + n_left) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(Z_PTR + (b * N + n_right) * D + d, mask=mask, other=0.0).to(tl.float32)
    )
    degree = 4.0
    if STENCIL == 8:
        neighbor_sum += (
            tl.load(Z_PTR + (b * N + n_tl) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(Z_PTR + (b * N + n_tr) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(Z_PTR + (b * N + n_bl) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(Z_PTR + (b * N + n_br) * D + d, mask=mask, other=0.0).to(tl.float32)
        )
        degree = 8.0
    dd = tl.load(DD_PTR + base, mask=mask, other=0.0).to(tl.float32)
    dphys = tl.load(DPHYS_PTR + d, mask=mask, other=0.0).to(tl.float32)
    alpha = DT * dd * dphys
    out = (rhs + alpha * neighbor_sum) / (1.0 + alpha * degree)
    tl.store(OUT_PTR + base, _v12_apply_activation(out, ACT), mask=mask)


@triton.jit
def _v12_imex_jacobi_bwd_elem_kernel(
    GZ_PTR,          # Input gradient (B, N, D)
    Z_PREV_PTR,      # Forward state z^(j) (B, N, D)
    Z_CURR_PTR,      # Forward state z^(j+1) (B, N, D)
    DD_PTR,          # delta_d (B, N, D)
    DPHYS_PTR,       # D_phys (D,)
    Q_PTR,           # Intermediate scaled output gradient (B, N, D)
    GRHS_ACC_PTR,    # Accumulator for rhs gradient (B, N, D)
    GDELTA_D_PTR,    # Gradient accumulator for delta_d (B, N, D)
    GDPHYS_PTR,      # Gradient accumulator for D_phys (D,)
    DT: tl.constexpr,
    B: tl.constexpr,
    HGRID: tl.constexpr,
    WGRID: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    STENCIL: tl.constexpr,
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

    base = (b * N + n) * D + d

    gz = tl.load(GZ_PTR + base, mask=mask, other=0.0).to(tl.float32)
    z_prev = tl.load(Z_PREV_PTR + base, mask=mask, other=0.0).to(tl.float32)
    z_curr = tl.load(Z_CURR_PTR + base, mask=mask, other=0.0).to(tl.float32)

    dd = tl.load(DD_PTR + base, mask=mask, other=0.0).to(tl.float32)
    dphys = tl.load(DPHYS_PTR + d, mask=mask, other=0.0).to(tl.float32)
    alpha = DT * dd * dphys

    degree = 4.0
    z_prev_sum = (
        tl.load(Z_PREV_PTR + (b * N + n_top) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(Z_PREV_PTR + (b * N + n_bot) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(Z_PREV_PTR + (b * N + n_left) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(Z_PREV_PTR + (b * N + n_right) * D + d, mask=mask, other=0.0).to(tl.float32)
    )

    if STENCIL == 8:
        n_tl = tl.where((row > 0) & (col > 0), n - WGRID - 1, tl.where(row > 0, n - WGRID, tl.where(col > 0, n - 1, n)))
        n_tr = tl.where((row > 0) & (col < WGRID - 1), n - WGRID + 1, tl.where(row > 0, n - WGRID, tl.where(col < WGRID - 1, n + 1, n)))
        n_bl = tl.where((row < HGRID - 1) & (col > 0), n + WGRID - 1, tl.where(row < HGRID - 1, n + WGRID, tl.where(col > 0, n - 1, n)))
        n_br = tl.where((row < HGRID - 1) & (col < WGRID - 1), n + WGRID + 1, tl.where(row < HGRID - 1, n + WGRID, tl.where(col < WGRID - 1, n + 1, n)))
        z_prev_sum += (
            tl.load(Z_PREV_PTR + (b * N + n_tl) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(Z_PREV_PTR + (b * N + n_tr) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(Z_PREV_PTR + (b * N + n_bl) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(Z_PREV_PTR + (b * N + n_br) * D + d, mask=mask, other=0.0).to(tl.float32)
        )
        degree = 8.0

    denom = 1.0 + alpha * degree
    
    # 1. Scale input gradient for spatial propagation: q = (alpha * gz) / (1 + alpha * degree)
    q = (alpha * gz) / denom
    tl.store(Q_PTR + base, q, mask=mask)

    # 2. Accumulate to RHS gradient: grhs += gz / (1 + alpha * degree)
    old_grhs = tl.load(GRHS_ACC_PTR + base, mask=mask, other=0.0).to(tl.float32)
    tl.store(GRHS_ACC_PTR + base, old_grhs + (gz / denom), mask=mask)

    # 3. Parameter gradients
    g_alpha = gz * (z_prev_sum - degree * z_curr) / denom

    old_gdd = tl.load(GDELTA_D_PTR + base, mask=mask, other=0.0).to(tl.float32)
    tl.store(GDELTA_D_PTR + base, old_gdd + g_alpha * DT * dphys, mask=mask)

    tl.atomic_add(GDPHYS_PTR + d, g_alpha * DT * dd, sem="relaxed", mask=mask)


@triton.jit
def _v12_imex_jacobi_bwd_spatial_kernel(
    Q_PTR,           # Scaled input gradient (B, N, D)
    GZ_NEXT_PTR,     # Output gradient for next iter (B, N, D)
    B: tl.constexpr,
    HGRID: tl.constexpr,
    WGRID: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    STENCIL: tl.constexpr,
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

    base = (b * N + n) * D + d

    # Adjoint spatial neighbor sum of q
    q_sum = (
        tl.load(Q_PTR + (b * N + n_top) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(Q_PTR + (b * N + n_bot) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(Q_PTR + (b * N + n_left) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(Q_PTR + (b * N + n_right) * D + d, mask=mask, other=0.0).to(tl.float32)
    )

    if STENCIL == 8:
        n_tl = tl.where((row > 0) & (col > 0), n - WGRID - 1, tl.where(row > 0, n - WGRID, tl.where(col > 0, n - 1, n)))
        n_tr = tl.where((row > 0) & (col < WGRID - 1), n - WGRID + 1, tl.where(row > 0, n - WGRID, tl.where(col < WGRID - 1, n + 1, n)))
        n_bl = tl.where((row < HGRID - 1) & (col > 0), n + WGRID - 1, tl.where(row < HGRID - 1, n + WGRID, tl.where(col > 0, n - 1, n)))
        n_br = tl.where((row < HGRID - 1) & (col < WGRID - 1), n + WGRID + 1, tl.where(row < HGRID - 1, n + WGRID, tl.where(col < WGRID - 1, n + 1, n)))
        q_sum += (
            tl.load(Q_PTR + (b * N + n_tl) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(Q_PTR + (b * N + n_tr) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(Q_PTR + (b * N + n_bl) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(Q_PTR + (b * N + n_br) * D + d, mask=mask, other=0.0).to(tl.float32)
        )

    tl.store(GZ_NEXT_PTR + base, q_sum, mask=mask)


@triton.jit
def _v12_imex_rhs_bwd_kernel(
    GRHS_PTR,        # Gradient with respect to rhs (B, N, D)
    H_PTR,           # Hidden state input to step h^(k) (B, N, D)
    H0_PTR,          # input projection h0 (B, N, D)
    DS_PTR,          # delta_s (B, N, D)
    A_PTR,           # A (D,)
    G_PREV_PTR,      # Output gradient for input state h^(k) (B, N, D)
    GDS_PTR,         # Gradient accumulator for delta_s (B, N, D)
    GH0_PTR,         # Gradient accumulator for h0 (B, N, D)
    GA_PTR,          # Gradient accumulator for A (D,)
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

    base = (b * N + n) * D + d

    grhs = tl.load(GRHS_PTR + base, mask=mask, other=0.0).to(tl.float32)
    h = tl.load(H_PTR + base, mask=mask, other=0.0).to(tl.float32)
    h0 = tl.load(H0_PTR + base, mask=mask, other=0.0).to(tl.float32)
    ds = tl.load(DS_PTR + base, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(A_PTR + d, mask=mask, other=0.0).to(tl.float32)

    g_prev = grhs * (1.0 + DT * ds * a)
    tl.store(G_PREV_PTR + base, g_prev, mask=mask)

    old_gh0 = tl.load(GH0_PTR + base, mask=mask, other=0.0).to(tl.float32)
    tl.store(GH0_PTR + base, old_gh0 + grhs * DT * ds, mask=mask)

    old_gds = tl.load(GDS_PTR + base, mask=mask, other=0.0).to(tl.float32)
    tl.store(GDS_PTR + base, old_gds + grhs * DT * (a * h + h0), mask=mask)

    tl.atomic_add(GA_PTR + d, grhs * DT * ds * h, sem="relaxed", mask=mask)


@triton.jit
def _v12_imex_act_bwd_kernel(
    G_PTR,          # incoming gradient (B, N, D)
    PREACT_PTR,     # preacts (B, N, D)
    G_PREACT_PTR,   # Output pre-activation gradient (B, N, D)
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ACT: tl.constexpr,
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    d_block = tl.program_id(2)
    d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d < D

    base = (b * N + n) * D + d
    g = tl.load(G_PTR + base, mask=mask, other=0.0).to(tl.float32)
    pre = tl.load(PREACT_PTR + base, mask=mask, other=0.0).to(tl.float32)

    g_pre = g * _v12_activation_grad(pre, ACT)
    tl.store(G_PREACT_PTR + base, g_pre, mask=mask)


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
def _v12_stage_adjoint_kernel(
    G_PTR, H_STATE_PTR, H0_PTR, DS_PTR, DD_PTR, A_PTR, DPHYS_PTR,
    G_PREV_PTR, GH0_ACC_PTR, GDS_PTR, GDD_PTR, GA_PTR, GDPHYS_PTR,
    DT: tl.constexpr, B: tl.constexpr, HGRID: tl.constexpr, WGRID: tl.constexpr,
    N: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr, STENCIL: tl.constexpr,
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

    base = (b * N + n) * D + d
    h = tl.load(H_STATE_PTR + base, mask=mask, other=0.0).to(tl.float32)
    h0 = tl.load(H0_PTR + base, mask=mask, other=0.0).to(tl.float32)
    ds = tl.load(DS_PTR + base, mask=mask, other=0.0).to(tl.float32)
    dd = tl.load(DD_PTR + base, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(A_PTR + d, mask=mask, other=0.0).to(tl.float32)
    dphys = tl.load(DPHYS_PTR + d, mask=mask, other=0.0).to(tl.float32)

    h_sum = (
        tl.load(H_STATE_PTR + (b * N + n_top) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(H_STATE_PTR + (b * N + n_bot) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(H_STATE_PTR + (b * N + n_left) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(H_STATE_PTR + (b * N + n_right) * D + d, mask=mask, other=0.0).to(tl.float32)
    )

    gp = tl.load(G_PTR + base, mask=mask, other=0.0).to(tl.float32)

    r = dd * gp
    r_sum = (
        tl.load(DD_PTR + (b * N + n_top) * D + d, mask=mask, other=0.0).to(tl.float32) * tl.load(G_PTR + (b * N + n_top) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(DD_PTR + (b * N + n_bot) * D + d, mask=mask, other=0.0).to(tl.float32) * tl.load(G_PTR + (b * N + n_bot) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(DD_PTR + (b * N + n_left) * D + d, mask=mask, other=0.0).to(tl.float32) * tl.load(G_PTR + (b * N + n_left) * D + d, mask=mask, other=0.0).to(tl.float32)
        + tl.load(DD_PTR + (b * N + n_right) * D + d, mask=mask, other=0.0).to(tl.float32) * tl.load(G_PTR + (b * N + n_right) * D + d, mask=mask, other=0.0).to(tl.float32)
    )

    degree = 4.0
    if STENCIL == 8:
        n_tl = tl.where((row > 0) & (col > 0), n - WGRID - 1, tl.where(row > 0, n - WGRID, tl.where(col > 0, n - 1, n)))
        n_tr = tl.where((row > 0) & (col < WGRID - 1), n - WGRID + 1, tl.where(row > 0, n - WGRID, tl.where(col < WGRID - 1, n + 1, n)))
        n_bl = tl.where((row < HGRID - 1) & (col > 0), n + WGRID - 1, tl.where(row < HGRID - 1, n + WGRID, tl.where(col > 0, n - 1, n)))
        n_br = tl.where((row < HGRID - 1) & (col < WGRID - 1), n + WGRID + 1, tl.where(row < HGRID - 1, n + WGRID, tl.where(col < WGRID - 1, n + 1, n)))

        h_sum += (
            tl.load(H_STATE_PTR + (b * N + n_tl) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(H_STATE_PTR + (b * N + n_tr) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(H_STATE_PTR + (b * N + n_bl) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(H_STATE_PTR + (b * N + n_br) * D + d, mask=mask, other=0.0).to(tl.float32)
        )

        r_sum += (
            tl.load(DD_PTR + (b * N + n_tl) * D + d, mask=mask, other=0.0).to(tl.float32) * tl.load(G_PTR + (b * N + n_tl) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(DD_PTR + (b * N + n_tr) * D + d, mask=mask, other=0.0).to(tl.float32) * tl.load(G_PTR + (b * N + n_tr) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(DD_PTR + (b * N + n_bl) * D + d, mask=mask, other=0.0).to(tl.float32) * tl.load(G_PTR + (b * N + n_bl) * D + d, mask=mask, other=0.0).to(tl.float32)
            + tl.load(DD_PTR + (b * N + n_br) * D + d, mask=mask, other=0.0).to(tl.float32) * tl.load(G_PTR + (b * N + n_br) * D + d, mask=mask, other=0.0).to(tl.float32)
        )
        degree = 8.0

    lap_h = h_sum - degree * h
    lap_r = dphys * (r_sum - degree * r)

    g_prev = gp * ds * a + lap_r
    old_g = tl.load(G_PREV_PTR + base, mask=mask, other=0.0).to(tl.float32)
    tl.store(G_PREV_PTR + base, old_g + DT * g_prev, mask=mask)

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


def cs_scan_v12_forward_integrator_cuda(
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
    integrator: str = "heun",
    imex_iters: int = 3,
    block_d: int = 128,
) -> torch.Tensor:
    _require_triton_cuda(h0, delta_s, delta_d, A, D_phys)
    if stencil not in {4, 8}:
        raise ValueError(f"Unsupported stencil={stencil}; expected 4 or 8.")
    _activation_id(activation)
    _integrator_id(integrator)

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
    grid = (bsz, n_tokens, triton.cdiv(d_dim, block_d))
    dt = 1.0 / float(K)
    act_id = _activation_id(activation)

    for _ in range(K):
        if integrator == "heun":
            pred = torch.empty_like(h)
            out = torch.empty_like(h)
            _v12_stage_kernel[grid](
                h,
                h,
                h0,
                delta_s,
                delta_d,
                A,
                D_phys_flat,
                pred,
                DT_SCALE=dt,
                B=bsz,
                HGRID=int(H),
                WGRID=int(W),
                N=n_tokens,
                D=d_dim,
                BLOCK_D=block_d,
                STENCIL=int(stencil),
                num_warps=4,
            )
            # Apply activation to predictor (matching PyTorch model's Heun behavior)
            pred = _torch_activation(pred, activation)
            _v12_heun_finish_kernel[grid](
                h,
                pred,
                h0,
                delta_s,
                delta_d,
                A,
                D_phys_flat,
                out,
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
            h = out
        elif integrator == "rk4":
            h2 = torch.empty_like(h)
            h3 = torch.empty_like(h)
            h4 = torch.empty_like(h)
            out = torch.empty_like(h)
            _v12_stage_kernel[grid](
                h,
                h,
                h0,
                delta_s,
                delta_d,
                A,
                D_phys_flat,
                h2,
                DT_SCALE=0.5 * dt,
                B=bsz,
                HGRID=int(H),
                WGRID=int(W),
                N=n_tokens,
                D=d_dim,
                BLOCK_D=block_d,
                STENCIL=int(stencil),
                num_warps=4,
            )
            _v12_stage_kernel[grid](
                h,
                h2,
                h0,
                delta_s,
                delta_d,
                A,
                D_phys_flat,
                h3,
                DT_SCALE=0.5 * dt,
                B=bsz,
                HGRID=int(H),
                WGRID=int(W),
                N=n_tokens,
                D=d_dim,
                BLOCK_D=block_d,
                STENCIL=int(stencil),
                num_warps=4,
            )
            _v12_stage_kernel[grid](
                h,
                h3,
                h0,
                delta_s,
                delta_d,
                A,
                D_phys_flat,
                h4,
                DT_SCALE=dt,
                B=bsz,
                HGRID=int(H),
                WGRID=int(W),
                N=n_tokens,
                D=d_dim,
                BLOCK_D=block_d,
                STENCIL=int(stencil),
                num_warps=4,
            )
            _v12_rk4_finish_kernel[grid](
                h,
                h2,
                h3,
                h4,
                h0,
                delta_s,
                delta_d,
                A,
                D_phys_flat,
                out,
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
            h = out
        elif integrator == "imex":
            rhs = torch.empty_like(h)
            z = torch.empty_like(h)
            z_next = torch.empty_like(h)
            _v12_imex_rhs_kernel[grid](
                h,
                h0,
                delta_s,
                A,
                rhs,
                DT=dt,
                B=bsz,
                N=n_tokens,
                D=d_dim,
                BLOCK_D=block_d,
                num_warps=4,
            )
            z.copy_(rhs)
            n_iters = max(1, int(imex_iters))
            for jacobi_iter in range(n_iters):
                _v12_imex_jacobi_kernel[grid](
                    rhs,
                    z,
                    delta_d,
                    D_phys_flat,
                    z_next,
                    DT=dt,
                    B=bsz,
                    HGRID=int(H),
                    WGRID=int(W),
                    N=n_tokens,
                    D=d_dim,
                    BLOCK_D=block_d,
                    STENCIL=int(stencil),
                    ACT=act_id if jacobi_iter == n_iters - 1 else 0,
                    num_warps=4,
                )
                z, z_next = z_next, z
            h = z
        else:
            raise RuntimeError(f"Unsupported integrator={integrator!r}")

    return h


def cs_scan_v12_forward_imex_cuda(
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
    imex_iters: int = 3,
    block_d: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    states = torch.empty((K, bsz, n_tokens, d_dim), device=h.device, dtype=h.dtype)
    preacts = torch.empty_like(states)
    jacobi_states = torch.empty((K, imex_iters + 1, bsz, n_tokens, d_dim), device=h.device, dtype=h.dtype)

    grid = (bsz, n_tokens, triton.cdiv(d_dim, block_d))
    dt = 1.0 / float(K)
    act_id = _activation_id(activation)

    for step in range(K):
        rhs = jacobi_states[step, 0]
        _v12_imex_rhs_kernel[grid](
            h,
            h0,
            delta_s,
            A,
            rhs,
            DT=dt,
            B=bsz,
            N=n_tokens,
            D=d_dim,
            BLOCK_D=block_d,
            num_warps=4,
        )

        z = rhs
        n_iters = max(1, int(imex_iters))
        for jacobi_iter in range(n_iters):
            z_next = jacobi_states[step, jacobi_iter + 1]
            _v12_imex_jacobi_kernel[grid](
                rhs,
                z,
                delta_d,
                D_phys_flat,
                z_next,
                DT=dt,
                B=bsz,
                HGRID=int(H),
                WGRID=int(W),
                N=n_tokens,
                D=d_dim,
                BLOCK_D=block_d,
                STENCIL=int(stencil),
                ACT=0,
                num_warps=4,
            )
            z = z_next

        preacts[step].copy_(z)
        h_next = states[step]
        h_next.copy_(_torch_activation(z, activation))
        h = h_next

    return h, states, preacts, jacobi_states


def cs_scan_v12_backward_imex_cuda(
    grad_output: torch.Tensor,
    states: torch.Tensor,
    preacts: torch.Tensor,
    jacobi_states: torch.Tensor,
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
    imex_iters: int = 3,
    block_d: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_triton_cuda(grad_output, states, preacts, jacobi_states, h0, delta_s, delta_d, A, D_phys)
    grad_output = grad_output.contiguous()
    states = states.contiguous()
    preacts = preacts.contiguous()
    jacobi_states = jacobi_states.contiguous()
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
        g_pre = torch.empty_like(g)
        _v12_imex_act_bwd_kernel[grid](
            g,
            preacts[step],
            g_pre,
            B=bsz,
            N=n_tokens,
            D=d_dim,
            BLOCK_D=block_d,
            ACT=act_id,
            num_warps=4,
        )

        gz = g_pre
        rhs_grad = torch.zeros_like(gz)
        q = torch.empty_like(gz)
        for jacobi_iter in range(imex_iters - 1, -1, -1):
            gz_next = torch.empty_like(gz)
            _v12_imex_jacobi_bwd_elem_kernel[grid](
                gz,
                jacobi_states[step, jacobi_iter],
                jacobi_states[step, jacobi_iter + 1],
                delta_d,
                D_phys_flat,
                q,
                rhs_grad,
                grad_delta_d,
                grad_D_phys_flat,
                DT=dt,
                B=bsz,
                HGRID=int(H),
                WGRID=int(W),
                N=n_tokens,
                D=d_dim,
                BLOCK_D=block_d,
                STENCIL=int(stencil),
                num_warps=4,
            )
            _v12_imex_jacobi_bwd_spatial_kernel[grid](
                q,
                gz_next,
                B=bsz,
                HGRID=int(H),
                WGRID=int(W),
                N=n_tokens,
                D=d_dim,
                BLOCK_D=block_d,
                STENCIL=int(stencil),
                num_warps=4,
            )
            gz = gz_next

        rhs_grad.add_(gz)

        h_in = states[step - 1] if step > 0 else h0
        _v12_imex_rhs_bwd_kernel[grid](
            rhs_grad,
            h_in,
            h0,
            delta_s,
            A,
            g_prev,
            grad_delta_s,
            grad_h0_input,
            grad_A,
            DT=dt,
            B=bsz,
            N=n_tokens,
            D=d_dim,
            BLOCK_D=block_d,
            num_warps=4,
        )
        g = g_prev

    grad_h0 = grad_h0_input + g
    return grad_h0, grad_delta_s, grad_delta_d, grad_A, grad_D_phys_flat.reshape_as(D_phys)


class CSScanV12IMEXFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h0, delta_s, delta_d, A, D_phys, K, H, W, stencil, activation, imex_iters):
        h, states, preacts, jacobi_states = cs_scan_v12_forward_imex_cuda(
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
            imex_iters=int(imex_iters),
        )
        ctx.save_for_backward(states, preacts, jacobi_states, h0, delta_s, delta_d, A, D_phys)
        ctx.K = int(K)
        ctx.H = int(H)
        ctx.W = int(W)
        ctx.stencil = int(stencil)
        ctx.activation = str(activation)
        ctx.imex_iters = int(imex_iters)
        return h

    @staticmethod
    def backward(ctx, grad_output):
        states, preacts, jacobi_states, h0, delta_s, delta_d, A, D_phys = ctx.saved_tensors
        grad_h0, grad_delta_s, grad_delta_d, grad_A, grad_D_phys = cs_scan_v12_backward_imex_cuda(
            grad_output,
            states,
            preacts,
            jacobi_states,
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
            imex_iters=ctx.imex_iters,
        )
        return grad_h0, grad_delta_s, grad_delta_d, grad_A, grad_D_phys, None, None, None, None, None, None


def cs_scan_v12_imex_cuda(
    h0,
    delta_s,
    delta_d,
    A,
    D_phys,
    K,
    H,
    W,
    stencil=4,
    activation="identity",
    imex_iters=3,
):
    return CSScanV12IMEXFunction.apply(
        h0,
        delta_s,
        delta_d,
        A,
        D_phys,
        K,
        H,
        W,
        int(stencil),
        str(activation),
        int(imex_iters),
    )


class CSScanV12IntegratorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h0, delta_s, delta_d, A, D_phys, K, H, W, stencil, activation, integrator, imex_iters):
        h = cs_scan_v12_forward_integrator_cuda(
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
            integrator=str(integrator),
            imex_iters=int(imex_iters),
        )
        ctx.save_for_backward(h0, delta_s, delta_d, A, D_phys)
        ctx.K = int(K)
        ctx.H = int(H)
        ctx.W = int(W)
        ctx.stencil = int(stencil)
        ctx.activation = str(activation)
        ctx.integrator = str(integrator)
        ctx.imex_iters = int(imex_iters)
        return h

    @staticmethod
    def backward(ctx, grad_output):
        h0, delta_s, delta_d, A, D_phys = ctx.saved_tensors
        with torch.enable_grad():
            h0_r = h0.detach().requires_grad_(True)
            delta_s_r = delta_s.detach().requires_grad_(True)
            delta_d_r = delta_d.detach().requires_grad_(True)
            A_r = A.detach().requires_grad_(True)
            D_phys_r = D_phys.detach().requires_grad_(True)
            out = _torch_reference_integrator_scan(
                h0_r,
                delta_s_r,
                delta_d_r,
                A_r,
                D_phys_r,
                ctx.K,
                ctx.H,
                ctx.W,
                stencil=ctx.stencil,
                activation=ctx.activation,
                integrator=ctx.integrator,
                imex_iters=ctx.imex_iters,
            )
            grads = torch.autograd.grad(
                out,
                (h0_r, delta_s_r, delta_d_r, A_r, D_phys_r),
                grad_output,
                allow_unused=False,
            )
        return grads[0], grads[1], grads[2], grads[3], grads[4], None, None, None, None, None, None, None


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


def cs_scan_v12_integrator_cuda(
    h0,
    delta_s,
    delta_d,
    A,
    D_phys,
    K,
    H,
    W,
    stencil=4,
    activation="identity",
    integrator="heun",
    imex_iters=3,
):
    if integrator == "rk4":
        return CSScanV12RK4Function.apply(
            h0,
            delta_s,
            delta_d,
            A,
            D_phys,
            K,
            H,
            W,
            int(stencil),
            str(activation),
        )
    elif integrator == "heun":
        return CSScanV12HeunFunction.apply(
            h0,
            delta_s,
            delta_d,
            A,
            D_phys,
            K,
            H,
            W,
            int(stencil),
            str(activation),
        )
    return CSScanV12IntegratorFunction.apply(
        h0,
        delta_s,
        delta_d,
        A,
        D_phys,
        K,
        H,
        W,
        int(stencil),
        str(activation),
        str(integrator),
        int(imex_iters),
    )


class CSScanV12RK4Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h0, delta_s, delta_d, A, D_phys, K, H, W, stencil, activation):
        h, rk4_states = cs_scan_v12_forward_rk4_cuda(
            h0, delta_s, delta_d, A, D_phys, int(K), int(H), int(W),
            stencil=int(stencil), activation=str(activation)
        )
        ctx.save_for_backward(rk4_states, h0, delta_s, delta_d, A, D_phys)
        ctx.K = int(K)
        ctx.H = int(H)
        ctx.W = int(W)
        ctx.stencil = int(stencil)
        ctx.activation = str(activation)
        return h

    @staticmethod
    def backward(ctx, grad_output):
        rk4_states, h0, delta_s, delta_d, A, D_phys = ctx.saved_tensors
        grad_h0, grad_delta_s, grad_delta_d, grad_A, grad_D_phys = cs_scan_v12_backward_rk4_cuda(
            grad_output,
            rk4_states,
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


def cs_scan_v12_forward_rk4_cuda(
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
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_triton_cuda(h0, delta_s, delta_d, A, D_phys)
    h0 = h0.contiguous()
    delta_s = delta_s.contiguous()
    delta_d = delta_d.contiguous()
    A = A.contiguous()
    D_phys_flat = D_phys.contiguous().view(-1)

    bsz, n_tokens, d_dim = h0.shape
    h = h0
    
    rk4_states = torch.empty((K, 5, bsz, n_tokens, d_dim), device=h.device, dtype=h.dtype)
    grid = (bsz, n_tokens, triton.cdiv(d_dim, block_d))
    dt = 1.0 / float(K)

    for step in range(K):
        rk4_states[step, 0].copy_(h)
        h2 = rk4_states[step, 1]
        h3 = rk4_states[step, 2]
        h4 = rk4_states[step, 3]
        
        _v12_stage_kernel[grid](
            h, h, h0, delta_s, delta_d, A, D_phys_flat, h2,
            DT_SCALE=0.5 * dt, B=bsz, HGRID=int(H), WGRID=int(W), N=n_tokens, D=d_dim, BLOCK_D=block_d, STENCIL=int(stencil)
        )
        
        _v12_stage_kernel[grid](
            h, h2, h0, delta_s, delta_d, A, D_phys_flat, h3,
            DT_SCALE=0.5 * dt, B=bsz, HGRID=int(H), WGRID=int(W), N=n_tokens, D=d_dim, BLOCK_D=block_d, STENCIL=int(stencil)
        )
        
        _v12_stage_kernel[grid](
            h, h3, h0, delta_s, delta_d, A, D_phys_flat, h4,
            DT_SCALE=dt, B=bsz, HGRID=int(H), WGRID=int(W), N=n_tokens, D=d_dim, BLOCK_D=block_d, STENCIL=int(stencil)
        )
        
        preact = rk4_states[step, 4]
        _v12_rk4_finish_preact_kernel[grid](
            h, h2, h3, h4, h0, delta_s, delta_d, A, D_phys_flat, preact,
            DT=dt, B=bsz, HGRID=int(H), WGRID=int(W), N=n_tokens, D=d_dim, BLOCK_D=block_d, STENCIL=int(stencil)
        )
        
        h = _torch_activation(preact, activation)
        
    return h, rk4_states


def cs_scan_v12_backward_rk4_cuda(
    grad_output: torch.Tensor,
    rk4_states: torch.Tensor,
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
    _require_triton_cuda(grad_output, rk4_states, h0, delta_s, delta_d, A, D_phys)
    grad_output = grad_output.contiguous()
    rk4_states = rk4_states.contiguous()
    h0 = h0.contiguous()
    delta_s = delta_s.contiguous()
    delta_d = delta_d.contiguous()
    A = A.contiguous()
    D_phys_flat = D_phys.contiguous().view(-1)

    bsz, n_tokens, d_dim = h0.shape
    g = grad_output

    grad_h0 = torch.zeros_like(h0)
    grad_delta_s = torch.zeros_like(delta_s)
    grad_delta_d = torch.zeros_like(delta_d)
    grad_A = torch.zeros_like(A)
    grad_D_phys_flat = torch.zeros_like(D_phys_flat)

    grid = (bsz, n_tokens, triton.cdiv(d_dim, block_d))
    dt = 1.0 / float(K)
    act_id = _activation_id(activation)

    for step in range(K - 1, -1, -1):
        pre = rk4_states[step, 4]
        g_pre = torch.empty_like(g)
        _v12_imex_act_bwd_kernel[grid](
            g, pre, g_pre,
            B=bsz, N=n_tokens, D=d_dim, BLOCK_D=block_d, ACT=act_id, num_warps=4
        )

        h = rk4_states[step, 0]
        h2 = rk4_states[step, 1]
        h3 = rk4_states[step, 2]
        h4 = rk4_states[step, 3]

        g_stage4 = (dt / 6.0) * g_pre
        g_h4 = torch.zeros_like(g)
        _v12_stage_adjoint_kernel[grid](
            g_stage4, h4, h0, delta_s, delta_d, A, D_phys_flat,
            g_h4, grad_h0, grad_delta_s, grad_delta_d, grad_A, grad_D_phys_flat,
            DT=1.0, B=bsz, HGRID=int(H), WGRID=int(W), N=n_tokens, D=d_dim, BLOCK_D=block_d, STENCIL=int(stencil), num_warps=4
        )

        g_stage3 = (dt / 3.0) * g_pre + dt * g_h4
        g_h3 = torch.zeros_like(g)
        _v12_stage_adjoint_kernel[grid](
            g_stage3, h3, h0, delta_s, delta_d, A, D_phys_flat,
            g_h3, grad_h0, grad_delta_s, grad_delta_d, grad_A, grad_D_phys_flat,
            DT=1.0, B=bsz, HGRID=int(H), WGRID=int(W), N=n_tokens, D=d_dim, BLOCK_D=block_d, STENCIL=int(stencil), num_warps=4
        )

        g_stage2 = (dt / 3.0) * g_pre + 0.5 * dt * g_h3
        g_h2 = torch.zeros_like(g)
        _v12_stage_adjoint_kernel[grid](
            g_stage2, h2, h0, delta_s, delta_d, A, D_phys_flat,
            g_h2, grad_h0, grad_delta_s, grad_delta_d, grad_A, grad_D_phys_flat,
            DT=1.0, B=bsz, HGRID=int(H), WGRID=int(W), N=n_tokens, D=d_dim, BLOCK_D=block_d, STENCIL=int(stencil), num_warps=4
        )

        g_stage1 = (dt / 6.0) * g_pre + 0.5 * dt * g_h2
        g_h1 = torch.zeros_like(g)
        _v12_stage_adjoint_kernel[grid](
            g_stage1, h, h0, delta_s, delta_d, A, D_phys_flat,
            g_h1, grad_h0, grad_delta_s, grad_delta_d, grad_A, grad_D_phys_flat,
            DT=1.0, B=bsz, HGRID=int(H), WGRID=int(W), N=n_tokens, D=d_dim, BLOCK_D=block_d, STENCIL=int(stencil), num_warps=4
        )

        g = g_pre + g_h4 + g_h3 + g_h2 + g_h1

    grad_h0 = grad_h0 + g

    return grad_h0, grad_delta_s, grad_delta_d, grad_A, grad_D_phys_flat.view_as(D_phys)


class CSScanV12HeunFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h0, delta_s, delta_d, A, D_phys, K, H, W, stencil, activation):
        h, heun_states = cs_scan_v12_forward_heun_cuda(
            h0, delta_s, delta_d, A, D_phys, int(K), int(H), int(W),
            stencil=int(stencil), activation=str(activation)
        )
        ctx.save_for_backward(heun_states, h0, delta_s, delta_d, A, D_phys)
        ctx.K = int(K)
        ctx.H = int(H)
        ctx.W = int(W)
        ctx.stencil = int(stencil)
        ctx.activation = str(activation)
        return h

    @staticmethod
    def backward(ctx, grad_output):
        heun_states, h0, delta_s, delta_d, A, D_phys = ctx.saved_tensors
        grad_h0, grad_delta_s, grad_delta_d, grad_A, grad_D_phys = cs_scan_v12_backward_heun_cuda(
            grad_output,
            heun_states,
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


def cs_scan_v12_forward_heun_cuda(
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
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_triton_cuda(h0, delta_s, delta_d, A, D_phys)
    h0 = h0.contiguous()
    delta_s = delta_s.contiguous()
    delta_d = delta_d.contiguous()
    A = A.contiguous()
    D_phys_flat = D_phys.contiguous().view(-1)

    bsz, n_tokens, d_dim = h0.shape
    h = h0
    
    heun_states = torch.empty((K, 3, bsz, n_tokens, d_dim), device=h.device, dtype=h.dtype)
    grid = (bsz, n_tokens, triton.cdiv(d_dim, block_d))
    dt = 1.0 / float(K)

    for step in range(K):
        heun_states[step, 0].copy_(h)
        h_2_pre = heun_states[step, 1]
        
        _v12_stage_kernel[grid](
            h, h, h0, delta_s, delta_d, A, D_phys_flat, h_2_pre,
            DT_SCALE=dt, B=bsz, HGRID=int(H), WGRID=int(W), N=n_tokens, D=d_dim, BLOCK_D=block_d, STENCIL=int(stencil)
        )
        
        pred_activated = _torch_activation(h_2_pre, activation)
        
        preact = heun_states[step, 2]
        _v12_heun_finish_kernel[grid](
            h, pred_activated, h0, delta_s, delta_d, A, D_phys_flat, preact,
            DT=dt, B=bsz, HGRID=int(H), WGRID=int(W), N=n_tokens, D=d_dim, BLOCK_D=block_d, STENCIL=int(stencil),
            ACT=0,
            num_warps=4,
        )
        h = _torch_activation(preact, activation)
        
    return h, heun_states


def cs_scan_v12_backward_heun_cuda(
    grad_output: torch.Tensor,
    heun_states: torch.Tensor,
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
    _require_triton_cuda(grad_output, heun_states, h0, delta_s, delta_d, A, D_phys)
    grad_output = grad_output.contiguous()
    heun_states = heun_states.contiguous()
    h0 = h0.contiguous()
    delta_s = delta_s.contiguous()
    delta_d = delta_d.contiguous()
    A = A.contiguous()
    D_phys_flat = D_phys.contiguous().view(-1)

    bsz, n_tokens, d_dim = h0.shape
    g = grad_output

    grad_h0 = torch.zeros_like(h0)
    grad_delta_s = torch.zeros_like(delta_s)
    grad_delta_d = torch.zeros_like(delta_d)
    grad_A = torch.zeros_like(A)
    grad_D_phys_flat = torch.zeros_like(D_phys_flat)

    grid = (bsz, n_tokens, triton.cdiv(d_dim, block_d))
    dt = 1.0 / float(K)
    act_id = _activation_id(activation)

    for step in range(K - 1, -1, -1):
        h = heun_states[step, 0]
        h_2_pre = heun_states[step, 1]
        preact = heun_states[step, 2]

        g_pre = torch.empty_like(g)
        _v12_imex_act_bwd_kernel[grid](
            g, preact, g_pre,
            B=bsz, N=n_tokens, D=d_dim, BLOCK_D=block_d, ACT=act_id, num_warps=4
        )

        g_k2 = 0.5 * dt * g_pre
        pred_activated = _torch_activation(h_2_pre, activation)
        g_h2 = torch.zeros_like(g)
        _v12_stage_adjoint_kernel[grid](
            g_k2, pred_activated, h0, delta_s, delta_d, A, D_phys_flat,
            g_h2, grad_h0, grad_delta_s, grad_delta_d, grad_A, grad_D_phys_flat,
            DT=1.0, B=bsz, HGRID=int(H), WGRID=int(W), N=n_tokens, D=d_dim, BLOCK_D=block_d, STENCIL=int(stencil), num_warps=4
        )

        g_h2_pre = torch.empty_like(g)
        _v12_imex_act_bwd_kernel[grid](
            g_h2, h_2_pre, g_h2_pre,
            B=bsz, N=n_tokens, D=d_dim, BLOCK_D=block_d, ACT=act_id, num_warps=4
        )

        g_k1 = 0.5 * dt * g_pre + dt * g_h2_pre
        g_h1 = torch.zeros_like(g)
        _v12_stage_adjoint_kernel[grid](
            g_k1, h, h0, delta_s, delta_d, A, D_phys_flat,
            g_h1, grad_h0, grad_delta_s, grad_delta_d, grad_A, grad_D_phys_flat,
            DT=1.0, B=bsz, HGRID=int(H), WGRID=int(W), N=n_tokens, D=d_dim, BLOCK_D=block_d, STENCIL=int(stencil), num_warps=4
        )

        g = g_pre + g_h2_pre + g_h1

    grad_h0 = grad_h0 + g

    return grad_h0, grad_delta_s, grad_delta_d, grad_A, grad_D_phys_flat.view_as(D_phys)
