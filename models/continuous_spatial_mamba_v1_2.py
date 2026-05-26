"""CS-Mamba V1.2: 3D-state continuous spatial recurrence.

This keeps the V1 idea (dual continuous-time gates + learnable decay +
thermodynamic 2D diffusion) but removes the extra state axis S. The recurrent
state is directly `(B, N, D_inner)`, closer to VMamba-style token tensors while
preserving the local PDE recurrence instead of directional scans.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
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


def laplacian_2d_neumann8(h_2d: torch.Tensor) -> torch.Tensor:
    h_pad = F.pad(h_2d, (1, 1, 1, 1), mode="replicate")
    return (
        h_pad[:, :, 0:-2, 1:-1]
        + h_pad[:, :, 2:, 1:-1]
        + h_pad[:, :, 1:-1, 0:-2]
        + h_pad[:, :, 1:-1, 2:]
        + h_pad[:, :, 0:-2, 0:-2]
        + h_pad[:, :, 0:-2, 2:]
        + h_pad[:, :, 2:, 0:-2]
        + h_pad[:, :, 2:, 2:]
        - 8.0 * h_2d
    )


def cs_mamba_v12_forward_reference(h0, delta_s, delta_d, A, D_phys, K, H, W):
    bsz, n_tokens, d_dim = h0.shape
    assert H * W == n_tokens, f"H*W={H * W} != N={n_tokens}"

    dt = 1.0 / K
    h = h0.transpose(1, 2).unflatten(2, (H, W))
    h0_spatial = h
    delta_s_spatial = delta_s.transpose(1, 2).unflatten(2, (H, W))
    delta_d_spatial = delta_d.transpose(1, 2).unflatten(2, (H, W))

    self_coeff = 1.0 + dt * delta_s_spatial * A.view(1, d_dim, 1, 1)
    input_term = dt * delta_s_spatial * h0_spatial
    diff_coeff = dt * delta_d_spatial * D_phys.view(1, d_dim, 1, 1)

    for _ in range(K):
        h = h * self_coeff + input_term + diff_coeff * laplacian_2d_neumann(h)

    return h.flatten(2).transpose(1, 2)


class ContinuousSpatialSSM_V12(nn.Module):
    def __init__(
        self,
        d_model: int,
        expand: int = 2,
        spatial_op: str = "laplacian",
        recurrence_nonlinearity: str = "identity",
        integrator: str = "euler",
        imex_iters: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        d_inner = int(expand * d_model)
        if spatial_op not in {"laplacian", "laplacian8", "conv2d", "conv1d"}:
            raise ValueError(f"Unsupported spatial_op={spatial_op!r}")
        if recurrence_nonlinearity not in {"identity", "silu", "tanh", "gelu", "relu6", "relu"}:
            raise ValueError(f"Unsupported recurrence_nonlinearity={recurrence_nonlinearity!r}")
        if integrator not in {"euler", "heun", "rk4", "imex"}:
            raise ValueError(f"Unsupported integrator={integrator!r}")
        self.spatial_op = spatial_op
        self.recurrence_nonlinearity = recurrence_nonlinearity
        self.integrator = integrator
        self.imex_iters = int(imex_iters)

        self.dt_self_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.dt_diff_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.B_proj = nn.Linear(d_inner, d_inner, bias=False)
        self.C_proj = nn.Linear(d_inner, d_inner, bias=False)
        self.D = nn.Parameter(torch.ones(d_inner))

        dt_init = math.log(math.exp(0.1) - 1.0)
        nn.init.constant_(self.dt_self_proj.bias, dt_init)
        nn.init.constant_(self.dt_diff_proj.bias, dt_init)
        nn.init.uniform_(self.dt_self_proj.weight, -1e-4, 1e-4)
        nn.init.uniform_(self.dt_diff_proj.weight, -1e-4, 1e-4)

        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_inner + 1, dtype=torch.float32)))
        self.diffusivity_raw = nn.Parameter(torch.zeros(1, d_inner, 1, 1))
        if spatial_op == "conv2d":
            self.spatial_conv2d = nn.Conv2d(
                d_inner,
                d_inner,
                kernel_size=3,
                padding=0,
                groups=d_inner,
                bias=False,
            )
            self._init_conv2d_as_laplacian()
        elif spatial_op == "conv1d":
            self.spatial_conv1d = nn.Conv1d(
                d_inner,
                d_inner,
                kernel_size=3,
                padding=0,
                groups=d_inner,
                bias=False,
            )
            self._init_conv1d_as_second_difference()

    def _init_conv2d_as_laplacian(self) -> None:
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=self.spatial_conv2d.weight.dtype,
            device=self.spatial_conv2d.weight.device,
        )
        with torch.no_grad():
            self.spatial_conv2d.weight.copy_(
                kernel.view(1, 1, 3, 3).repeat(self.spatial_conv2d.weight.shape[0], 1, 1, 1)
            )

    def _init_conv1d_as_second_difference(self) -> None:
        kernel = torch.tensor(
            [1.0, -2.0, 1.0],
            dtype=self.spatial_conv1d.weight.dtype,
            device=self.spatial_conv1d.weight.device,
        )
        with torch.no_grad():
            self.spatial_conv1d.weight.copy_(
                kernel.view(1, 1, 3).repeat(self.spatial_conv1d.weight.shape[0], 1, 1)
            )

    def _spatial_mix(self, h_2d: torch.Tensor) -> torch.Tensor:
        if self.spatial_op == "laplacian":
            return laplacian_2d_neumann(h_2d)
        if self.spatial_op == "laplacian8":
            return laplacian_2d_neumann8(h_2d)
        if self.spatial_op == "conv2d":
            return self.spatial_conv2d(F.pad(h_2d, (1, 1, 1, 1), mode="replicate"))

        bsz, d_dim, height, width = h_2d.shape
        h_1d = h_2d.flatten(2)
        mixed = self.spatial_conv1d(F.pad(h_1d, (1, 1), mode="replicate"))
        return mixed.view(bsz, d_dim, height, width)

    def _neighbor_sum(self, h_2d: torch.Tensor) -> tuple[torch.Tensor, int]:
        h_pad = F.pad(h_2d, (1, 1, 1, 1), mode="replicate")
        axial = (
            h_pad[:, :, 0:-2, 1:-1]
            + h_pad[:, :, 2:, 1:-1]
            + h_pad[:, :, 1:-1, 0:-2]
            + h_pad[:, :, 1:-1, 2:]
        )
        if self.spatial_op == "laplacian":
            return axial, 4
        if self.spatial_op == "laplacian8":
            diagonal = (
                h_pad[:, :, 0:-2, 0:-2]
                + h_pad[:, :, 0:-2, 2:]
                + h_pad[:, :, 2:, 0:-2]
                + h_pad[:, :, 2:, 2:]
            )
            return axial + diagonal, 8
        raise RuntimeError("IMEX integrator is only defined for laplacian and laplacian8 spatial operators.")

    def _recurrence_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.recurrence_nonlinearity == "identity":
            return x
        if self.recurrence_nonlinearity == "silu":
            return F.silu(x)
        if self.recurrence_nonlinearity == "tanh":
            return torch.tanh(x)
        if self.recurrence_nonlinearity == "gelu":
            return F.gelu(x)
        if self.recurrence_nonlinearity == "relu6":
            return F.relu6(x)
        if self.recurrence_nonlinearity == "relu":
            return F.relu(x)
        raise RuntimeError(f"Unsupported recurrence_nonlinearity={self.recurrence_nonlinearity!r}")

    def _rhs(
        self,
        h: torch.Tensor,
        delta_s_spatial: torch.Tensor,
        delta_d_spatial: torch.Tensor,
        A: torch.Tensor,
        h0_spatial: torch.Tensor,
        D_phys: torch.Tensor,
    ) -> torch.Tensor:
        return (
            delta_s_spatial * (A.view(1, -1, 1, 1) * h + h0_spatial)
            + delta_d_spatial * D_phys.view(1, -1, 1, 1) * self._spatial_mix(h)
        )

    def _imex_step(
        self,
        h: torch.Tensor,
        dt: float,
        delta_s_spatial: torch.Tensor,
        delta_d_spatial: torch.Tensor,
        A: torch.Tensor,
        h0_spatial: torch.Tensor,
        D_phys: torch.Tensor,
    ) -> torch.Tensor:
        if self.spatial_op not in {"laplacian", "laplacian8"}:
            raise RuntimeError("IMEX integrator is only supported for laplacian and laplacian8.")

        rhs = h + dt * delta_s_spatial * (A.view(1, -1, 1, 1) * h + h0_spatial)
        alpha = dt * delta_d_spatial * D_phys.view(1, -1, 1, 1)
        z = rhs
        for _ in range(self.imex_iters):
            neighbor_sum, degree = self._neighbor_sum(z)
            z = (rhs + alpha * neighbor_sum) / (1.0 + alpha * float(degree))
        return z

    def forward(self, x: torch.Tensor, K_steps: int = 3, use_triton: bool = False) -> torch.Tensor:
        bsz, n_tokens, d_dim = x.shape
        del bsz
        H = W = int(math.sqrt(n_tokens))
        assert H * W == n_tokens, "CS-Mamba V1.2 requires a square token grid."

        A = -F.softplus(self.A_log)
        delta_self = torch.clamp(F.softplus(self.dt_self_proj(x)), max=0.15)
        delta_diff = torch.clamp(F.softplus(self.dt_diff_proj(x)), max=0.15)

        h0 = x * torch.tanh(self.B_proj(x))
        D_phys = torch.sigmoid(self.diffusivity_raw) * 0.5

        use_fast_triton = (
            use_triton
            and x.is_cuda
            and self.spatial_op in {"laplacian", "laplacian8"}
            and self.recurrence_nonlinearity == "identity"
            and self.integrator == "euler"
        )
        if use_fast_triton:
            if self.spatial_op == "laplacian":
                from triton_kernels.csma_triton_scan_v12 import cs_scan_v12_cuda
                h = cs_scan_v12_cuda(h0, delta_self, delta_diff, A, D_phys, K_steps, H, W)
            else:
                from triton_kernels.csma_triton_scan_v12 import cs_scan_v12_flex_cuda
                h = cs_scan_v12_flex_cuda(
                    h0,
                    delta_self,
                    delta_diff,
                    A,
                    D_phys,
                    K_steps,
                    H,
                    W,
                    stencil=8,
                    activation=self.recurrence_nonlinearity,
                )
        else:
            h = h0.transpose(1, 2).unflatten(2, (H, W))
            h0_spatial = h
            delta_s_spatial = delta_self.transpose(1, 2).unflatten(2, (H, W))
            delta_d_spatial = delta_diff.transpose(1, 2).unflatten(2, (H, W))

            dt = 1.0 / K_steps

            for _ in range(K_steps):
                if self.integrator == "euler":
                    h_next = h + dt * self._rhs(h, delta_s_spatial, delta_d_spatial, A, h0_spatial, D_phys)
                elif self.integrator == "heun":
                    k1 = self._rhs(h, delta_s_spatial, delta_d_spatial, A, h0_spatial, D_phys)
                    h_pred = self._recurrence_activation(h + dt * k1)
                    k2 = self._rhs(h_pred, delta_s_spatial, delta_d_spatial, A, h0_spatial, D_phys)
                    h_next = h + 0.5 * dt * (k1 + k2)
                elif self.integrator == "rk4":
                    k1 = self._rhs(h, delta_s_spatial, delta_d_spatial, A, h0_spatial, D_phys)
                    k2 = self._rhs(h + 0.5 * dt * k1, delta_s_spatial, delta_d_spatial, A, h0_spatial, D_phys)
                    k3 = self._rhs(h + 0.5 * dt * k2, delta_s_spatial, delta_d_spatial, A, h0_spatial, D_phys)
                    k4 = self._rhs(h + dt * k3, delta_s_spatial, delta_d_spatial, A, h0_spatial, D_phys)
                    h_next = h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                elif self.integrator == "imex":
                    h_next = self._imex_step(h, dt, delta_s_spatial, delta_d_spatial, A, h0_spatial, D_phys)
                else:
                    raise RuntimeError(f"Unsupported integrator={self.integrator!r}")
                h = self._recurrence_activation(h_next)

            h = h.flatten(2).transpose(1, 2)

        y = h * torch.sigmoid(self.C_proj(x))
        return y + x * self.D


class ContinuousSpatialMambaBlock_V12(nn.Module):
    def __init__(
        self,
        d_model: int,
        expand: int = 2,
        spatial_op: str = "laplacian",
        recurrence_nonlinearity: str = "identity",
        integrator: str = "euler",
        imex_iters: int = 3,
    ):
        super().__init__()
        d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.local_conv2d = nn.Conv2d(
            d_inner,
            d_inner,
            kernel_size=3,
            padding=1,
            groups=d_inner,
            bias=True,
        )
        self.activation = nn.SiLU()
        self.continuous_ssm = ContinuousSpatialSSM_V12(
            d_model=d_model,
            expand=expand,
            spatial_op=spatial_op,
            recurrence_nonlinearity=recurrence_nonlinearity,
            integrator=integrator,
            imex_iters=imex_iters,
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor, K_steps: int = 3, use_triton: bool = False) -> torch.Tensor:
        residual = x
        bsz, n_tokens, _ = x.shape
        H = W = int(math.sqrt(n_tokens))
        assert H * W == n_tokens, "CS-Mamba V1.2 requires a square token grid."

        u, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        u_2d = u.transpose(1, 2).reshape(bsz, -1, H, W)
        u_2d = self.local_conv2d(u_2d)
        u = u_2d.flatten(2).transpose(1, 2)
        u = self.activation(u)

        if self.training and u.device.type not in ("xla",):
            from torch.utils.checkpoint import checkpoint
            y_ssm = checkpoint(self.continuous_ssm, u, K_steps, use_triton, use_reentrant=False)
        else:
            y_ssm = self.continuous_ssm(u, K_steps=K_steps, use_triton=use_triton)

        y = self.out_proj(y_ssm * F.silu(z))
        return residual + y


class ContinuousSpatialMambaClassifier_V12(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from models.patch_encoder import PatchEmbedding

        self.embedder = PatchEmbedding(
            img_size=getattr(cfg, "canvas_size", getattr(cfg, "img_size", 32)),
            patch_size=cfg.patch_size,
            in_channels=3,
            d_embed=cfg.d_embed,
        )
        self.spatial_op = getattr(cfg, "spatial_op", "laplacian")
        self.recurrence_nonlinearity = getattr(cfg, "recurrence_nonlinearity", "identity")
        self.integrator = getattr(cfg, "integrator", "euler")
        self.imex_iters = getattr(cfg, "imex_iters", 3)
        self.blocks = nn.ModuleList([
            ContinuousSpatialMambaBlock_V12(
                cfg.d_embed,
                spatial_op=self.spatial_op,
                recurrence_nonlinearity=self.recurrence_nonlinearity,
                integrator=self.integrator,
                imex_iters=self.imex_iters,
            )
            for _ in range(cfg.n_mamba_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_embed)
        self.head = nn.Linear(cfg.d_embed, getattr(cfg, "n_classes", 10))
        self.K_steps = getattr(cfg, "K_steps", 3)

    def forward(self, x, use_triton: bool = False):
        x = self.embedder(x)
        for block in self.blocks:
            x = block(x, K_steps=self.K_steps, use_triton=use_triton)
        return self.head(self.norm(x).mean(dim=1))
