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
    def __init__(self, d_model: int, expand: int = 2, spatial_op: str = "laplacian"):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        d_inner = int(expand * d_model)
        if spatial_op not in {"laplacian", "conv2d", "conv1d"}:
            raise ValueError(f"Unsupported spatial_op={spatial_op!r}")
        self.spatial_op = spatial_op

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
        if self.spatial_op == "conv2d":
            return self.spatial_conv2d(F.pad(h_2d, (1, 1, 1, 1), mode="replicate"))

        bsz, d_dim, height, width = h_2d.shape
        h_1d = h_2d.flatten(2)
        mixed = self.spatial_conv1d(F.pad(h_1d, (1, 1), mode="replicate"))
        return mixed.view(bsz, d_dim, height, width)

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

        if use_triton and x.is_cuda and self.spatial_op == "laplacian":
            from triton_kernels.csma_triton_scan_v12 import cs_scan_v12_cuda
            h = cs_scan_v12_cuda(h0, delta_self, delta_diff, A, D_phys, K_steps, H, W)
        else:
            h = h0.transpose(1, 2).unflatten(2, (H, W))
            h0_spatial = h
            delta_s_spatial = delta_self.transpose(1, 2).unflatten(2, (H, W))
            delta_d_spatial = delta_diff.transpose(1, 2).unflatten(2, (H, W))

            dt = 1.0 / K_steps
            self_coeff = 1.0 + dt * delta_s_spatial * A.view(1, d_dim, 1, 1)
            input_term = dt * delta_s_spatial * h0_spatial
            diff_coeff = dt * delta_d_spatial * D_phys.view(1, d_dim, 1, 1)

            for _ in range(K_steps):
                h = h * self_coeff + input_term + diff_coeff * self._spatial_mix(h)

            h = h.flatten(2).transpose(1, 2)

        y = h * torch.sigmoid(self.C_proj(x))
        return y + x * self.D


class ContinuousSpatialMambaBlock_V12(nn.Module):
    def __init__(self, d_model: int, expand: int = 2, spatial_op: str = "laplacian"):
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
        self.continuous_ssm = ContinuousSpatialSSM_V12(d_model=d_model, expand=expand, spatial_op=spatial_op)
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
        self.blocks = nn.ModuleList([
            ContinuousSpatialMambaBlock_V12(cfg.d_embed, spatial_op=self.spatial_op)
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
