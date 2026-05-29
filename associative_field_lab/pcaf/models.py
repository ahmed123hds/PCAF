from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .triton_bucket import query_bucket_indices
except Exception:  # pragma: no cover
    query_bucket_indices = None


HASH_MOD = 2_147_483_647


def sinusoidal_positions(max_len: int, d_model: int) -> torch.Tensor:
    positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(positions * div)
    pe[:, 1::2] = torch.cos(positions * div)
    return pe


def causal_ngram_hash(tokens: torch.Tensor, order: int) -> torch.Tensor:
    """Hash the token context ending at each position.

    For order=3, position i hashes tokens [i-2, i-1, i], with zeros used before
    the beginning of the sequence. Equal local contexts produce equal hashes.
    """

    if order <= 0:
        raise ValueError("order must be positive")
    h = torch.zeros_like(tokens, dtype=torch.long)
    for shift in range(order - 1, -1, -1):
        shifted = torch.zeros_like(tokens, dtype=torch.long)
        if shift == 0:
            shifted = tokens.long()
        else:
            shifted[:, shift:] = tokens[:, :-shift].long()
        h = torch.remainder(h * 1_000_003 + shifted + 97, HASH_MOD)
    return h


class CausalConvBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            groups=d_model,
            bias=True,
        )
        self.pointwise = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = F.pad(y.transpose(1, 2), (self.kernel_size - 1, 0))
        y = self.depthwise(y).transpose(1, 2)
        y = self.pointwise(y)
        return x + self.dropout(y)


class SparseAssociativeField(nn.Module):
    """Parallel Causal Associative Field prototype.

    For a sequence x_0 ... x_T, the layer constructs records:
        key = x_i, value = x_{i+1}

    The final token is treated as a query. Retrieval can be full over all
    records, exact oracle over matching raw token IDs, or hash-restricted over
    a small content bucket. The learned part is the resolver over candidates.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        num_classes: int,
        d_model: int = 128,
        d_hidden: int = 256,
        num_buckets: int = 4096,
        top_k: int = 8,
        candidate_mode: str = "hash",
        dropout: float = 0.1,
        use_global: bool = True,
    ) -> None:
        super().__init__()
        if candidate_mode not in {
            "full",
            "hash",
            "oracle",
            "triton_hash",
            "semantic_hash",
            "hybrid_semantic_hash",
        }:
            raise ValueError(
                "candidate_mode must be one of: full, hash, oracle, triton_hash, "
                "semantic_hash, hybrid_semantic_hash"
            )

        self.num_buckets = num_buckets
        self.top_k = top_k
        self.candidate_mode = candidate_mode
        self.use_global = use_global
        self.scale = d_model**-0.5

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.global_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        feature_dim = d_model * (3 if use_global else 2)
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, num_classes),
        )

    def token_hash(self, tokens: torch.Tensor) -> torch.Tensor:
        # Large odd constants give a stable integer hash without Python loops.
        return torch.remainder(tokens.long() * 1_000_003 + 97_531, self.num_buckets)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, seq_len]")
        if tokens.size(1) < 3:
            raise ValueError("sequence length must be at least 3")

        x = self.token_emb(tokens)

        record_key_tokens = tokens[:, :-1]
        query_tokens = tokens[:, -1]

        record_keys = F.normalize(self.key_proj(x[:, :-1]), dim=-1)
        record_values = self.value_proj(x[:, 1:])
        query = F.normalize(self.query_proj(x[:, -1]), dim=-1)

        if (
            self.candidate_mode == "triton_hash"
            and query_bucket_indices is not None
            and tokens.is_cuda
            and self.top_k > 0
        ):
            candidate_idx = query_bucket_indices(
                record_key_tokens,
                query_tokens,
                num_buckets=self.num_buckets,
                max_bucket_size=self.top_k,
            )
            context = self._resolve_from_indices(
                query, record_keys, record_values, candidate_idx
            )
        else:
            scores = torch.einsum("bnd,bd->bn", record_keys, query) * self.scale
            candidate_mask = self._candidate_mask(record_key_tokens, query_tokens)
            context = self._resolve(scores, record_values, candidate_mask)

        if self.use_global:
            global_context = self.global_proj(x[:, :-1].mean(dim=1))
            features = torch.cat([x[:, -1], context, global_context], dim=-1)
        else:
            features = torch.cat([x[:, -1], context], dim=-1)

        return self.head(features)

    def _candidate_mask(
        self, record_key_tokens: torch.Tensor, query_tokens: torch.Tensor
    ) -> torch.Tensor:
        if self.candidate_mode == "full":
            return torch.ones_like(record_key_tokens, dtype=torch.bool)
        if self.candidate_mode == "oracle":
            return record_key_tokens == query_tokens.unsqueeze(1)

        record_bucket = self.token_hash(record_key_tokens)
        query_bucket = self.token_hash(query_tokens).unsqueeze(1)
        return record_bucket == query_bucket

    def _resolve_from_indices(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        candidate_idx: torch.Tensor,
    ) -> torch.Tensor:
        valid = candidate_idx >= 0
        safe_idx = candidate_idx.clamp_min(0)
        gather_idx = safe_idx.unsqueeze(-1).expand(-1, -1, values.size(-1))
        cand_keys = keys.gather(dim=1, index=gather_idx)
        cand_values = values.gather(dim=1, index=gather_idx)

        scores = torch.einsum("bkd,bd->bk", cand_keys, query) * self.scale
        scores = scores.masked_fill(~valid, -1.0e9)
        weights = torch.softmax(scores, dim=1) * valid.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        return torch.einsum("bk,bkd->bd", weights, cand_values)

    def _resolve(
        self,
        scores: torch.Tensor,
        values: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked_scores = scores.masked_fill(~candidate_mask, -1.0e9)
        n_records = scores.size(1)
        k = n_records if self.top_k <= 0 else min(self.top_k, n_records)

        if k < n_records:
            top_scores, top_idx = masked_scores.topk(k, dim=1)
            gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, values.size(-1))
            top_values = values.gather(dim=1, index=gather_idx)
            top_mask = candidate_mask.gather(dim=1, index=top_idx)
        else:
            top_scores = masked_scores
            top_values = values
            top_mask = candidate_mask

        weights = torch.softmax(top_scores, dim=1) * top_mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        return torch.einsum("bn,bnd->bd", weights, top_values)


class ContextAssociativeLM(nn.Module):
    """Context-keyed PCAF for language modeling.

    This version fixes the main weakness of the first PCAF prototype: records
    are keyed by a local n-gram context instead of one token. The output is a
    mixture of a parametric local LM path and sparse retrieved next-token
    probabilities.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int = 256,
        d_hidden: int = 1024,
        num_buckets: int = 32768,
        top_k: int = 16,
        candidate_mode: str = "triton_hash",
        context_order: int = 3,
        local_layers: int = 2,
        local_kernel_size: int = 5,
        dropout: float = 0.1,
        use_cache: bool = True,
        use_gate: bool = True,
        fixed_cache_weight: float = 0.5,
        semantic_buckets: int = 256,
        semantic_temperature: float = 0.2,
        semantic_score_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if candidate_mode not in {
            "full",
            "hash",
            "oracle",
            "triton_hash",
            "semantic_hash",
            "hybrid_semantic_hash",
        }:
            raise ValueError(
                "candidate_mode must be one of: full, hash, oracle, triton_hash, "
                "semantic_hash, hybrid_semantic_hash"
            )

        self.vocab_size = vocab_size
        self.num_buckets = num_buckets
        self.top_k = top_k
        self.candidate_mode = candidate_mode
        self.context_order = context_order
        self.use_cache = use_cache
        self.use_gate = use_gate
        self.fixed_cache_weight = fixed_cache_weight
        self.semantic_buckets = semantic_buckets
        self.semantic_temperature = semantic_temperature
        self.semantic_score_scale = semantic_score_scale
        self.scale = d_model**-0.5
        self.cache_eps = 1.0e-6

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.local_blocks = nn.ModuleList(
            [
                CausalConvBlock(d_model, d_hidden, local_kernel_size, dropout)
                for _ in range(local_layers)
            ]
        )

        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.semantic_router = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, semantic_buckets, bias=False),
        )
        self.param_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, vocab_size),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.recency_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, seq_len]")
        if tokens.size(1) < self.context_order + 1:
            raise ValueError("sequence length must exceed context_order")

        x = self.token_emb(tokens)
        for block in self.local_blocks:
            x = block(x)

        states = x
        final_state = states[:, -1]
        param_logits = self.param_head(final_state)
        param_log_probs = F.log_softmax(param_logits, dim=-1)
        if not self.use_cache:
            return param_log_probs

        context_hashes = causal_ngram_hash(tokens, self.context_order)
        record_hashes = context_hashes[:, :-1]
        query_hashes = context_hashes[:, -1]
        value_tokens = tokens[:, 1:]

        record_keys = F.normalize(self.key_proj(states[:, :-1]), dim=-1)
        query = F.normalize(self.query_proj(final_state), dim=-1)

        candidate_route_scores = None
        if self.candidate_mode in {"semantic_hash", "hybrid_semantic_hash"}:
            semantic_idx, semantic_scores = self._semantic_candidate_indices(
                states[:, :-1], final_state
            )
            if self.candidate_mode == "hybrid_semantic_hash":
                token_idx = self._token_candidate_indices(
                    record_hashes, query_hashes, query, record_keys
                )
                token_route_scores = self._gather_route_scores(
                    semantic_scores, token_idx
                )
                semantic_route_scores = self._gather_route_scores(
                    semantic_scores, semantic_idx
                )
                candidate_idx = torch.cat([token_idx, semantic_idx], dim=1)
                candidate_route_scores = torch.cat(
                    [token_route_scores, semantic_route_scores], dim=1
                )
            else:
                candidate_idx = semantic_idx
                candidate_route_scores = self._gather_route_scores(
                    semantic_scores, semantic_idx
                )
        else:
            candidate_idx = self._token_candidate_indices(
                record_hashes, query_hashes, query, record_keys
            )

        cache_log_probs, has_cache = self._cache_log_probs(
            query=query,
            record_keys=record_keys,
            value_tokens=value_tokens,
            candidate_idx=candidate_idx,
            candidate_route_scores=candidate_route_scores,
        )

        if self.use_gate:
            gate = torch.sigmoid(self.gate(final_state))
        else:
            gate = torch.full_like(param_logits[:, :1], self.fixed_cache_weight)
        gate = gate * has_cache.float().unsqueeze(1)
        return torch.logaddexp(
            torch.log1p(-gate.clamp(max=1.0 - 1.0e-5)) + param_log_probs,
            torch.log(gate.clamp_min(1.0e-5)) + cache_log_probs,
        )

    def _token_candidate_indices(
        self,
        record_hashes: torch.Tensor,
        query_hashes: torch.Tensor,
        query: torch.Tensor,
        record_keys: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.candidate_mode == "triton_hash"
            and query_bucket_indices is not None
            and record_hashes.is_cuda
            and self.top_k > 0
        ):
            return query_bucket_indices(
                record_hashes,
                query_hashes,
                num_buckets=self.num_buckets,
                max_bucket_size=self.top_k,
            )

        n_records = record_hashes.size(1)
        if self.candidate_mode == "full":
            candidate_mask = torch.ones_like(record_hashes, dtype=torch.bool)
        elif self.candidate_mode == "oracle":
            candidate_mask = record_hashes == query_hashes.unsqueeze(1)
        else:
            record_bucket = torch.remainder(
                record_hashes.long() * 1_000_003 + 97_531,
                self.num_buckets,
            )
            query_bucket = torch.remainder(
                query_hashes.long() * 1_000_003 + 97_531,
                self.num_buckets,
            ).unsqueeze(1)
            candidate_mask = record_bucket == query_bucket

        scores = torch.einsum("bnd,bd->bn", record_keys, query) * self.scale
        pos = torch.linspace(0.0, 1.0, n_records, device=scores.device)
        scores = scores + self.recency_scale * pos.unsqueeze(0)
        scores = scores.masked_fill(~candidate_mask, -1.0e9)
        k = n_records if self.top_k <= 0 else min(self.top_k, n_records)
        top_scores, top_idx = scores.topk(k, dim=1)
        valid = top_scores > -1.0e8
        return top_idx.masked_fill(~valid, -1)

    def _semantic_candidate_indices(
        self,
        record_states: torch.Tensor,
        final_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        record_logits = self.semantic_router(record_states)
        query_logits = self.semantic_router(final_state)
        temperature = max(float(self.semantic_temperature), 1.0e-4)
        record_probs = torch.softmax(record_logits / temperature, dim=-1)
        query_probs = torch.softmax(query_logits / temperature, dim=-1)
        semantic_scores = torch.einsum("bnc,bc->bn", record_probs, query_probs)

        n_records = record_states.size(1)
        pos = torch.linspace(0.0, 1.0, n_records, device=record_states.device)
        route_scores = semantic_scores + 1.0e-4 * self.recency_scale * pos.unsqueeze(0)
        k = n_records if self.top_k <= 0 else min(self.top_k, n_records)
        _, top_idx = route_scores.topk(k, dim=1)
        return top_idx, semantic_scores

    def _gather_route_scores(
        self,
        route_scores: torch.Tensor,
        candidate_idx: torch.Tensor,
    ) -> torch.Tensor:
        valid = candidate_idx >= 0
        safe_idx = candidate_idx.clamp_min(0)
        gathered = route_scores.gather(dim=1, index=safe_idx)
        return gathered.masked_fill(~valid, 0.0)

    def _cache_log_probs(
        self,
        *,
        query: torch.Tensor,
        record_keys: torch.Tensor,
        value_tokens: torch.Tensor,
        candidate_idx: torch.Tensor,
        candidate_route_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = candidate_idx >= 0
        safe_idx = candidate_idx.clamp_min(0)
        gather_key_idx = safe_idx.unsqueeze(-1).expand(-1, -1, record_keys.size(-1))
        cand_keys = record_keys.gather(dim=1, index=gather_key_idx)
        cand_tokens = value_tokens.gather(dim=1, index=safe_idx)

        scores = torch.einsum("bkd,bd->bk", cand_keys, query) * self.scale
        if candidate_route_scores is not None:
            scores = scores + self.semantic_score_scale * torch.log(
                candidate_route_scores.clamp_min(1.0e-6)
            )
        n_records = record_keys.size(1)
        recency = safe_idx.float() / max(float(n_records - 1), 1.0)
        scores = scores + self.recency_scale * recency
        scores = scores.masked_fill(~valid, -1.0e9)

        weights = torch.softmax(scores, dim=1) * valid.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)

        cache_probs = torch.zeros(
            (value_tokens.size(0), self.vocab_size),
            device=value_tokens.device,
            dtype=record_keys.dtype,
        )
        cache_probs.scatter_add_(dim=1, index=cand_tokens, src=weights)
        has_cache = valid.any(dim=1)
        return torch.log(cache_probs.clamp_min(self.cache_eps)), has_cache


class CausalSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        d_hidden: int,
        num_heads: int,
        dropout: float,
        attention_mode: str,
        window_size: int,
        global_tokens: int,
    ) -> None:
        super().__init__()
        if attention_mode not in {"dense", "local", "global_local"}:
            raise ValueError("unknown attention_mode")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.attention_mode = attention_mode
        self.window_size = window_size
        self.global_tokens = global_tokens
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(y)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class SparseAttentionClassifier(nn.Module):
    """Causal attention baseline with dense, local, or global+local masks."""

    def __init__(
        self,
        *,
        vocab_size: int,
        num_classes: int,
        d_model: int = 192,
        d_hidden: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        attention_mode: str = "local",
        window_size: int = 128,
        global_tokens: int = 8,
    ) -> None:
        super().__init__()
        self.attention_mode = attention_mode
        self.window_size = window_size
        self.global_tokens = global_tokens
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.register_buffer(
            "pos_emb", sinusoidal_positions(max_seq_len, d_model), persistent=False
        )
        self.layers = nn.ModuleList(
            [
                CausalSelfAttentionBlock(
                    d_model=d_model,
                    d_hidden=d_hidden,
                    num_heads=num_heads,
                    dropout=dropout,
                    attention_mode=attention_mode,
                    window_size=window_size,
                    global_tokens=global_tokens,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, num_classes),
        )
        self._mask_cache: dict[tuple[int, str, int, int, torch.device], torch.Tensor] = {}

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seq_len = tokens.size(1)
        if seq_len > self.pos_emb.size(0):
            raise ValueError(
                f"seq_len={seq_len} exceeds max_seq_len={self.pos_emb.size(0)}"
            )
        x = self.token_emb(tokens) + self.pos_emb[:seq_len].to(tokens.device)
        attn_mask = self._attention_mask(seq_len, tokens.device)
        for layer in self.layers:
            x = layer(x, attn_mask)
        return self.head(x[:, -1])

    def _attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (
            seq_len,
            self.attention_mode,
            self.window_size,
            self.global_tokens,
            device,
        )
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached

        q = torch.arange(seq_len, device=device).unsqueeze(1)
        k = torch.arange(seq_len, device=device).unsqueeze(0)
        allowed = k <= q

        if self.attention_mode == "local":
            allowed = allowed & ((q - k) < self.window_size)
        elif self.attention_mode == "global_local":
            local = (q - k) < self.window_size
            global_key = k < self.global_tokens
            allowed = allowed & (local | global_key)

        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.float32)
        mask = mask.masked_fill(~allowed, float("-inf"))
        self._mask_cache[key] = mask
        return mask


class LocalConvLM(nn.Module):
    """Local parametric LM ablation for pcaf_context."""

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int = 256,
        d_hidden: int = 1024,
        local_layers: int = 2,
        local_kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.local_blocks = nn.ModuleList(
            [
                CausalConvBlock(d_model, d_hidden, local_kernel_size, dropout)
                for _ in range(local_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, vocab_size),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(tokens)
        for block in self.local_blocks:
            x = block(x)
        return self.head(x[:, -1])


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_classes: int,
        d_model: int = 128,
        d_hidden: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.register_buffer(
            "pos_emb", sinusoidal_positions(max_seq_len, d_model), persistent=False
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_hidden,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, num_classes),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seq_len = tokens.size(1)
        if seq_len > self.pos_emb.size(0):
            raise ValueError(
                f"seq_len={seq_len} exceeds max_seq_len={self.pos_emb.size(0)}"
            )

        x = self.token_emb(tokens) + self.pos_emb[:seq_len].to(tokens.device)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        encoded = self.encoder(x, mask=causal_mask)
        return self.head(encoded[:, -1])
