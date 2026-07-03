"""Proprioception-queried cross-attention encoder for fusing a depth image with
proprioceptive information.

Minimal, single-camera version inspired by CReF (Cross-modal and Recurrent
Fusion, Hao et al. 2026) and the SRU memory ``CrossAttentionFuseModule``.

Pipeline (per frame)::

    depth ─Conv tokenizer─► N image tokens ─(+pos)─► self-attn x L ─► LN ─┐ (K, V)
    proprio ─info MLP─► 1 query token ─► LN ───────────────────────────────┤ (Q)
                                                                            ▼
                           CrossAttn(Q=proprio, K=V=image) ─► 1 fused token ─► out_proj

Following CReF (Eq. 8-9), both attention inputs are LayerNorm-ed: the token
stream gets a final LN before serving as K/V (with ``num_self_attn_layers=0``
this is exactly the paper's ``E_d = LN(Z_t)``), and the proprioceptive query
token is normalized before projection (``Q = LN(e_p) W_q``, where ``W_q`` is
the in-projection inside ``nn.MultiheadAttention``).

Unlike the full CReF block, this variant drops the Gated Residual Fusion and the
highway output gate: the fused depth feature is simply concatenated with the raw
proprioception downstream (handled by ``ParallelLayer``) and fed to the existing
recurrent memory.
"""

import torch
import torch.nn as nn

from instinct_rl.modules.conv2d import Conv2dModel


class _PreNormSelfAttnLayer(nn.Module):
    """Pre-norm transformer encoder layer: self-attention + FFN, both residual."""

    def __init__(self, d_model: int, num_heads: int, expand_dim: int, nonlinearity) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        # standard transformer FFN: no activation after the second Linear, so the
        # residual update is not biased to the activation's output range.
        self.ffn = nn.Sequential(
            nn.Linear(d_model, expand_dim),
            nonlinearity(),
            nn.Linear(expand_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        sa, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + sa
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttnFuseHeadModel(nn.Module):
    """Conv tokenizer + self-attention + proprioception-queried cross-attention.

    Args:
        image_shape: Shape of the depth image ``(C, H, W)``.
        info_dim: Total dimension of the proprioceptive query input.
        output_size: Dimension of the fused output feature.
        channels: Output channels of each conv layer. ``channels[-1]`` is the
            token dimension ``d_model`` and must be divisible by ``num_heads``.
        kernel_sizes: Kernel size of each conv layer.
        strides: Stride of each conv layer.
        paddings: Padding of each conv layer. Defaults to zeros.
        num_heads: Number of attention heads.
        num_self_attn_layers: Number of stacked pre-norm self-attention layers
            applied to the image tokens before cross-attention. ``0`` skips
            self-attention entirely (tokenizer -> LN -> K/V, the exact CReF
            form; the learned positional embedding is still added).
        ffn_expansion: Hidden-size multiplier for the self-attention FFN.
        info_hidden_sizes: Hidden layer widths of the proprioceptive query MLP
            (info -> query token). The output is always ``d_model``. Defaults to
            ``[d_model * ffn_expansion]`` (one hidden layer).
        nonlinearity: Activation module (or its name in ``torch.nn``).
        use_maxpool: Whether the conv tokenizer uses max-pooling for downsampling.
    """

    def __init__(
        self,
        image_shape,
        info_dim,
        output_size,
        channels,
        kernel_sizes,
        strides,
        paddings=None,
        num_heads: int = 4,
        num_self_attn_layers: int = 1,
        ffn_expansion: int = 2,
        info_hidden_sizes=None,
        nonlinearity=nn.ELU,
        use_maxpool: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(nonlinearity, str):
            nonlinearity = getattr(nn, nonlinearity)

        c, h, w = image_shape
        self.image_shape = tuple(image_shape)
        self.info_dim = info_dim

        d_model = channels[-1]
        assert d_model % num_heads == 0, "channels[-1] (d_model) must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads

        # --- image path: conv tokenizer ---
        self.conv = Conv2dModel(
            in_channels=c,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            nonlinearity=nonlinearity,
            use_maxpool=use_maxpool,
        )
        h2, w2 = self.conv.conv_out_resolution(h, w)
        self.token_resolution = (h2, w2)
        self.n_tokens = h2 * w2

        # learned positional embedding over the (fixed) token grid
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        expand_dim = d_model * ffn_expansion

        # stacked self-attention layers over the image tokens (pre-norm)
        self.self_attn_layers = nn.ModuleList(
            [_PreNormSelfAttnLayer(d_model, num_heads, expand_dim, nonlinearity) for _ in range(num_self_attn_layers)]
        )

        # --- proprio path: info -> single query token ---
        # Hidden widths are configurable; the output dim is forced to d_model so
        # that the query matches the image key/value embedding dim. Default keeps
        # the previous behaviour: a single hidden layer of size d_model*ffn_expansion.
        if info_hidden_sizes is None:
            info_hidden_sizes = [expand_dim]
        info_dims = [info_dim] + list(info_hidden_sizes) + [d_model]
        info_layers = []
        for i, (in_dim, out_dim) in enumerate(zip(info_dims[:-1], info_dims[1:])):
            info_layers.append(nn.Linear(in_dim, out_dim))
            # the final layer stays linear: the query token is LayerNorm-ed and
            # projected by the attention in-projection, an activation before it
            # would only restrict the query to the activation's output range.
            if i < len(info_dims) - 2:
                info_layers.append(nonlinearity())
        self.info_proj = nn.Sequential(*info_layers)

        # interface norms for cross-attention (CReF Eq. 8-9): final LN over the
        # (pre-norm, hence unnormalized) token stream used as K/V, and LN on the
        # proprioceptive query token.
        self.token_norm = nn.LayerNorm(d_model)
        self.query_norm = nn.LayerNorm(d_model)

        # cross-attention sub-layer (proprio query, image key/value)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        self.out_proj = nn.Identity() if output_size == d_model else nn.Linear(d_model, output_size)
        self._output_size = output_size

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, img: torch.Tensor, info: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Depth image of shape ``(N, C, H, W)``.
            info: Proprioceptive vector of shape ``(N, info_dim)``.

        Returns:
            Fused feature of shape ``(N, output_size)``.
        """
        # image tokens: (N, d_model, H', W') -> (N, n_tokens, d_model)
        x = self.conv(img).flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        # stacked self-attention over image tokens (pre-norm + residual)
        for layer in self.self_attn_layers:
            x = layer(x)
        kv = self.token_norm(x)

        # proprio as query, image tokens as key/value
        q = self.query_norm(self.info_proj(info)).unsqueeze(1)  # (N, 1, d_model)
        ca, _ = self.cross_attn(q, kv, kv, need_weights=False)  # (N, 1, d_model)

        return self.out_proj(ca.squeeze(1))  # (N, output_size)
