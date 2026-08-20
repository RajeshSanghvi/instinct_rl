"""AME-style xyz heightmap attention for instinct_rl encoder pipelines."""

import math

import torch
import torch.nn as nn


def _resolve_xyz_layout(image_shape, map_shape):
    image_shape = tuple(int(size) for size in image_shape)
    if len(image_shape) == 1:
        if map_shape is None:
            raise ValueError("map_shape=(H, W, 3) is required for a flattened xyz heightmap")
        map_shape = tuple(int(size) for size in map_shape)
        if len(map_shape) != 3 or map_shape[-1] != 3:
            raise ValueError(f"map_shape must be (H, W, 3), got {map_shape}")
        if math.prod(map_shape) != image_shape[0]:
            raise ValueError(
                f"Flattened heightmap has {image_shape[0]} values, but map_shape={map_shape} "
                f"requires {math.prod(map_shape)}"
            )
        return (3, map_shape[0], map_shape[1]), "flat_hwc", map_shape

    if map_shape is not None:
        raise ValueError("map_shape is only used when the observation component is flattened")
    if len(image_shape) != 3:
        raise ValueError(f"xyz heightmap shape must be flat, CHW, or HWC; got {image_shape}")
    if image_shape[0] == 3:
        return image_shape, "chw", (image_shape[1], image_shape[2], 3)
    if image_shape[-1] == 3:
        return (3, image_shape[0], image_shape[1]), "hwc", image_shape
    raise ValueError(f"xyz heightmap must contain exactly three coordinate channels; got {image_shape}")


class _AMEHeightmapTokenizer(nn.Sequential):
    def __init__(self, mha_dim: int, cnn_downsample: bool) -> None:
        first_stride = 2 if cnn_downsample else 1
        second_kernel = 3 if cnn_downsample else 5
        second_padding = 1 if cnn_downsample else 2
        super().__init__(
            nn.Conv2d(3, 16, kernel_size=5, padding=2, stride=first_stride),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, mha_dim, kernel_size=second_kernel, padding=second_padding),
            nn.ReLU(),
            nn.BatchNorm2d(mha_dim),
        )


class AMEHeightmapAttentionHeadModel(nn.Module):
    """Reproduce AME's proprioception-queried xyz heightmap attention.

    The module applies AME's two-layer CNN directly to xyz channels, flattens
    the CNN feature grid into terrain tokens, projects proprioception with one
    Linear layer to form a single Query, and uses the terrain tokens as Key and
    Value. It deliberately has no positional encoding, token self-attention, or
    LayerNorm at the cross-attention interface.

    ``image_shape`` may be flattened, ``(3, H, W)``, or ``(H, W, 3)``. For a
    flattened observation, ``map_shape=(H, W, 3)`` defines its point-major HWC
    layout. The raw proprioceptive components remain available to the downstream
    instinct_rl policy; this module returns only AME's attended terrain feature
    and, when enabled, the global terrain feature.

    Args:
        image_shape: Shape reported by the xyz heightmap observation component.
        info_dim: Total size of the proprioceptive Query input.
        output_size: Must be ``mha_dim`` without global context and
            ``2 * mha_dim`` with global context.
        map_shape: HWC layout of a flattened xyz heightmap.
        mha_dim: Terrain token and attention embedding dimension.
        num_heads: Number of cross-attention heads.
        cnn_downsample: Use AME's stride-2 first convolution.
        attach_global: Add AME2's token MLP + max-pool global context to the
            Query and returned feature.
    """

    def __init__(
        self,
        image_shape,
        info_dim: int,
        output_size: int,
        map_shape=None,
        mha_dim: int = 64,
        num_heads: int = 16,
        cnn_downsample: bool = True,
        attach_global: bool = False,
    ) -> None:
        super().__init__()
        if mha_dim % num_heads != 0:
            raise ValueError(f"mha_dim={mha_dim} must be divisible by num_heads={num_heads}")

        expected_output_size = mha_dim * (2 if attach_global else 1)
        if output_size != expected_output_size:
            raise ValueError(
                f"output_size must be {expected_output_size} for mha_dim={mha_dim} "
                f"and attach_global={attach_global}, got {output_size}"
            )

        conv_image_shape, input_layout, resolved_map_shape = _resolve_xyz_layout(image_shape, map_shape)
        self.image_shape = tuple(image_shape)
        self.map_shape = resolved_map_shape
        self.conv_image_shape = conv_image_shape
        self.input_layout = input_layout
        self.info_dim = info_dim
        self.mha_dim = mha_dim
        self.num_heads = num_heads
        self.cnn_downsample = cnn_downsample
        self.attach_global = attach_global
        self._output_size = output_size

        self.map_cnn = _AMEHeightmapTokenizer(mha_dim, cnn_downsample)
        self.proprio_embedding = nn.Linear(info_dim, mha_dim)
        self.mha = nn.MultiheadAttention(embed_dim=mha_dim, num_heads=num_heads, batch_first=True)

        if attach_global:
            self.global_encoder = nn.Sequential(
                nn.Linear(mha_dim, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, mha_dim),
            )
            self.query_projector = nn.Linear(2 * mha_dim, mha_dim)
        else:
            self.global_encoder = None
            self.query_projector = None

    @property
    def output_size(self) -> int:
        return self._output_size

    def _as_chw(self, heightmap: torch.Tensor) -> torch.Tensor:
        if self.input_layout == "flat_hwc":
            heightmap = heightmap.reshape(-1, *self.map_shape).permute(0, 3, 1, 2)
        elif self.input_layout == "hwc":
            heightmap = heightmap.reshape(-1, *self.map_shape).permute(0, 3, 1, 2)
        else:
            heightmap = heightmap.reshape(-1, *self.conv_image_shape)
        return heightmap.contiguous()

    def _encode(self, heightmap: torch.Tensor, info: torch.Tensor, need_weights: bool):
        local_features = self.map_cnn(self._as_chw(heightmap)).flatten(2).transpose(1, 2)
        proprio_embedding = self.proprio_embedding(info.reshape(-1, info.shape[-1]))

        global_features = None
        query = proprio_embedding
        if self.attach_global:
            global_features = self.global_encoder(local_features).max(dim=1).values
            query = self.query_projector(torch.cat([global_features, proprio_embedding], dim=-1))

        attended, attention_weights = self.mha(
            query=query.unsqueeze(1),
            key=local_features,
            value=local_features,
            need_weights=need_weights,
        )
        encoded = attended.squeeze(1)
        if global_features is not None:
            encoded = torch.cat([global_features, encoded], dim=-1)
        return encoded, attention_weights

    def forward(self, heightmap: torch.Tensor, info: torch.Tensor) -> torch.Tensor:
        encoded, _ = self._encode(heightmap, info, need_weights=False)
        return encoded

    def forward_with_attention(self, heightmap: torch.Tensor, info: torch.Tensor):
        """Return the encoded feature and AME-style head-averaged attention weights."""
        encoded, attention_weights = self._encode(heightmap, info, need_weights=True)
        return encoded, attention_weights
