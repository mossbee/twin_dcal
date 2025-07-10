"""
DCAL Modules Package
Core implementation of Dual Cross-Attention Learning for Twin Face Verification
"""

from .attention import (
    MultiHeadSelfAttention,
    GlobalLocalCrossAttention, 
    PairWiseCrossAttention,
    AttentionRollout
)
from .transformer import (
    TransformerBlock,
    GLCABlock,
    PWCABlock,
    DCALEncoder
)
from .backbone import (
    PatchEmbedding,
    PositionalEmbedding,
    VisionTransformerBackbone
)

__all__ = [
    'MultiHeadSelfAttention',
    'GlobalLocalCrossAttention',
    'PairWiseCrossAttention', 
    'AttentionRollout',
    'TransformerBlock',
    'GLCABlock',
    'PWCABlock',
    'DCALEncoder',
    'PatchEmbedding',
    'PositionalEmbedding',
    'VisionTransformerBackbone'
] 