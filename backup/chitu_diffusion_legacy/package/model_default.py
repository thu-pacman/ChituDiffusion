from dataclasses import dataclass, field

@dataclass(frozen=True)
class WanModelDefaults:
    """WanModel 的默认超参常量池（与 __init__ 默认值保持同步）"""
    model_type: str = 't2v'
    patch_size: tuple = (1, 2, 2)
    text_len: int = 512
    in_dim: int = 16
    dim: int = 2048
    ffn_dim: int = 8192
    freq_dim: int = 256
    text_dim: int = 4096
    out_dim: int = 16
    num_heads: int = 16
    num_layers: int = 32
    window_size: tuple = (-1, -1)
    qk_norm: bool = True
    cross_attn_norm: bool = True
    eps: float = 1e-6
