"""H20 calibration and validated Fast CP execution capabilities.

These are kernel costs and capabilities, not a per-shape dispatch table.
"""

from dataclasses import dataclass
from types import MappingProxyType

KERNEL_COSTS = MappingProxyType(
    {
        (64, 64, True): (7.151995, 0.794668, 0.020053922),
        (128, 64, True): (5.775996, 1.862289, 0.036416492),
        (128, 128, True): (5.231974, 2.811794, 0.036344894),
        (192, 64, False): (5.264019, 2.812087, 0.054600717),
    }
)
CUDNN_COST = (8.431994, 0.708539, 0.036459461)


@dataclass(frozen=True)
class H20NVLinkProfile:
    name: str = "h20_nvlink"
    version: int = 1
    cp_degrees: tuple[int, ...] = (2, 4, 8)
    ulysses_transfers: tuple[str, ...] = ("sm", "tma", "ce")
    agkv_transfer: str = "ce"
    remote_streams: int = 1
    peer_order: str = "xor"
    outer_graph_capture: bool = False

    def supports_device(self, name, capability, sms, nvlink):
        return (
            name in {"H20", "NVIDIA H20"}
            and capability == (9, 0)
            and 0 < sms <= 78
            and nvlink is True
        )


H20_NVLINK = H20NVLinkProfile()
