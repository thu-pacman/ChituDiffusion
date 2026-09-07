"""How the local GPUs are wired to each other.

Several parallel plans have to choose between a transfer schedule that spreads a
message across peers and one that serialises it, and the right answer is a
property of the fabric rather than of the model. A PCIe GPU fans every peer out
of a single x16 egress port, so spreading a transfer only costs efficiency; an
NVLink GPU owns an independent link per peer and rewards the opposite. This
module answers that question once per process, from the driver rather than from
a benchmark, so that every rank of a node-local group reaches the same
conclusion.
"""

from __future__ import annotations

import functools
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_PCI_DEVICES = Path("/sys/bus/pci/devices")
_NVLINK_CELL = re.compile(r"NV\d+")
_GPU_LABEL = re.compile(r"GPU\d+")
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
# nvidia-smi separates topology columns with tabs, older builds pad with spaces.
_COLUMN_SPLIT = re.compile(r"\t|\s{2,}")
_UNKNOWN_NUMA = -1
_TOPOLOGY_TIMEOUT_S = 15.0


@dataclass(frozen=True, slots=True)
class InterconnectProfile:
    """The fabric between the CUDA devices visible to this process."""

    nvlink: bool | None
    """Whether any two local GPUs share an NVLink; None when undetermined."""

    numa_nodes: tuple[int, ...]
    """NUMA node per device ordinal, -1 where the host does not report one."""

    @property
    def shared_egress(self) -> bool:
        """Whether a GPU reaches all of its peers through one shared port.

        An undetermined fabric counts as NVLink: a host this module cannot read
        keeps the schedules that were the historical default instead of being
        moved onto a fan-out that only pays off on PCIe.
        """
        return self.nvlink is False

    def describe(self) -> str:
        fabric = "pcie" if self.nvlink is False else "nvlink" if self.nvlink else "unknown"
        return f"fabric={fabric} numa_nodes={list(self.numa_nodes)}"


@functools.cache
def local_interconnect() -> InterconnectProfile:
    """Probe the local fabric once per process."""
    profile = InterconnectProfile(
        nvlink=_detect_nvlink(),
        numa_nodes=_detect_numa_nodes(),
    )
    logger.info("local GPU interconnect: %s", profile.describe())
    return profile


def _detect_numa_nodes() -> tuple[int, ...]:
    """Read each device's NUMA node from sysfs, keyed by its PCI address."""
    if not torch.cuda.is_available():
        return ()
    nodes = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        address = (
            f"{properties.pci_domain_id:04x}:"
            f"{properties.pci_bus_id:02x}:"
            f"{properties.pci_device_id:02x}.0"
        )
        path = _PCI_DEVICES / address / "numa_node"
        try:
            nodes.append(int(path.read_text().strip()))
        except (OSError, ValueError):
            nodes.append(_UNKNOWN_NUMA)
    return tuple(nodes)


def _detect_nvlink() -> bool | None:
    """Look for an NVLink cell in the driver's GPU-to-GPU topology matrix.

    Returns None when the matrix cannot be read or parsed, which the callers
    treat as the conservative answer rather than as "no NVLink".
    """
    try:
        completed = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=_TOPOLOGY_TIMEOUT_S,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        logger.debug("could not read the GPU topology matrix: %s", error)
        return None
    return _parse_nvlink(completed.stdout)


def _parse_nvlink(matrix: str) -> bool | None:
    """Return whether any GPU-to-GPU cell of `nvidia-smi topo -m` is NVLink.

    The matrix also carries NIC columns and affinity columns, so only the first
    `gpu_columns` cells of a GPU row are peer links.
    """
    rows = [
        [cell.strip() for cell in _COLUMN_SPLIT.split(_ANSI_ESCAPE.sub("", line))]
        for line in matrix.splitlines()
    ]
    # The header is the one row that opens with a blank corner cell.
    header = next(
        (row for row in rows if row and not row[0] and any(cell.startswith("GPU") for cell in row)),
        None,
    )
    if header is None:
        return None
    gpu_columns = sum(1 for cell in header if cell.startswith("GPU"))
    seen = False
    for cells in rows:
        if not cells or not _GPU_LABEL.fullmatch(cells[0]):
            continue
        seen = True
        if any(_NVLINK_CELL.fullmatch(cell) for cell in cells[1 : 1 + gpu_columns]):
            return True
    return False if seen else None
