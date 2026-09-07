from __future__ import annotations

from chitu_diffusion.parallel.interconnect import InterconnectProfile, _parse_nvlink

# Trimmed from `nvidia-smi topo -m` on the 8x RTX PRO 5000 host: no NVLink, GPUs
# 0-3 on one NUMA node and 4-7 on the other, with the NIC and affinity columns
# that must not be mistaken for peer links.
_PCIE_MATRIX = (
    "\t\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\tNIC0\tCPU Affinity\tNUMA Affinity\x1b[0m\n"
    "GPU0\t X \tNODE\tSYS\tSYS\tNODE\t0-191\t0\n"
    "GPU1\tNODE\t X \tSYS\tSYS\tNODE\t0-191\t0\n"
    "GPU2\tSYS\tSYS\t X \tNODE\tSYS\t192-383\t1\n"
    "GPU3\tSYS\tSYS\tNODE\t X \tSYS\t192-383\t1\n"
    "NIC0\tNODE\tNODE\tSYS\tSYS\t X \t\t\n"
)

_NVLINK_MATRIX = (
    "\tGPU0\tGPU1\tNIC0\tCPU Affinity\tNUMA Affinity\n"
    "GPU0\t X \tNV18\tSYS\t0-95\t0\n"
    "GPU1\tNV18\t X \tSYS\t0-95\t0\n"
    "NIC0\tSYS\tSYS\t X \t\t\n"
)

# Older builds pad the columns with spaces instead of separating them with tabs.
_SPACED_MATRIX = (
    "        GPU0    GPU1    CPU Affinity\n"
    "GPU0     X      NV4     0-95\n"
    "GPU1    NV4      X      0-95\n"
)


def test_parse_nvlink_reads_peer_links_only() -> None:
    assert _parse_nvlink(_PCIE_MATRIX) is False
    assert _parse_nvlink(_NVLINK_MATRIX) is True
    assert _parse_nvlink(_SPACED_MATRIX) is True


def test_parse_nvlink_reports_an_unreadable_matrix_as_unknown() -> None:
    assert _parse_nvlink("") is None
    assert _parse_nvlink("nvidia-smi: command not found") is None
    assert _parse_nvlink("\tGPU0\tGPU1\n") is None


def test_shared_egress_only_claims_a_confirmed_pcie_fabric() -> None:
    assert InterconnectProfile(nvlink=False, numa_nodes=(0, 1)).shared_egress
    assert not InterconnectProfile(nvlink=True, numa_nodes=(0, 1)).shared_egress
    assert not InterconnectProfile(nvlink=None, numa_nodes=()).shared_egress


def test_describe_names_the_fabric_and_the_numa_split() -> None:
    profile = InterconnectProfile(nvlink=False, numa_nodes=(0, 0, 1, 1))
    assert profile.describe() == "fabric=pcie numa_nodes=[0, 0, 1, 1]"
    assert "unknown" in InterconnectProfile(nvlink=None, numa_nodes=()).describe()
