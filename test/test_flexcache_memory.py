import torch

from chitu_diffusion.flexcache.flexcache_manager import FlexCacheManager


def test_cache_tensor_stats_counts_tensor_shapes_dtypes_and_bytes():
    manager = FlexCacheManager(max_cache_memory=20)
    first = torch.zeros((2, 3), dtype=torch.float16)
    second = torch.zeros((4,), dtype=torch.float32)

    manager.cache["first"] = first
    manager.cache["nested"] = {"second": second}

    increased, stats = manager.update_peak_cache_memory()

    assert increased is True
    assert stats["entries"] == 2
    assert stats["tensors"] == 2
    assert stats["bytes"] == first.numel() * first.element_size() + second.numel() * second.element_size()
    assert stats["peak_bytes"] == stats["bytes"]
    assert stats["peak_entries"] == 2
    assert stats["peak_tensors"] == 2
    assert {
        (tuple(item["shape"]), item["dtype"], item["bytes"])
        for item in stats["tensor_details"]
    } == {
        ((2, 3), "torch.float16", 12),
        ((4,), "torch.float32", 16),
    }
    assert {
        (tuple(item["shape"]), item["dtype"], item["count"], item["bytes"])
        for item in stats["tensor_summary"]
    } == {
        ((2, 3), "torch.float16", 1, 12),
        ((4,), "torch.float32", 1, 16),
    }


def test_peak_cache_memory_only_increases_and_clear_resets_stats():
    manager = FlexCacheManager(max_cache_memory=20)
    manager.cache["large"] = torch.zeros((8,), dtype=torch.float32)

    increased, stats = manager.update_peak_cache_memory()
    assert increased is True
    assert stats["peak_bytes"] == 32

    manager.cache["large"] = torch.zeros((2,), dtype=torch.float32)
    increased, stats = manager.update_peak_cache_memory()
    assert increased is False
    assert stats["bytes"] == 8
    assert stats["peak_bytes"] == 32

    manager.clear_cache()
    assert manager.cache == {}
    assert manager.peak_cache_bytes == 0
    assert manager.peak_cache_entries == 0
    assert manager.peak_cache_tensors == 0
