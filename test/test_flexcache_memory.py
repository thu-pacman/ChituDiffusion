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


def test_compute_stats_accumulate_saved_units_by_scope(monkeypatch):
    monkeypatch.delenv("CHITU_CURRENT_OUTPUT_DIR", raising=False)
    manager = FlexCacheManager(max_cache_memory=20)

    manager.record_compute(baseline_units=10, actual_units=4, scope="token_forward", unit="tokens")
    manager.record_compute(baseline_units=1, actual_units=0, scope="model_compute", unit="model_forward")

    summary = manager.compute_summary()

    assert summary["baseline_units"] == 11
    assert summary["actual_units"] == 4
    assert summary["saved_units"] == 7
    assert summary["saving_ratio"] == 7 / 11
    assert summary["scope_summary"]["token_forward"]["saving_ratio"] == 0.6
    assert summary["scope_summary"]["model_compute"]["saved_units"] == 1

    manager.reset_compute_stats()
    assert manager.compute_summary()["baseline_units"] == 0


def test_cache_memory_events_buffer_only_new_peak_keys(monkeypatch):
    manager = FlexCacheManager(max_cache_memory=20)
    calls = []

    def fake_update_peak_cache_memory():
        calls.append("scan")
        increased = len(calls) == 1
        return increased, {
            "entries": 1,
            "tensors": 1,
            "bytes": 8,
            "peak_entries": 1,
            "peak_tensors": 1,
            "peak_bytes": 8,
            "peak_increased": increased,
            "tensor_summary": [],
        }

    monkeypatch.setattr(manager, "update_peak_cache_memory", fake_update_peak_cache_memory)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    monkeypatch.setenv("CHITU_CURRENT_OUTPUT_DIR", "/tmp/chitu-test")

    manager.record_cache_memory("flexcache_store", extra={"cache_key": "a"})
    manager.record_cache_memory("flexcache_store", extra={"cache_key": "a"})
    manager.record_cache_memory("flexcache_store", extra={"cache_key": "b"})

    assert calls == ["scan", "scan"]
    assert len(manager.cache_memory_events) == 1
    assert manager.cache_memory_events[0]["cache_key"] == "a"


def test_flush_cache_memory_events_writes_buffered_events(monkeypatch):
    manager = FlexCacheManager(max_cache_memory=20)
    written = []

    def fake_append(path, key, item, base_payload=None):
        written.append((path, key, item, base_payload))

    monkeypatch.setenv("CHITU_CURRENT_OUTPUT_DIR", "/tmp/chitu-test")
    monkeypatch.setattr("chitu_diffusion.flexcache.flexcache_manager.append_json_list_item", fake_append)
    manager.cache_memory_events.append({"stage": "flexcache_store", "rank": 0})

    manager.flush_cache_memory_events()

    assert len(written) == 1
    assert written[0][1] == "events"
    assert written[0][2]["stage"] == "flexcache_store"
    assert manager.cache_memory_events == []
