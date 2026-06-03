import json

from chitu_diffusion.observability.timer import Timer


def test_save_task_statistics_json_uses_task_local_records(tmp_path):
    Timer.reset()
    Timer.record("dit_forward", 100.0)
    Timer.record_event("dit_forward", {"task_id": "task-a", "elapsed_ms": 100.0})
    Timer.record_event("dit_forward_step", {"task_id": "task-a", "elapsed_ms": 100.0})
    Timer.record("dit_forward", 20.0)
    Timer.record_event("dit_forward", {"task_id": "task-b", "elapsed_ms": 20.0})
    Timer.record_event("dit_forward_step", {"task_id": "task-b", "elapsed_ms": 20.0})

    path = tmp_path / "task-b.json"
    Timer.save_task_statistics_json(str(path), "task-b")

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["timers"]["dit_forward"]["total_ms"] == 20.0
    assert payload["timers"]["dit_forward"]["samples"] == 1
    assert payload["timers"]["dit_forward_step"]["total_ms"] == 20.0
    assert len(payload["records"]["dit_forward"]) == 1

    Timer.reset()
