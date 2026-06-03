import time
import torch
from typing import Any, Callable, Dict
from contextlib import ContextDecorator

class Timer:
    """Static timer implementation with CUDA synchronization"""
    _timers: Dict[str, Dict] = {}
    _records: Dict[str, list[Dict[str, Any]]] = {}
    _global_enabled = True

    @staticmethod
    def _ensure_timer_exists(name: str):
        """Ensure timer entry exists with proper initialization"""
        if name not in Timer._timers:
            Timer._timers[name] = {
                'times': [],
                'start_time': 0,
                'is_timing_valid': False
            }

    @staticmethod
    def get_timer(name: str) -> ContextDecorator:
        """Get a context manager for timing a code block
        
        Args:
            name: Unique identifier for the timer
        """
        Timer._ensure_timer_exists(name)
            
        class _TimerContext(ContextDecorator):
            def __enter__(self):
                if Timer._global_enabled:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    Timer._timers[name]['start_time'] = time.perf_counter()
                    Timer._timers[name]['is_timing_valid'] = True
                return self

            def __exit__(self, *args):
                if Timer._global_enabled and Timer._timers[name]['is_timing_valid']:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed_time = (time.perf_counter() - Timer._timers[name]['start_time']) * 1000
                    Timer._timers[name]['times'].append(elapsed_time)
                Timer._timers[name]['is_timing_valid'] = False

        return _TimerContext()

    @staticmethod
    def record(name: str, elapsed_ms: float):
        """Record one timing sample in milliseconds for a timer name."""
        if not Timer._global_enabled:
            return
        Timer._ensure_timer_exists(name)
        Timer._timers[name]['times'].append(float(elapsed_ms))

    @staticmethod
    def time_call(name: str, fn: Callable, *args, **kwargs):
        """Run a callable and record its elapsed time in milliseconds."""
        if not Timer._global_enabled:
            return fn(*args, **kwargs), 0.0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        result = fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        Timer.record(name, elapsed_ms)
        return result, elapsed_ms

    @staticmethod
    def record_event(name: str, payload: Dict[str, Any]):
        """Record structured timing metadata that should be preserved in JSON."""
        if not Timer._global_enabled:
            return
        Timer._records.setdefault(name, []).append(dict(payload))

    @staticmethod
    def print_statistics():
        """Print timing statistics for all timers"""
        if not Timer._global_enabled:
            print("Timing is disabled")
            return

        print("\n===== Timing Statistics =====")
        header = "{:<20} | {:>10} | {:>10} | {:>10} | {:>8}"
        print(header.format("Timer Name", "Min (ms)", "Max (ms)", "Avg (ms)", "Samples"))
        print("-" * 66)
        
        row_format = "{:<20} | {:>10.4f} | {:>10.4f} | {:>10.4f} | {:>8d}"
        for name, timer in Timer._timers.items():
            times = timer['times']
            if times:
                print(row_format.format(
                    name[:20], 
                    min(times),
                    max(times),
                    sum(times) / len(times),
                    len(times)
                ))
        print("=" * 66 + "\n")

    @staticmethod
    def statistics_dict() -> Dict[str, Dict[str, float]]:
        stats = {}
        for name, timer in Timer._timers.items():
            item = Timer._statistics_from_times(timer['times'])
            if item is not None:
                stats[name] = item
        return stats

    @staticmethod
    def _statistics_from_times(times: list[float]) -> Dict[str, float] | None:
        if not times:
            return None
        return {
            "min_ms": min(times),
            "max_ms": max(times),
            "avg_ms": sum(times) / len(times),
            "total_ms": sum(times),
            "samples": len(times),
        }

    @staticmethod
    def records_dict() -> Dict[str, list[Dict[str, Any]]]:
        return {name: list(records) for name, records in Timer._records.items()}

    @staticmethod
    def records_for_task(task_id: str) -> Dict[str, list[Dict[str, Any]]]:
        task_records = {}
        for name, records in Timer._records.items():
            filtered = [record for record in records if record.get("task_id") == task_id]
            if filtered:
                task_records[name] = filtered
        return task_records

    @staticmethod
    def task_statistics_dict(task_id: str) -> Dict[str, Dict[str, float]]:
        stats = {}
        for name, records in Timer.records_for_task(task_id).items():
            times = [
                float(record["elapsed_ms"])
                for record in records
                if isinstance(record.get("elapsed_ms"), (int, float))
            ]
            item = Timer._statistics_from_times(times)
            if item is not None:
                stats[name] = item
        return stats

    @staticmethod
    def save_statistics_json(filepath="timing_stats.json"):
        if not Timer._global_enabled:
            return

        import json
        import os
        from datetime import datetime

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timers": Timer.statistics_dict(),
            "records": Timer.records_dict(),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def save_task_statistics_json(filepath: str, task_id: str):
        if not Timer._global_enabled:
            return

        import json
        import os
        from datetime import datetime

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_id": task_id,
            "timers": Timer.task_statistics_dict(task_id),
            "records": Timer.records_for_task(task_id),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def enable():
        """Enable timing measurement"""
        Timer._global_enabled = True

    @staticmethod
    def disable():
        """Disable timing measurement"""
        Timer._global_enabled = False
        for timer in Timer._timers.values():
            timer['is_timing_valid'] = False

    @staticmethod
    def is_enabled() -> bool:
        """Check if timing is enabled"""
        return Timer._global_enabled

    @staticmethod
    def reset():
        """Reset all timers"""
        Timer._timers.clear()
        Timer._records.clear()
