import time
import torch
from typing import Dict, Optional
from contextlib import ContextDecorator

class Timer:
    """Static timer implementation with CUDA synchronization"""
    _timers: Dict[str, Dict] = {}
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
    def save_statistics(filepath="timing_stats.csv"):
        """Save timing statistics to CSV file in append mode"""
        if not Timer._global_enabled:
            return

        from datetime import datetime
        import csv
        
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write timestamp and header if file is empty
            if f.tell() == 0:
                writer.writerow(['Timestamp', 'Timer Name', 'Min (ms)', 'Max (ms)', 'Avg (ms)', 'Samples'])
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for name, timer in Timer._timers.items():
                times = timer['times']
                if times:
                    writer.writerow([
                        timestamp,
                        name,
                        min(times),
                        max(times),
                        sum(times) / len(times),
                        len(times)
                    ])

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