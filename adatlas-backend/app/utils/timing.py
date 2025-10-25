import time
from contextlib import contextmanager

@contextmanager
def stopwatch(label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[TIMER] {label}: {dt:.3f}s")
