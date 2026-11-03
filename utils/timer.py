"""
Timer Utility Module
====================

Provides a context-manager-based timer for measuring execution time
of code blocks. Used throughout the application for performance tracking.
"""

import time


class Timer:
    """
    Context manager for measuring elapsed time of a code block.

    Usage:
        with Timer() as t:
            # ... code to measure ...
        print(f"Elapsed: {t.elapsed:.4f} sec")
    """

    def __init__(self):
        self._start: float = 0.0
        self._end: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end = time.perf_counter()
        self.elapsed = self._end - self._start
        return False  # Do not suppress exceptions
