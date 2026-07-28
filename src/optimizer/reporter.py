from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import sys


class _TeeStream:
    def __init__(self, *streams):
        self.streams = [stream for stream in streams if stream is not None]

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class _NullStream:
    def write(self, text):
        return len(text)

    def flush(self):
        return None


class OptimizerReporter:
    def __init__(self, *, enabled=True, verbosity=1, log_file=None, append=False, stream=None):
        self.enabled = bool(enabled)
        self.verbosity = int(verbosity)
        self.log_file = log_file
        self.append = bool(append)
        self.stream = sys.stdout if stream is None else stream

    def should_report(self, level=1):
        return self.enabled and int(level) <= self.verbosity

    def print(self, *args, level=1, **kwargs):
        if self.should_report(level):
            print(*args, **kwargs)

    @contextmanager
    def capture_prints(self):
        log_handle = None
        try:
            if self.log_file:
                mode = "a" if self.append else "w"
                log_handle = open(self.log_file, mode, encoding="utf-8")

            if self.should_report():
                target = _TeeStream(self.stream, log_handle) if log_handle is not None else self.stream
            else:
                target = _TeeStream(log_handle) if log_handle is not None else _NullStream()

            with redirect_stdout(target):
                yield self
        finally:
            if log_handle is not None:
                log_handle.close()


def make_reporter(print_report=True, reporter=None, *, verbosity=1, log_file=None, append=False):
    if reporter is None:
        return OptimizerReporter(
            enabled=print_report,
            verbosity=verbosity,
            log_file=log_file,
            append=append,
        )
    if not isinstance(reporter, OptimizerReporter):
        raise TypeError("reporter must be an OptimizerReporter instance")
    return reporter
