"""
Tee class for splitting output to multiple streams.

This module provides a file-like object that redirects output to multiple streams simultaneously.
"""

import time


class RateLimitedTee:
    """
    A custom file-like object that rate-limits progress bar updates to log file.

    This prevents massive log files when tqdm progress bars update every 0.1s.
    Terminal output is NEVER rate-limited - only log file writes.
    """

    def __init__(self, terminal_stream, log_file, log_interval_seconds=600):
        """
        Initialize with terminal and log file streams.

        Args:
            terminal_stream: Stream for terminal output (always written immediately)
            log_file: File for log output (rate-limited for progress bars)
            log_interval_seconds: Minimum seconds between progress bar log writes (default: 600 = 10 minutes)
        """
        self.terminal = terminal_stream
        self.log_file = log_file
        self.log_interval = log_interval_seconds
        self.last_progress_log_time = 0
        self.last_progress_line = ""

    def _is_progress_bar_line(self, text):
        """Detect if text is a tqdm progress bar update."""
        if not text:
            return False
        # Progress bars typically contain:
        # - Carriage returns (\\r) for in-place updates
        # - Progress bar characters (█, ▏, %, |)
        # - Rate indicators (it/s, roads/s, climbs/s)
        progress_indicators = ['\\r', '█', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '%|', '/s]', 'it/s', 'roads/s', 'climbs/s', 'segs/s']
        return any(indicator in text for indicator in progress_indicators)

    def write(self, obj):
        """
        Write output to terminal immediately, rate-limit progress bars to log file.

        Args:
            obj: Object to write (typically a string)
        """
        # ALWAYS write to terminal immediately (no rate limiting for user)
        self.terminal.write(obj)
        self.terminal.flush()

        # For log file: rate-limit progress bar updates
        if self._is_progress_bar_line(obj):
            current_time = time.time()
            time_since_last_log = current_time - self.last_progress_log_time

            # Only log progress bars every N seconds
            if time_since_last_log >= self.log_interval:
                self.log_file.write(obj)
                self.log_file.flush()
                self.last_progress_log_time = current_time
                self.last_progress_line = obj
            # else: skip logging this progress update (terminal still shows it)
        else:
            # Non-progress output (errors, messages, etc.) - always log immediately
            self.log_file.write(obj)
            self.log_file.flush()

    def flush(self):
        """Flush both streams."""
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        """Return True if terminal stream is a TTY (needed for tqdm)."""
        return hasattr(self.terminal, "isatty") and self.terminal.isatty()

    def fileno(self):
        """Return the fileno of the terminal stream (needed for tqdm)."""
        if hasattr(self.terminal, "fileno"):
            try:
                return self.terminal.fileno()
            except (OSError, ValueError):
                pass
        raise AttributeError("Terminal stream has no fileno()")


class Tee:
    """
    A custom file-like object that redirects output to multiple streams.

    This is useful for logging output to both console and file simultaneously.
    """

    def __init__(self, *files):
        """
        Initialize with multiple file streams.

        Args:
            *files: Variable number of file-like objects to write to
        """
        self.files = files

    def write(self, obj):
        """
        Write the output to every file/stream provided.

        Args:
            obj: Object to write (typically a string)
        """
        for f in self.files:
            f.write(obj)
            f.flush()  # Ensure immediate writing

    def flush(self):
        """Flush the buffer for every file/stream."""
        for f in self.files:
            f.flush()
