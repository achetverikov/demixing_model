"""Subprocess helpers with optional tee-to-log-file support."""
import os
import re
import select
import subprocess
import sys

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class LogCleaner:
    """Strip ANSI codes and collapse carriage-return redraws for a log file."""

    def __init__(self, log):
        self.log = log
        self.line = []
        self.pending_cr = False

    def write(self, text: str):
        """Strip ANSI escape codes from text and buffer it, collapsing CR overwrite lines.

        Args:
            text: Raw text chunk (may contain ANSI codes and carriage returns).
        """
        text = ANSI_RE.sub("", text)
        for char in text:
            if self.pending_cr:
                if char == "\n":
                    self._flush_line()
                    self.pending_cr = False
                    continue
                self.line = []
                self.pending_cr = False

            if char == "\r":
                self.pending_cr = True
            elif char == "\n":
                self._flush_line()
            else:
                self.line.append(char)

    def close(self):
        """Flush any buffered partial line and flush the underlying log file."""
        if self.line:
            self._flush_line()
        self.log.flush()

    def _flush_line(self):
        """Write the accumulated line buffer to the log with a trailing newline and reset it."""
        self.log.write("".join(self.line) + "\n")
        self.log.flush()
        self.line = []


def announce(title: str, log=None):
    """Print a titled separator banner to stdout and optionally to a log file.

    Args:
        title: Headline text displayed between separator lines.
        log: Optional file-like object to also receive the banner.
    """
    message = f"\n{'=' * 60}\n{title}\n{'=' * 60}\n"
    sys.stdout.write(message)
    sys.stdout.flush()
    if log:
        log.write(message)
        log.flush()


def run(cmd: list, cwd, env, log=None):
    """Run a subprocess, streaming output to stdout and optionally to a log file.

    On POSIX TTYs with a log target the subprocess runs under a PTY so progress
    bars and colors render correctly; ANSI codes are stripped from the log copy.
    Exits the process with the subprocess return code on failure.

    Args:
        cmd: Command and arguments (passed to subprocess.Popen).
        cwd: Working directory for the subprocess.
        env: Environment mapping for the subprocess.
        log: Optional file-like object to receive a clean copy of stdout.
    """
    header = f"\n$ {' '.join(str(c) for c in cmd)}\n{'─' * 60}\n"
    sys.stdout.write(header)
    sys.stdout.flush()
    if log:
        log.write(header)
        log.flush()

    if log and os.name == "posix" and sys.stdout.isatty():
        _run_with_pty(cmd, cwd, env, log)
        sys.stdout.write("\n")
        sys.stdout.flush()
        return

    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if log:
            log.write(line)
            log.flush()
    proc.wait()
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def _run_with_pty(cmd, cwd, env, log):
    """Run a subprocess under a PTY, tee-ing raw bytes to stdout and cleaned text to log.

    Args:
        cmd: Command and arguments list.
        cwd: Working directory for the subprocess.
        env: Environment mapping for the subprocess.
        log: File-like object to receive ANSI-stripped output via LogCleaner.
    """
    import pty
    master_fd, slave_fd = pty.openpty()
    cleaner = LogCleaner(log)
    proc = None
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env,
            stdin=subprocess.DEVNULL,
            stdout=slave_fd, stderr=subprocess.STDOUT,
            close_fds=True,
        )
        os.close(slave_fd)
        slave_fd = None

        while True:
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if master_fd in ready:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                cleaner.write(chunk.decode(errors="replace"))

            if proc.poll() is not None:
                while True:
                    ready, _, _ = select.select([master_fd], [], [], 0)
                    if master_fd not in ready:
                        break
                    try:
                        chunk = os.read(master_fd, 4096)
                    except OSError:
                        break
                    if not chunk:
                        break
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                    cleaner.write(chunk.decode(errors="replace"))
                break
    finally:
        cleaner.close()
        if slave_fd is not None:
            os.close(slave_fd)
        os.close(master_fd)

    proc.wait()
    if proc is None or proc.returncode != 0:
        sys.exit(proc.returncode if proc is not None else 1)
