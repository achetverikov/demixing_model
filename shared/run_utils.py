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
        if self.line:
            self._flush_line()
        self.log.flush()

    def _flush_line(self):
        self.log.write("".join(self.line) + "\n")
        self.log.flush()
        self.line = []


def announce(title: str, log=None):
    message = f"\n{'=' * 60}\n{title}\n{'=' * 60}\n"
    sys.stdout.write(message)
    sys.stdout.flush()
    if log:
        log.write(message)
        log.flush()


def run(cmd: list, cwd, env, log=None):
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
