#!/usr/bin/env python3
"""Install the shared pins, inheriting the image's CUDA-enabled JAX runtime."""

import os
from pathlib import Path
import subprocess
import sys

VENV_DIR = Path(os.environ.get("DEVCONTAINER_VENV", "/workspaces/.venv"))
PYTHON = VENV_DIR / "bin" / "python"


def main():
    if not PYTHON.exists():
        subprocess.check_call([
            sys.executable, "-m", "venv", "--system-site-packages", str(VENV_DIR)
        ])
    subprocess.check_call([
        str(PYTHON), "-m", "pip", "install", "-r",
        str(Path(__file__).with_name("requirements.txt")),
    ])


if __name__ == "__main__":
    main()
