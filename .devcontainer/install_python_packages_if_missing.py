#!/usr/bin/env python3

import json
import os
import subprocess
import sys
from pathlib import Path

VENV_DIR = Path(os.environ.get("DEVCONTAINER_VENV", "/workspaces/.venv"))
PYTHON = VENV_DIR / "bin" / "python"

BUILD_TOOLS = (
    ("setuptools", "setuptools"),
    ("wheel", "wheel"),
)

PINNED_PACKAGES = (
    # jax/jaxlib are nightly dev builds in the base image; inherited via --system-site-packages
    ("flax", "0.12.7", "flax==0.12.7"),
    ("optax", "0.2.8", "optax==0.2.8"),
    ("numpy", "2.4.6", "numpy==2.4.6"),
    ("pandas", "2.3.3", "pandas==2.3.3"),
    ("matplotlib", "3.10.3", "matplotlib==3.10.3"),
    ("seaborn", "0.13.2", "seaborn==0.13.2"),
    ("pyarrow", "24.0.0", "pyarrow==24.0.0"),
    ("streamlit", "1.47.0", "streamlit==1.47.0"),
    ("plotly", "6.2.0", "plotly==6.2.0"),
    ("boto3", "1.34.162", "boto3==1.34.162"),
    ("redis", "6.4.0", "redis==6.4.0"),
)


def _install_venv_apt_package():
    subprocess.check_call(["sudo", "apt-get", "update", "-qq"])
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    if subprocess.run(
        ["sudo", "apt-get", "install", "-y", "--no-install-recommends", f"python{ver}-venv"]
    ).returncode != 0:
        subprocess.check_call(
            ["sudo", "apt-get", "install", "-y", "--no-install-recommends", "python3-venv"]
        )


def ensure_venv():
    pip = VENV_DIR / "bin" / "pip"

    if PYTHON.exists() and pip.exists():
        return

    if PYTHON.exists() and not pip.exists():
        print(f"Venv at {VENV_DIR} is missing pip, recreating...", flush=True)
        import shutil
        shutil.rmtree(VENV_DIR)
    else:
        print(f"Virtualenv not found at {VENV_DIR}, creating...", flush=True)

    result = subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(VENV_DIR)])
    if result.returncode != 0:
        print("python3-venv not available, installing via apt...", flush=True)
        _install_venv_apt_package()
        subprocess.check_call([sys.executable, "-m", "venv", "--system-site-packages", str(VENV_DIR)])

    if not pip.exists():
        print("Bootstrapping pip into venv...", flush=True)
        subprocess.check_call([str(PYTHON), "-m", "ensurepip", "--upgrade"])

    print(f"Virtualenv ready at {VENV_DIR}", flush=True)


def installed_versions(package_names):
    pip = VENV_DIR / "bin" / "pip"
    result = subprocess.run(
        [str(pip), "show", *package_names],
        capture_output=True, text=True,
    )
    versions = {name: None for name in package_names}
    current_name = None
    for line in result.stdout.splitlines():
        if line.startswith("Name: "):
            current_name = line[6:].strip()
        elif line.startswith("Version: ") and current_name is not None:
            for name in package_names:
                if name.lower() == current_name.lower():
                    versions[name] = line[9:].strip()
            current_name = None
    return versions


def pip_install(specs):
    subprocess.check_call([str(PYTHON), "-m", "pip", "install", *specs])


def main():
    ensure_venv()

    build_names = [name for name, _ in BUILD_TOOLS]
    build_versions = installed_versions(build_names)
    missing_build_tools = [
        spec for name, spec in BUILD_TOOLS if build_versions.get(name) is None
    ]
    if missing_build_tools:
        print("Missing Python build tools:", ", ".join(missing_build_tools), flush=True)
        print("Installing missing Python build tools...", flush=True)
        pip_install(missing_build_tools)
    else:
        print("No missing Python build tools.", flush=True)

    package_names = [name for name, _, _ in PINNED_PACKAGES]
    versions = installed_versions(package_names)
    install_specs = []
    for name, required_version, spec in PINNED_PACKAGES:
        installed_version = versions.get(name)
        if installed_version != required_version:
            status = "missing" if installed_version is None else f"installed {installed_version}"
            print(f"Python package needs install: {spec} ({status}, required {required_version})", flush=True)
            install_specs.append(spec)

    if install_specs:
        print("Installing missing or mismatched Python packages:", ", ".join(install_specs), flush=True)
        pip_install(install_specs)
    else:
        print("No missing or mismatched Python packages.", flush=True)


if __name__ == "__main__":
    main()
