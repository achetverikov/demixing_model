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


def ensure_venv():
    if PYTHON.exists():
        return

    print(f"Virtualenv not found at {VENV_DIR}, creating...")
    result = subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(VENV_DIR)])
    if result.returncode != 0:
        print("python3-venv not available, installing via apt...")
        subprocess.check_call(
            ["sudo", "apt-get", "install", "-y", "--no-install-recommends", "python3-venv"]
        )
        subprocess.check_call([sys.executable, "-m", "venv", "--system-site-packages", str(VENV_DIR)])

    print(f"Created virtualenv at {VENV_DIR}")


def installed_versions(package_names):
    code = "\n".join((
        "import importlib.metadata as metadata",
        "import json",
        "import sys",
        "versions = {}",
        "for name in sys.argv[1:]:",
        "    try:",
        "        versions[name] = metadata.version(name)",
        "    except metadata.PackageNotFoundError:",
        "        versions[name] = None",
        "print(json.dumps(versions))",
    ))
    output = subprocess.check_output(
        [str(PYTHON), "-c", code, *package_names],
        text=True,
    )
    return json.loads(output)


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
        print("Missing Python build tools:", ", ".join(missing_build_tools))
        print("Installing missing Python build tools...")
        pip_install(missing_build_tools)
    else:
        print("No missing Python build tools.")

    package_names = [name for name, _, _ in PINNED_PACKAGES]
    versions = installed_versions(package_names)
    install_specs = []
    for name, required_version, spec in PINNED_PACKAGES:
        installed_version = versions.get(name)
        if installed_version != required_version:
            status = "missing" if installed_version is None else f"installed {installed_version}"
            print(f"Python package needs install: {spec} ({status}, required {required_version})")
            install_specs.append(spec)

    if install_specs:
        print("Installing missing or mismatched Python packages:", ", ".join(install_specs))
        pip_install(install_specs)
    else:
        print("No missing or mismatched Python packages.")


if __name__ == "__main__":
    main()
