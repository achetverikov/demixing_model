import subprocess
import sys


def test_config_import_does_not_initialize_jax():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import shared.config; assert 'jax' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
