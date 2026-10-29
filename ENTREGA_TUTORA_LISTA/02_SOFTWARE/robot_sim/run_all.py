"""Run the current headless CartPole verification safely.

The old A/B/C/D launcher depended on a removed ``verify_cartpole.py`` and
edited source files in place to create its variants. Use the maintained
``verify_cartpole_2000.py`` entry point instead.
"""

import subprocess
import sys
from pathlib import Path


def main():
    script_dir = Path(__file__).resolve().parent
    verification = script_dir / "verify_cartpole_2000.py"
    if not verification.is_file():
        raise FileNotFoundError(f"CartPole verification script not found: {verification}")

    print(f"Running CartPole verification: {verification.name}", flush=True)
    subprocess.run(
        [sys.executable, str(verification)],
        cwd=script_dir,
        check=True,
    )


if __name__ == "__main__":
    main()
