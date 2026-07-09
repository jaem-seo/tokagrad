"""Install Google DeepMind fusion_surrogates into the current environment.

Usage:
  python scripts/install_fusion_surrogates.py

This simply runs:
  python -m pip install fusion_surrogates

The package is optional. TokaGrad can run without it using other models.
"""
import subprocess
import sys

cmd = [sys.executable, "-m", "pip", "install", "fusion_surrogates"]
print("Running:", " ".join(cmd))
raise SystemExit(subprocess.call(cmd))
