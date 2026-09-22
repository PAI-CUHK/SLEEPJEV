# Installation

SLEEPJEV requires Python 3.10, 3.11, or 3.12. Install a machine-compatible PyTorch wheel when using CUDA or another accelerator.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,dev]"
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test,dev]"
```

Use `.[sleep]` for EDF/XML utilities, `.[train]` for the published training data extras, and `.[all]` for the complete development environment. Verify with `python -m sleepjev demo` and `pytest -q`.
