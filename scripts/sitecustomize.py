"""Auto-loaded by Python on every interpreter startup in this venv (see the
`site` module docs). Sets safe JAX/XLA memory defaults BEFORE jax is
imported by anything else, so ad-hoc scripts and pytest runs are protected
even if the caller forgot to export the vars AGENT.md documents.

This is a unified-memory system (Strix Halo APU, gfx1151): "GPU" memory is
carved out of the same pool as system RAM, and JAX's default behavior
(PREALLOCATE=true, MEM_FRACTION=0.75) grabs a large fraction of it on the
first JAX operation of *any* process, held for that process's entire
lifetime, regardless of actual need. With several concurrent processes
(common when running review/test agents in parallel) this exhausts system
RAM and OOM-kills unrelated programs.

Explicit env vars set by the caller always win (setdefault only fills gaps).

This file is tracked in the repo at scripts/sitecustomize.py — the copy that
actually takes effect lives at .venv/lib/python3.12/site-packages/sitecustomize.py,
which is NOT tracked (it's inside .venv) and is lost on a venv rebuild. Reinstall
it with `bash scripts/install_sitecustomize.sh` after any `uv sync --reinstall`
or fresh `.venv` creation.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
