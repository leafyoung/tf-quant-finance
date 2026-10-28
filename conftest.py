"""Pytest bootstrap shared by both the TF baseline (.venv-tf) and JAX (.venv).

- Activates the pip-installed ROCm runtime (gfx1151) before JAX imports it, so the
  GPU backend registers without a hand-set LD_LIBRARY_PATH.
- Enables float64 globally (quant finance needs it; JAX defaults to float32).

Both blocks are defensive: if the package isn't installed (e.g. the TF baseline env
has no jax/rocm_sdk), they no-op so TF tests are unaffected.
"""
import os

# Safe JAX/XLA memory defaults — see the OOM postmortem in AGENT.md.
# This is a unified-memory system (Strix Halo, gfx1151): "GPU" memory is
# carved out of the same pool as system RAM. JAX's own defaults
# (PREALLOCATE=true, MEM_FRACTION=0.75) grab a large fraction of total RAM
# on the very first JAX op of any process and hold it for that process's
# whole lifetime — fine for one process, an OOM risk for several run
# concurrently. Must be set before `import jax`. setdefault() so an
# explicit export (e.g. a deliberately larger MEM_FRACTION for one heavy
# run) always wins. Redundant with .venv's sitecustomize.py (belt and
# braces: this still protects test runs if that file is ever lost to a
# venv rebuild).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")


def _try_init_jax_rocm():
    try:
        import rocm_sdk
        rocm_sdk.initialize_process()
    except Exception:
        pass  # not a JAX env, or rocm_sdk absent — fine
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass  # not a JAX env


_try_init_jax_rocm()
