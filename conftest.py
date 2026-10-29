"""Pytest bootstrap shared by both the TF baseline (.venv-tf) and JAX (.venv).

- Activates the pip-installed ROCm runtime (gfx1151) before JAX imports it, so the
  GPU backend registers without a hand-set LD_LIBRARY_PATH.
- Enables float64 globally (quant finance needs it; JAX defaults to float32).

Both blocks are defensive: if the package isn't installed (e.g. the TF baseline env
has no jax/rocm_sdk), they no-op so TF tests are unaffected.
"""
import os

import pytest

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


@pytest.fixture(autouse=True)
def _reset_unseeded_rng_stream():
    """Give every test the fresh-process unseeded-RNG stream.

    The shim's unseeded `tf.random.normal/uniform` draw from a process-global
    PRNG key (mirroring TF's stateful unseeded RNG, which is likewise
    process-global and order-dependent — see _tf.py). Every earlier test that
    consumes an unseeded draw advances the key, so MC tests with tight
    tolerances (e.g. hull_white zero_coupon_bond_option
    test_correctness_time_dep_simulation, atol=1e-4 on a 500k-sample price)
    flip between pass/fail depending on which tests ran before them — the
    expected prices encode the fresh-process stream. Upstream never saw this
    because Bazel ran each test file in its own process. Resetting the key
    before each test reproduces that isolation inside a single pytest process.
    Guarded: the TF baseline env has no _tf shim.
    """
    try:
        from tf_quant_finance import _tf
        _tf._global_key = _tf.jax.random.PRNGKey(0)
    except Exception:
        pass
    yield
