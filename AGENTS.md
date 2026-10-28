# AGENTS.md — tf-quant-finance (JAX migration)

TF-quant-finance library migrated from TensorFlow to JAX via a shim layer.
Branch `feat/jax-migration`. Read `MIGRATION_STATUS.md` before changing model code;
`BASELINE_TF.md` records the original TF pass/fail baseline (current: 1343 passed / 13 failed).

## Architecture

- `tf_quant_finance/_tf.py` — TF→JAX shim (~1600 lines). All modules import TF as
  `from tf_quant_finance import _tf as tf`; edits to TF-idiom code go through the shim, not per-module rewrites.
- Modules: `black_scholes/` (vanilla, CRR binomial, implied vol, Asian, variance swaps),
  `models/` (Heston, SABR, CIR, Hull-White, HJM, GBM, Longstaff-Schwartz, Euler/Milstein sampling),
  `rates/` (curve bootstrap, Hagan-West, Nelson-Siegel-Svensson), `math/` (PDE, QMC, root search, optimizer, interpolation).
- `conftest.py` enables x64 (quant finance needs float64) + ROCm init.

## Running tests — CRITICAL memory rules

JAX on gfx1151 (Strix Halo, unified memory) pre-allocates from the same pool as system
RAM; defaults grab ~75% and concurrent processes OOM-kill the box. Defenses (both
`setdefault`, explicit export wins): `scripts/sitecustomize.py` auto-installed into the
venv via `bash scripts/install_sitecustomize.sh` (the venv copy is NOT git-tracked), and `conftest.py`. Default MEM_FRACTION=0.4.

```bash
# Full suite (serial, ~8-10 min)
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
  nice -n 19 uv run pytest tf_quant_finance/ -n 1 --tb=no -q -p no:warnings

# Single module
... uv run pytest tf_quant_finance/models/hjm/ -n 1 --tb=no -q -p no:warnings
```

Rules:
- Always `-n 1` (serial) — parallel workers segfault on gfx1151.
- Always `nice -n 19`; always set PREALLOCATE=false.
- NEVER run `tf_quant_finance/experimental/pricing_platform/` — hipSparse GPU crash (ROCm bug).
- Full `models/` suite >15 min — timeout 1200+ or run per-module.

## Environment

- uv-managed (uv.lock); Python 3.12 venv at `.venv/`; JAX 0.9.2 ROCm 7.14 nightly wheels.
- No torch/jax in system python — always go through `uv run`.

## Sibling repos (cross-check effort, 2026-10-26)

Cross-check suite planned at `/var/home/yangye/devv/qfin/pricing/cross-check`.
**quantlib (C++) is the golden standard.** Pairings: quantburn↔quantorch, quantburn↔quantlib,
tf-quant↔quantlib, quantcandle↔quantorch, quantcandle↔quantburn. Backends: ROCm for
torch (quantorch) and jax (here); Vulkan + CPU for Burn and Candle.
This repo's overlap with quantlib: Black-Scholes/vanilla prices, CRR binomial, implied vol,
Asian options, Heston, SABR, CIR, Hull-White, Brownian bridge, math primitives.

## Docs to read first

- `MIGRATION_STATUS.md` — per-module pass rates, known issues
- `MIGRATION_PLAN.md`, `BASELINE_TF.md` — strategy and TF baseline
- `AGENT.md` — earlier version of this file (content folded here)
