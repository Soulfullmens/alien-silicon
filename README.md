# Alien Silicon

An AI engine that writes fast operator code (GEMM, attention, softmax, norms) for chips it has never seen. A deterministic judge it cannot cheat checks every kernel. The same engine tells chip designers which design runs AI models best before the chip is built.

The full plan is in [`docs/master_plan.md`](docs/master_plan.md) (v2.3).

## Status

**Week 2 of Phase 1 (Shared Core).** Guardrail 1 applies: no AMD and no agent work until Gates 1 and 2 pass.

| Done | Where |
|---|---|
| Tile IR structures, `@kernel` / `@program` DSL, JSON round trip, SHA-256 | `aliensilicon/ir/` |
| AST allow-list for agent source (judge step 1) | `judge/static_check.py` |
| Paper-check kernels: row copy, naive softmax (§8.4), naive GEMM | `aliensilicon/tasks/baselines/` |
| IR reference with the TileLang mapping | `docs/ir_reference.md` |
| NumPy interpreter: exact f16/f32 rounding, seeded garbage memory and MUFU error, deterministic | `aliensilicon/interp/` |
| W1 GEMM, W2 softmax, W3 attention: spec, float64 reference, naive baseline, correct at τ on D1–D4 and S1–S4 | `aliensilicon/tasks/` |
| Judge: IR JSON replay check, hidden tests, race and input-mutation detection | `judge/` |
| Red-team round 1: 13 working attacks found and fixed, each with a regression test | `docs/red_team.md` |

## Which folder belongs to which track

| Track | Folders |
|---|---|
| **Shared core** (Phase 1, weeks 1–12) | `aliensilicon/{ir,interp,cost,chips,physics,tasks}`, `judge/`, `backends/cuda/` |
| **Shared engine** (Phase 2) | `agent/`, `workspace/`, `ledger/`, `runner/`, `configs/` |
| **Track A: science** (C2 benchmark, C3 paper, C6 T4 kernels) | `aliensilicon/{search,taxonomy,bench}`, `analysis/`, `kernels_t4/` |
| **Track B: product** (C4 Studio, C5 AMD demo). Blocked until Gate 2 | `aliensilicon/studio/`, `backends/tilelang/` |
| Both (C7 playground) | `notebooks/` |

Every folder that is still empty has a README saying what goes there and when.

## Quick start

```bash
make install   # pip install -e ".[dev]"
make test
make lint
```

Build a kernel the way the judge will see it:

```python
from pathlib import Path
from judge.static_check import build_submission

k = build_submission(Path("aliensilicon/tasks/baselines/softmax_naive.py").read_text())["softmax_naive"]
print(k.sha256())
print(k.to_json())
```
