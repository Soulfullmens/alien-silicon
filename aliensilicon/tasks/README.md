# tasks

**Track:** Shared core · **Started:** Week 2

Workloads from §13. Each task in `TASKS` (`__init__.py`) has a spec, symbolic input and output shapes, 3 concrete shapes (all must pass), shape constraints for the Z3 bounds proof, a NumPy float64 reference, D1–D4 and S1–S4 input generators, and a naive baseline in `baselines/`. Baselines are written in the DSL and must pass the judge's allow-list as they are.

| ID | Status |
|---|---|
| W1 GEMM | baseline `gemm_naive` (64×64 tiles, K step 32) |
| W2 row softmax | baseline `softmax_naive` (§8.4) |
| W3 attention | baseline `attention_naive`, non-causal, d = 64; causal still to do |
| W4–W7 | later |

`check(out, ref)` is the correctness rule. τ = 1e-3 is locked.
