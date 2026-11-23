# Red-team report: Weeks 0–2 code

**Date:** 2026-11-23. **Scope:** Tile IR + DSL, judge allow-list, interpreter and tasks, as of commit `595d0d8`.

**Method:** attack every trust boundary with a working exploit, fix what breaks, and add a regression test per attack (`tests/test_red_team.py`). Then list what is still weak and what would make the engine stronger.

**The judge has to be stronger than the agent.** §14 promises a judge the agent cannot cheat. Every cheat that gets through becomes a fake "discovery" in the paper. The cost model (Weeks 3–5) is coming soon, and it will reward any hole that makes a kernel look faster.

## 1. Attacks that worked, and their fixes

| # | Attack | Severity | What happened | Fix | Test |
|---|---|---|---|---|---|
| A1 | Shell command hidden in a **string annotation**: `def k(X: "...__subclasses__()...system('touch …')")` | **Critical: full sandbox escape** | A string is an allowed `Constant` node, so the allow-list accepted it. The DSL then resolved annotations with `inspect.signature(eval_str=True)`, which `eval()`s the string. The only thing that stopped it was a `from __future__ import annotations` line in `static_check.py`, which `compile()` passed on to the submission **by accident**. Without that flag, the exploit wrote a file (confirmed). | (1) The allow-list accepts only the literal form `GMEM[dtype, (dims…)]`. (2) The DSL never evaluates annotation strings. (3) `compile(..., dont_inherit=True)`. | `test_a1_*` |
| A2 | `alloc(RMEM, (100000, 100000), f32)` | High: DoS | MemoryError (37 GB) | At most 2²⁰ elements per alloc | `test_a2_*` |
| A3 | Two nested `serial(100000)` loops | High: DoS | The interpreter never finished | Work budget: each op costs a fixed overhead plus the elements it produces (plus m·n·k for `mma`). A hostile kernel stops in about 2 s. Grid capped at 65,536 CTAs. | `test_a3_*` |
| A4 | CTA 0 zeroes all of `Y`, then every CTA writes its own row | High: wrong on real GPUs | Correct in our serial grid order, and **passed**. On a GPU, CTAs run in any order, so rows get wiped. | Ownership tracking per launch: no GMEM element may be written by two CTAs, or read by one CTA after another wrote it. A shuffled-order rerun stays as a second check. | `test_a4_*` |
| A5 | Kernel overwrites its own input as scratch space | Medium: cheat | Not flagged | `judge/correctness.py` checks that inputs are byte-identical after the run | `test_a5_*` |
| A6 | Hand-written IR JSON: unknown op, `fast_exp`, dangling refs, lying about a dtype (a free hidden convert), extra attrs (`coalesced: true`), wrong unit, bad grid axis, deep nesting | **High: skips every builder check** | Raw KeyError/IndexError/RecursionError crashes. The dtype lie was **accepted silently**. | `judge/ir_check.py` rebuilds the IR through the DSL and accepts it only if the result is byte-identical. Size cap 2 MB. All errors become `Rejected`. | `test_a6_*` (10 tampers) |
| A7–A9 | `1e999` literal, `program_id(5)` on a 1-D grid, a 20-digit int literal, shape dims 0 or −4 | Low: judge crashes | Raw ValueError/IndexError/OverflowError | The builder checks literals (finite, fit in i32), grid axes and shape dims. Any raw Python error during the build becomes `Rejected`. | `test_a7_a9_*`, `test_raw_python_errors_*` |
| B1 | GEMM with **every `barrier()` removed** | **High: wrong on real GPUs, and the cost model would reward it** | **Passed all tests.** Threads run as NumPy vector lanes, so a missing barrier can never fail in the interpreter. | Shared-memory hazard tracking: reading SMEM/TMEM written since the last barrier, writing SMEM read since the last barrier, or touching an async copy's destination before `wait` raises `race:` | `test_b1_*` |
| B2 | Softmax **without max-subtraction** | Medium: hides an overflow bug | **Passed all D-tests**, because inputs never exceed 3. Gives NaN on inputs ×100. | Added S-tests S1–S4 (constant rows, zero blocks, large values, sorted with alternating signs). S3 catches it. | `test_b2_*` |
| T1 | All-zero softmax output | High: cheat | Passed the original rule \|err\| ≤ τ(1+\|ref\|) | The rule is now \|err\| ≤ τ(\|ref\| + max\|ref\|) (Week 2 decision log) | `test_reference_catches_a_wrong_kernel` |

Attacks that failed (the code held): import tricks, `__class__` attribute chains, builtins (`open`, `eval`), lambdas and comprehensions, `global`, `while`, `try`, nested functions, rebinding DSL names, `**kwargs`, f-strings, bytes, default args, top-level code (`tests/test_static_check.py`, 25 cases).

## 2. Still weak: known and not fixed yet

Ranked by how much each threatens the paper's claims.

1. **The judge still runs agent code in-process.** The allow-list is the only barrier, and A1 shows a single gap is enough. *Change:* build the DSL inside the no-network, read-only Docker sandbox (§22), and send only IR JSON to `judge/ir_check.py`, which never runs agent code. **Do this before the first agent run.**
2. **The simulator is more accurate than real hardware.** `mma` and `fma` compute in float64 and round once, but real tensor cores round, and sometimes truncate, at each k-step. A kernel that relies on that extra accuracy (long-K accumulation in f16) passes here and fails on a T4. *Change:* make MMA accumulation precision and rounding mode part of the chip profile (Week 3), then measure them on the T4 (Week 9).
3. **The tolerance accepts dropping small values.** On a peaked softmax, zeroing every probability below 1e-4 (92% of entries) still passes, because the bound scales with max\|ref\|. That's real block sparsity, an approximation, not plain correctness. **This is your decision before freezing the pre-registration.** Is it a legitimate trick (count it as a discovery) or a hack (add a per-row or relative-L2 criterion)?
4. **Each task tests only one shape.** A kernel can hard-code 128×128 and pass. §14.1 S5 asks for non-multiple shapes. *Change:* each task gets 2–3 shapes, including one that isn't a multiple of 16/64, and all of them must pass.
5. **Bounds are checked only for the inputs we test.** Out-of-bounds access is caught at run time for the tested shapes. Proving it for all shapes is the Z3 step in §14.1. *Change:* make index expressions affine (they already are, through `iadd`/`imul`) and prove bounds symbolically.
6. **Race tracking is conservative and per tile.** It works on whole tiles, so two disjoint halves of one SMEM tile written by different threads count as a conflict, and a real race between different elements of one RMEM tile isn't tracked at all. That's fine for the baselines, but it could produce false alarms on clever kernels. *Change:* track by region (slice ranges) once a real kernel needs it.
7. **Results are bit-identical on one machine, not necessarily across machines.** `np.exp2`/`np.sin` use libm/SIMD code paths that differ by platform and NumPy version. *Change:* pin NumPy in a lockfile and record `numpy.__version__`, plus a hash of the interpreter and judge, in every result (§14.3 asks for the judge hash).
8. **Reduction order is NumPy's pairwise order**, not the chip's tree order. That matters for f16 sums. *Change:* model the tree order per scope (warp shuffle, CTA tree).
9. **No white-box adversary (WB) yet.** The fixed D and S tests can still be learned by a kernel that branches on its input. The WB adversary in §14.1 is the real defence, together with fresh seeds at audit time.
10. **The interpreter is slow**, about 10 µs per op in Python. It's fine for the baselines, but agent search will run thousands of kernels. *Change:* vectorize per grid (one NumPy op over all CTAs at once), or JIT the hot ops with Numba.

## 3. What would make it more powerful

These go beyond fixing holes. Each turns a "we tested it" claim into a stronger one, and some are publishable on their own.

- **Judge mutation score.** Automatically mutate each honest baseline: drop a barrier, shift an index by 1, drop the max-subtraction, change a scale, swap `exp2` for the identity. Then count how many mutants the judge catches. That gives one number for how strong the judge is, it guides the hacker–fixer loop in §15, and it belongs in the paper. Today's B1 and B2 would have been found automatically.
- **Translation validation.** For small shapes, prove that a kernel equals its reference symbolically (SMT over reals, or equality saturation with e-graphs). Testing can only show bugs. A proof shows there are none, and it would make Level-C (algebraic) discoveries checkable by machine.
- **Rigorous floating-point error bounds.** Push interval or affine arithmetic through the IR to compute a worst-case error bound for each kernel. Then "correct" means within a proven bound, not just inside τ on 8 inputs. That would answer reviewer worry (3) above on principle.
- **Differential fuzzing of the simulator.** Generate random valid IR programs with Hypothesis, run them in the interpreter and in a plain NumPy reference, and later in the CUDA backend. That's the Csmith idea for our own toolchain: bugs in the simulator are bugs in every result.
- **Race detection as a feature.** The hazard tracker added here already catches races that are invisible to plain testing. Extended to regions and warps, it could check agent kernels before they ever reach a GPU, and also check hand-written TileLang kernels.
- **Numerics driven by the chip profile.** MMA accumulator width, MUFU error distribution, FTZ/denormal handling and rounding modes would all come from the chip profile. Alien chips then differ in numerics as well as speed, a dimension no other benchmark has.
- **Content-addressed provenance.** Every result record holds the IR hash, judge hash, interpreter hash, NumPy version and seed. Any number in the paper can then be reproduced from its hash.

## 4. Changes made in this round

- `judge/static_check.py`: strict annotation shape, `dont_inherit=True`, raw build errors become `Rejected`.
- `judge/ir_check.py` (new): IR JSON accepted only if the DSL rebuilds it byte-identically.
- `judge/correctness.py` (new): D1–D4 and S1–S4 tests. Each runs twice, with different garbage memory and CTA order. It checks that inputs are unchanged, that both runs are bit-identical, and the tolerance.
- `aliensilicon/ir/dsl.py`: never evaluates annotations. Checks literals, shapes, loop bounds and grid axes.
- `aliensilicon/interp`: tile and grid caps, work budget, shuffled CTA order, separate random streams for garbage and MUFU, SMEM barrier and async hazard tracking, GMEM ownership tracking.
- `aliensilicon/tasks`: S1–S4 inputs, with a per-task S3 scale.
- `tests/test_red_team.py`: a regression test for every attack above.
