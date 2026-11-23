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

## 2. Weak points, and their status after round 2

| # | Weak point | Status |
|---|---|---|
| 1 | Agent code built in the judge's own process | **Fixed.** `judge/sandbox.py` builds in a child process with its own network namespace and an unprivileged uid (nobody). File writes, forking, CPU (10 s), memory (2 GB), wall time and output size are all capped, and the environment is empty. Only IR JSON comes back, and `ir_check` re-validates it, so the parent never runs agent code. Tested: network, data writes, fork, CPU and memory are all contained (`tests/test_hardening.py`). Production adds `runner/sandbox/Dockerfile` (read-only fs, `--network none`, `--cap-drop ALL`). **Remaining:** the child can read files and create empty files, which Docker's `--read-only` closes. |
| 2 | Simulator more accurate than hardware | **Mechanism built.** `mma_k_chunk=c` rounds the accumulator after every c-wide slice of K, like a tensor core. The judge takes it as a parameter, and the Week 3 chip profile will set it (T4: k = 8). **Finding:** with τ = 1e-3, f16 accumulation still passes for K ≤ 128, so on these sizes it is a legitimate speed trick and not a hack. |
| 3 | Tolerance accepts dropping small softmax values | **Still your decision** (see the decision log). Nothing changed. |
| 4 | One shape per task | **Fixed.** Each task has 3 shapes (for example GEMM 128³, 64×192×96, 192×64×32; softmax cols 256/200/1000; attention seq 256/128/48), and every test runs on every shape (24 runs per kernel). A kernel with a hard-coded shape fails. GEMM and attention shapes stay multiples of their tile sizes until the IR gets a masked load. |
| 5 | Bounds checked only for the tested inputs | **Fixed.** `judge/bounds.py` uses Z3 to prove that every index and slice is in bounds and that source and destination extents match, for **every** shape that satisfies the task's constraints. All baselines are proven. A hard-coded loop count or an off-by-one slice gets a concrete counterexample shape. Indices that depend on data come back "unproven", and the runtime checks cover those. |
| 6 | Race tracking per whole tile | **Fixed.** Hazards are tracked per element, so disjoint regions of one tile never conflict. |
| 7 | Bit-identical only on one machine | **Partly fixed.** Every result records a judge hash (over judge, IR, interpreter and task code), plus the NumPy and Python versions and the sandbox capabilities. **Remaining:** pin versions in a lockfile before the main runs. |
| 8 | NumPy reduction order | Not done. Needs the chip's tree order, which comes with the chip profile. |
| 9 | No white-box adversary | Not done. Needs the agent/LLM layer in Phase 2. |
| 10 | Slow interpreter | Not done. Each kernel takes 0.1–0.3 s per full judge run today, which is fine for Phase 1. |

## 2b. Round 2 additions

- **Judge mutation score** (`python -m judge.mutate`): each baseline is broken in small ways (delete a statement, bump an index constant, scale a float, swap mufu/reduce/cmp ops, swap `sub` operands, drop `init=`/`transpose_b=`/`keepdims=`), and we count how many mutants the judge catches. **Full judge: 70/70 killed (100%).** The Week 2 judge (D1 only, one shape) misses 4 of them: the missing max-subtraction, max→min swapped (twice), and a 1% scale change in attention. So the score really does tell a strong judge from a weak one.
- **One pipeline:** `judge.evaluate(source, task)` runs sandbox build → IR replay → Z3 bounds → 24 hidden tests, and returns a result record with provenance.
- **Output shape check:** a kernel that declares the wrong output shape used to compare via NumPy broadcasting. Now it fails.

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
