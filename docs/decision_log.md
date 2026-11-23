# Decision log

Newest first. One entry per decision that changes the spec or that a reviewer might ask about.

## 2026-11-23 · Hardening round 2 (see `red_team.md` §2)

1. **Agent source is built only in the sandbox** (`judge/sandbox.py`). The judge process never executes it.
2. **Z3 bounds proof is judge step 2.** A counterexample rejects the kernel. "Unproven" (data-dependent indices or a timeout) falls back to the runtime checks.
3. **Every task has 3 shapes, and all must pass.** Tasks declare output shapes and shape constraints (for example `M % 64 == 0`). Grid functions use `tasks.div` so the same lambda works on ints and on Z3 symbols.
4. **`mma_k_chunk`** models per-k-step accumulator rounding. Its value is a chip property, set in Week 3.
5. **Result records carry provenance:** judge hash, IR hash, NumPy and Python versions, sandbox capabilities.
6. **Still open for you:** the softmax small-probability question (round 1, item 6).

## 2026-11-23 · Red-team round 1 (see `red_team.md`)

1. **Annotations must be the literal `GMEM[dtype, (dims…)]`.** String annotations allowed a sandbox escape (A1). `from __future__ import annotations` is not supported in kernel code.
2. **The judge trusts only IR JSON that the DSL rebuilds byte-identically** (`judge/ir_check.py`). The interpreter assumes validated IR.
3. **Synchronization is checked, not assumed.** SMEM/TMEM read-after-write, write-after-read, and async copies used before `wait`, all without a `barrier()`, are errors. So is a GMEM element written by two CTAs, or read after another CTA wrote it, within one launch. This makes the IR's synchronization semantics stricter than "whatever the interpreter happens to do".
4. **Hidden tests are now D1–D4 plus S1–S4.** S5 (other shapes) is still to do. The S3 scale is per task: GEMM ×10, softmax and attention ×100.
5. **Resource caps:** 2²⁰ elements per alloc, 65,536 CTAs, and a work budget of about 2 s of interpretation per run.
6. **Open question for you:** should dropping softmax probabilities below about 1e-4 count as correct? See red_team.md §2.3.

## 2026-11-23 · Week 2 interpreter

1. **Correctness rule (needs your sign-off before the pre-registration is frozen):** pass iff every output is finite and |out − ref| ≤ τ·(|ref| + max|ref|), with τ = 1e-3. A fixed absolute term of τ let a softmax that was 1% wrong, or even all zeros, pass, because softmax values are about 1/cols ≈ 0.004. The rule above scales with the output and fails both cases (`tests/test_interp.py::test_reference_catches_a_wrong_kernel`).
2. **`mma(..., transpose_b=True)`.** Attention needs Q·Kᵀ, and the IR has no transpose. This maps 1-to-1 to `T.gemm(transpose_B=True)`.
3. **NumPy broadcasting for elementwise ops, plus `reduce(..., keepdims=True)`.** A row max of shape `(rows, 1)` can then be subtracted from a `(rows, cols)` tile. Symbolic dims must match exactly.
4. **Numeric semantics follow PTX where CUDA defines them:** float→int conversion saturates, and shift amounts clamp to the type width. See `docs/ir_reference.md`.
5. **W3 baseline is non-causal with head dim 64.** Causal needs per-element row/column indices, and the IR has no iota primitive yet. That gets decided when causal W3 is built.

## 2026-11-23 · Week 0/1 setup

1. **Repo layout follows I-15.4 exactly.** Track B folders (`aliensilicon/studio/`, `backends/tilelang/`) exist only as READMEs because Guardrail 1 blocks that work until Gate 2.
2. **Added `alloc(..., init=x)`.** The interpreter fills new buffers with garbage (§9), so a GEMM accumulator needs a way to start at zero. It maps 1-to-1 to TileLang `T.clear`/`T.fill`, and it is generic, not a trick. This is a small addition to §8.3 group 1.
3. **`load`/`store` never convert dtypes.** §8.4 loads f16 into an f32 tile. We load into f16 and call `convert`, so every conversion is a visible op with a cost.
4. **Loops use `for i in serial(n):`, not Python `range`.** The body is traced once into a single `for` op, which keeps the IR small and the loop rolled. Python `for`/`range` is rejected by the allow-list.
5. **Spec names that clash with Python** use a trailing underscore: `and_`, `or_`, `if_`. `max`, `min` and `abs` keep their spec names inside kernel code, because the allow-list blocks the builtins anyway.
6. **Comparison masks get dtype `bool`.** Only `cmp` creates it, and it isn't a valid `alloc` dtype.
7. **Allow-list scope (judge step 1).** No attribute access, no names starting with `_`, no imports except DSL names from `aliensilicon.ir`, no builtins, no `while`/`try`/lambda/comprehensions/arithmetic operators. Index arithmetic goes through `iadd`/`imul`, so the cost model sees it.
8. **Shortcut:** `build_submission` runs the allow-listed source in-process. It must move into the no-network Docker sandbox (§22) before any agent run.
