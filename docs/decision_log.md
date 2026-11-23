# Decision log

Newest first. One entry per decision that changes the spec or that a reviewer might ask about.

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
