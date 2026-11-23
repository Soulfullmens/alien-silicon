# Tile IR reference (v1)

The spec is §8 and I-5 of `master_plan.md`. This page records what the code in `aliensilicon/ir/` actually does. Update it whenever the IR changes, and bump `IR_VERSION` in `core.py`.

## Serialized form

- `Kernel.to_json()` gives canonical JSON (sorted keys, no whitespace, no NaN). `Kernel.sha256()` hashes that string.
- Top-level fields: `ir_version`, `type` (`kernel` or `program`), `name`, `params`, `grid`, `threads`, `values`, `body`, `meta`.
- `values[i] = {id, dtype, shape, space}`. Shape dims are ints or symbolic names (`"rows"`) that are bound at launch.
- `Op = {kind, inputs, outputs, attrs, unit, blocks}`. An input is a value ref `"%3"`, a number, or `{"slice": [start, stop]}`. `blocks` holds the bodies of `for` and `if`.
- Ops that write into an existing tile (`load`, `store`, `async_copy`, `mma`, `gather_load`) list that tile in `outputs`. Indexing goes in `attrs.src_index` and `attrs.dst_index`.

## Primitives

| # | DSL | IR `kind` | `unit` | TileLang (I-5) |
|---|---|---|---|---|
| 1 | `alloc(space, shape, dtype, init=, pad=, swizzle=, stride=)` | `alloc` | none | `T.alloc_shared` / `T.alloc_fragment` / `T.alloc_local`; `init` → `T.clear` / `T.fill` |
| 2 | `load(src, dst)`, `store(src, dst)` | `load`, `store` | GMEM / TMEM / SMEM | `T.copy` |
| 3 | `async_copy(src, dst, group)`, `wait(group)` | `async_copy`, `wait` | COPY, control | `T.Pipelined(..., num_stages=k)` |
| 4 | `mma(a, b, acc, transpose_b=False, clear=False)` (acc += a @ b or a @ bᵀ; clear overwrites) | `mma` | MMA | `T.gemm(..., transpose_B=, clear_accum=)` |
| 5 | `add sub mul fma max min neg abs` | same | FMA | expressions in `T.Parallel` |
| 6 | `mufu(op, x)`, op ∈ exp2 log2 rcp rsqrt sin cos | `mufu` + `attrs.op` | MUFU | `T.exp2`, `T.log2`, … |
| 7 | `iadd isub imul and_ or_ xor shl shr` | `iadd` … `and` `or` … | INT (`imul` → IMUL) | TIR integer ops |
| 8 | `bitcast(x, dt)` (equal width), `convert(x, dt)` | same | INT, FMA | TIR reinterpret / cast |
| 9 | `cmp(op, a, b)` → bool, `select(mask, a, b)` | same | FMA | `T.if_then_else` |
| 10 | `reduce(op, x, axis, scope, keepdims=False)`, op ∈ max min sum | `reduce` | REDUCE (thread scope → FMA) | `T.reduce_max` / `T.reduce_sum` |
| 11 | `for i in serial(n, unroll=False):` | `for` | control | `T.serial` / `T.unroll` |
| 12 | `with if_(cond):` | `if` | control | TIR `if` |
| 13 | `barrier()` | `barrier` | control | implicit in TileLang / `T.sync_threads` |
| 14 | `program_id(axis)`, `num_programs(axis)` | same | none | block indices of `T.Kernel(...)` |
| 15 | `gather_load(src, indices, dst)` | `gather_load` | GMEM, `coalesced: false` | indexed `T.copy` |
| 16 | `@program` + `buffer(shape, dtype)` + kernel calls | `Program.launches` | none | TileLang kernels called in sequence |

**The TileLang names above have not been checked against the current TileLang docs yet.** Week 1 task: check them and fix this table.

## Rules the builder enforces

- Elementwise ops need one dtype across their tile operands. Shapes follow NumPy broadcasting (a row max kept as `(rows, 1)` broadcasts over `(rows, cols)`), and symbolic dims must match exactly. Number literals take the tile's dtype. Float ops reject int tiles, and int ops reject float tiles.
- `load` and `store` do not convert. Load into a tile of the source dtype, then `convert` it (see `softmax_naive.py`).
- `alloc` can't create GMEM. GMEM comes from kernel params or `buffer()` in a `@program`.
- Kernels return nothing. Outputs are written with `store`.
- `for` bodies are traced once and kept as one rolled loop op. `unroll` is only a mark.

## Interpreter semantics (`aliensilicon/interp`)

- f16 and f32 ops run in their own dtype, rounded to nearest-even after every op. `fma` and `mma` compute in float64 and round once to the result dtype, so `mma` accumulates in whatever dtype the accumulator tile has.
- New SMEM/RMEM/TMEM tiles, and GMEM outputs not passed as inputs, hold seeded random bits (NaN and Inf included). A kernel that skips writing an output, or reads memory it never set, gives different results for different seeds.
- `mufu` is exact unless `mufu_ulp_error` is set. Then it adds seeded uniform error of up to that many ulps.
- Integer ops wrap. `shl` and `shr` clamp the shift amount to the type width, as in PTX. Float→int `convert` truncates toward zero, saturates, and maps NaN to 0. Int→int narrowing wraps.
- Any out-of-bounds index or slice raises `InterpError`. `gather_load` clamps its indices instead (group 15).
- `reduce(sum)` uses NumPy pairwise order in the tile dtype. This may need to match the chip's tree order later.
- `barrier`, `wait` and `async_copy` run immediately. Race and hazard modelling is the cost model's job.

## Synchronization rules (checked by the interpreter)

- Reading an SMEM/TMEM tile written since the last `barrier()` is a race. So is writing one that was read since the last barrier. Allocating with `init` counts as a write.
- An `async_copy` destination can't be read or written until `wait(group)`, and then only after a `barrier()`.
- Within one launch, a GMEM element may be written by at most one CTA, and no CTA may read an element another CTA wrote. Cross-CTA communication goes through separate launches in a `@program`.

## Judge entry points

- `judge.evaluate(source, task, mufu_ulp_error=, mma_k_chunk=)`: the full pipeline. Returns a record with `correct`, `stage`, `bounds`, `tests` and `provenance`.
- `judge.sandbox.build_in_sandbox(source)`: builds agent DSL source in an isolated child, never in-process.
- `judge.static_check.build_submission(source)`: the AST allow-list plus the build. It runs inside the sandbox child. Kernel params must be annotated literally as `GMEM[dtype, (dims…)]`.
- `judge.ir_check.load_kernel(json)` / `load_program(json)`: the only way IR JSON gets in. It is replayed through the DSL and must match byte for byte.
- `judge.bounds.prove_bounds(kernel, task)`: a Z3 proof that every access is in bounds, with matching extents, for every valid shape.
- `judge.correctness.run_tests(task, kernel)`: D1–D4 and S1–S4 on every task shape. Each runs twice, with different garbage and a shuffled CTA order. Results must be bit-identical, inputs untouched, output shapes right, and outputs within τ.
- `python -m judge.mutate`: the judge mutation score over all baselines.

Kernel param names must match the task's input/output names. The bounds proof links each kernel dim to the task's dim by name and position.

## Not checked yet (later weeks)

- Whether index or slice extents match the destination shape. This is the Z3 bounds check in judge step 2.
- Register, SMEM and TMEM limits against a chip profile (Week 3).
- Mapping `mma` tile shapes onto the chip's native MMA shape (cost model, Weeks 4–5).
- `if_` has no else branch. Use `select`, or add an else branch when a workload needs one.

## Paper-check kernels (Week 1 done-check)

`aliensilicon/tasks/baselines/`: `copy_rows`, `softmax_naive`, `gemm_naive` and `attention_naive`. All of them pass the judge allow-list and build into IR JSON (`tests/test_ir.py`). The last three are correct at τ on D1–D4 in the interpreter (`tests/test_interp.py`).
