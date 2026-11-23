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
| 4 | `mma(a, b, acc)` (acc += a @ b) | `mma` | MMA | `T.gemm` |
| 5 | `add sub mul fma max min neg abs` | same | FMA | expressions in `T.Parallel` |
| 6 | `mufu(op, x)`, op ∈ exp2 log2 rcp rsqrt sin cos | `mufu` + `attrs.op` | MUFU | `T.exp2`, `T.log2`, … |
| 7 | `iadd isub imul and_ or_ xor shl shr` | `iadd` … `and` `or` … | INT (`imul` → IMUL) | TIR integer ops |
| 8 | `bitcast(x, dt)` (equal width), `convert(x, dt)` | same | INT, FMA | TIR reinterpret / cast |
| 9 | `cmp(op, a, b)` → bool, `select(mask, a, b)` | same | FMA | `T.if_then_else` |
| 10 | `reduce(op, x, axis, scope)`, op ∈ max min sum | `reduce` | REDUCE (thread scope → FMA) | `T.reduce_max` / `T.reduce_sum` |
| 11 | `for i in serial(n, unroll=False):` | `for` | control | `T.serial` / `T.unroll` |
| 12 | `with if_(cond):` | `if` | control | TIR `if` |
| 13 | `barrier()` | `barrier` | control | implicit in TileLang / `T.sync_threads` |
| 14 | `program_id(axis)`, `num_programs(axis)` | same | none | block indices of `T.Kernel(...)` |
| 15 | `gather_load(src, indices, dst)` | `gather_load` | GMEM, `coalesced: false` | indexed `T.copy` |
| 16 | `@program` + `buffer(shape, dtype)` + kernel calls | `Program.launches` | none | TileLang kernels called in sequence |

**The TileLang names above have not been checked against the current TileLang docs yet.** Week 1 task: check them and fix this table.

## Rules the builder enforces

- Elementwise ops need one dtype across their tile operands. Number literals broadcast. Scalars `()` broadcast against tiles. Float ops reject int tiles, and int ops reject float tiles.
- `load` and `store` do not convert. Load into a tile of the source dtype, then `convert` it (see `softmax_naive.py`).
- `alloc` can't create GMEM. GMEM comes from kernel params or `buffer()` in a `@program`.
- Kernels return nothing. Outputs are written with `store`.
- `for` bodies are traced once and kept as one rolled loop op. `unroll` is only a mark.

## Not checked yet (later weeks)

- Whether index or slice extents match the destination shape. This is the Z3 bounds check in judge step 2.
- Register, SMEM and TMEM limits against a chip profile (Week 3).
- Mapping `mma` tile shapes onto the chip's native MMA shape (cost model, Weeks 4–5).
- `if_` has no else branch. Use `select`, or add an else branch when a workload needs one.

## Paper-check kernels (Week 1 done-check)

`aliensilicon/tasks/baselines/`: `copy_rows`, `softmax_naive` and `gemm_naive`. All three pass the judge allow-list and build into IR JSON (`tests/test_ir.py`).
