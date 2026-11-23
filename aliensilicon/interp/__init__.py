"""NumPy interpreter for Tile IR (§9): exact dtypes, deterministic, seeded.

- f16/f32 ops run in their dtype (NumPy rounds each op to nearest-even).
- fma computes in float64 and rounds once. mma does too, unless `mma_k_chunk=c`: then it rounds
  the accumulator after every c-wide slice of K, like a tensor core (c is a chip property).
- New SMEM/RMEM/TMEM tiles and unset GMEM outputs hold seeded garbage bits.
- mufu adds seeded error of up to `mufu_ulp_error` ulps.
- CTAs run one after another over the grid; threads are NumPy vector lanes.
"""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from aliensilicon.ir import Block, Kernel, Op, Program

NP_DTYPE = {"f16": np.float16, "f32": np.float32, "i32": np.int32, "i8": np.int8, "bool": np.bool_}
_MUFU = {
    "exp2": np.exp2, "log2": np.log2, "rcp": lambda x: 1.0 / x, "rsqrt": lambda x: 1.0 / np.sqrt(x),
    "sin": np.sin, "cos": np.cos,
}
_REDUCE = {"max": np.max, "min": np.min, "sum": np.sum}
_CMP = {"lt": np.less, "le": np.less_equal, "gt": np.greater, "ge": np.greater_equal,
        "eq": np.equal, "ne": np.not_equal}


class InterpError(Exception):
    """The kernel did something illegal at run time (out-of-bounds, bad shapes, over budget)."""


# Hard caps so a hostile kernel cannot hang or OOM the judge.
MAX_TILE_ELEMENTS = 1 << 20  # per SMEM/RMEM/TMEM alloc
MAX_CTAS = 1 << 16
DEFAULT_WORK_BUDGET = 2 * 10**9  # element-ops per run; each op also costs OP_OVERHEAD
OP_OVERHEAD = 10_000  # ~10 us of Python dispatch per op: caps a run at ~200k ops (~2 s)


class _Budget:
    def __init__(self, limit: int):
        self.left = limit

    def charge(self, n: int) -> None:
        self.left -= n
        if self.left < 0:
            raise InterpError("work budget exhausted (kernel too slow to interpret)")


def garbage(shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
    """Random bit patterns: includes NaN/Inf for floats, so reading unset memory shows up."""
    dt = np.dtype(NP_DTYPE[dtype])
    if dt == np.bool_:
        return rng.integers(0, 2, size=shape).astype(np.bool_)
    bits = rng.integers(0, 256, size=(*shape, dt.itemsize), dtype=np.uint8)
    return bits.view(dt).reshape(shape).copy()


def resolve_dims(kernel_or_program: Kernel | Program, inputs: dict[str, np.ndarray],
                 dims: dict[str, int] | None) -> dict[str, int]:
    """Bind symbolic dims from explicit `dims` plus the shapes of the inputs."""
    out = dict(dims or {})
    if isinstance(kernel_or_program, Kernel):
        shapes = {p["name"]: kernel_or_program.values[p["value"]].shape for p in kernel_or_program.params}
    else:
        shapes = {p["name"]: tuple(p["shape"]) for p in kernel_or_program.params}
    for name, arr in inputs.items():
        if name not in shapes:
            raise InterpError(f"unknown input {name!r}")
        spec = shapes[name]
        if len(spec) != arr.ndim:
            raise InterpError(f"input {name}: rank {arr.ndim} != spec rank {len(spec)}")
        for d, n in zip(spec, arr.shape, strict=True):
            if isinstance(d, str):
                if out.setdefault(d, n) != n:
                    raise InterpError(f"dim {d!r} bound to both {out[d]} and {n}")
            elif d != n:
                raise InterpError(f"input {name}: shape {arr.shape} != spec {spec}")
    return out


def _concrete(shape: tuple[Any, ...], dims: dict[str, int]) -> tuple[int, ...]:
    try:
        return tuple(dims[d] if isinstance(d, str) else d for d in shape)
    except KeyError as e:
        raise InterpError(f"symbolic dim {e.args[0]!r} is not bound; pass it in dims=") from None


def run_kernel(kernel: Kernel, inputs: dict[str, np.ndarray], dims: dict[str, int] | None = None, *,
               seed: int = 0, mufu_ulp_error: float = 0.0, mma_k_chunk: int | None = None,
               cta_order: int | None = None,
               garbage_seed: int | None = None, work_budget: int = DEFAULT_WORK_BUDGET) -> dict[str, np.ndarray]:
    """Run `kernel` and return every GMEM param by name.

    Params missing from `inputs` start as garbage, so a kernel must write all of its outputs.
    `cta_order=None` runs CTAs in grid order; an int shuffles them with that seed. Real GPUs
    give no order, so a correct kernel must give bit-identical output for every order.
    """
    dims = resolve_dims(kernel, inputs, dims)
    gseed = seed if garbage_seed is None else garbage_seed
    rng = np.random.default_rng([gseed, 0])
    gmem = {}
    for p in kernel.params:
        v = kernel.values[p["value"]]
        if p["name"] in inputs:
            arr = np.asarray(inputs[p["name"]])
            if arr.dtype != NP_DTYPE[v.dtype]:
                raise InterpError(f"input {p['name']}: dtype {arr.dtype} != {v.dtype}")
            gmem[p["name"]] = arr.copy()
        else:
            gmem[p["name"]] = garbage(_concrete(v.shape, dims), v.dtype, rng)
    _launch(kernel, gmem, dims, seed, gseed, 0, (mufu_ulp_error, mma_k_chunk), cta_order, _Budget(work_budget))
    return gmem


def run_program(program: Program, inputs: dict[str, np.ndarray], dims: dict[str, int] | None = None, *,
                seed: int = 0, mufu_ulp_error: float = 0.0, mma_k_chunk: int | None = None,
                cta_order: int | None = None,
                garbage_seed: int | None = None, work_budget: int = DEFAULT_WORK_BUDGET) -> dict[str, np.ndarray]:
    """Run each launch in order; returns the program's params (not its temporaries)."""
    # shortcut: kernels and program share one dims namespace, split it if names start to clash.
    dims = resolve_dims(program, inputs, dims)
    gseed = seed if garbage_seed is None else garbage_seed
    rng = np.random.default_rng([gseed, 0])
    bufs = {}
    for p in program.params + program.buffers:
        if p["name"] in inputs:
            arr = np.asarray(inputs[p["name"]])
            if arr.dtype != NP_DTYPE[p["dtype"]]:
                raise InterpError(f"input {p['name']}: dtype {arr.dtype} != {p['dtype']}")
            bufs[p["name"]] = arr.copy()
        else:
            bufs[p["name"]] = garbage(_concrete(tuple(p["shape"]), dims), p["dtype"], rng)
    kernels = {k.name: k for k in program.kernels}
    budget = _Budget(work_budget)
    for i, launch in enumerate(program.launches):
        k = kernels[launch["kernel"]]
        bound = {p["name"]: bufs[a] for p, a in zip(k.params, launch["args"], strict=True)}
        for p in k.params:
            want = _concrete(k.values[p["value"]].shape, dims)
            if bound[p["name"]].shape != want:
                raise InterpError(f"launch {i} ({k.name}): {p['name']} shape {bound[p['name']].shape} != {want}")
        _launch(k, bound, dims, seed, gseed, i + 1, (mufu_ulp_error, mma_k_chunk), cta_order, budget)
    return {p["name"]: bufs[p["name"]] for p in program.params}


def _launch(k: Kernel, gmem: dict[str, np.ndarray], dims: dict[str, int], seed: int, gseed: int, launch: int,
            numerics: tuple[float, int | None], cta_order: int | None, budget: _Budget) -> None:
    grid = _concrete(k.grid, dims)
    pids = list(itertools.product(*(range(g) for g in grid)))
    if len(pids) > MAX_CTAS:
        raise InterpError(f"grid of {len(pids)} CTAs exceeds {MAX_CTAS}")
    order = list(range(len(pids)))
    if cta_order is not None:
        np.random.default_rng([cta_order, launch]).shuffle(order)
    # Which CTA wrote each GMEM element in this launch; keyed by array so aliased args share one map.
    by_array = {id(a): np.full(a.shape, -1, np.int32) for a in gmem.values()}
    owners = {p["value"]: by_array[id(gmem[p["name"]])] for p in k.params}
    for cta_index in order:
        env: dict[int, np.ndarray] = {p["value"]: gmem[p["name"]] for p in k.params}
        # rng keyed by the CTA's grid position, not run order, so shuffling changes nothing for honest kernels
        rngs = (np.random.default_rng([gseed, launch + 1, cta_index]),
                np.random.default_rng([seed, launch + 1, cta_index, 1]))
        cta = _CTA(k, env, dims, pids[cta_index], grid, rngs, numerics, budget, owners, cta_index)
        with np.errstate(all="ignore"):  # overflow, NaN and divide-by-zero follow IEEE / wraparound rules
            cta.block(k.body)


class _CTA:
    def __init__(self, k: Kernel, env: dict[int, np.ndarray], dims: dict[str, int], pid: tuple[int, ...],
                 grid: tuple[int, ...], rngs: tuple[np.random.Generator, np.random.Generator],
                 numerics: tuple[float, int | None], budget: _Budget, owners: dict[int, np.ndarray],
                 cta_id: int):
        self.k, self.env, self.dims, self.pid, self.grid = k, env, dims, pid, grid
        (self.garbage_rng, self.mufu_rng), self.budget = rngs, budget
        self.mufu_ulp_error, self.mma_k_chunk = numerics
        self.owners, self.cta_id = owners, cta_id
        # Shared-memory hazards since the last barrier, as per-element masks (threads are vector
        # lanes here, so races never happen by themselves; we detect them instead).
        self.written: dict[int, np.ndarray] = {}
        self.read: dict[int, np.ndarray] = {}
        self.pending: dict[int, dict[int, np.ndarray]] = {}  # async_copy dst -> {group: mask} until wait
        self.op_reads: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------ helpers

    def dtype(self, vid: int) -> np.dtype:
        return np.dtype(NP_DTYPE[self.k.values[vid].dtype])

    def shared(self, vid: int) -> bool:
        return self.k.values[vid].space in ("SMEM", "TMEM")

    def get(self, x: Any, dtype: np.dtype | None = None) -> np.ndarray:
        if isinstance(x, str):
            vid = int(x[1:])
            self.touch(vid, (), write=False)
            return self.env[vid]
        return np.asarray(x, dtype=dtype)  # literal: takes the dtype of the result

    def set(self, vid: int, arr: np.ndarray) -> None:
        want = _concrete(self.k.values[vid].shape, self.dims)
        arr = np.asarray(arr).astype(self.dtype(vid), copy=False)
        if arr.shape != want:
            raise InterpError(f"%{vid}: computed shape {arr.shape} != declared {want}")
        self.env[vid] = arr

    def scalar(self, x: Any) -> int:
        return int(self.get(x))

    def index(self, arr: np.ndarray, idx: list[Any] | None) -> tuple[Any, ...]:
        """Turn an IR index into a NumPy index, failing on any out-of-bounds access."""
        if idx is None:
            return ()
        out: list[Any] = []
        for item, size in zip(idx, arr.shape, strict=True):
            if isinstance(item, dict):
                start, stop = item["slice"]
                start = 0 if start is None else self.scalar(start)
                stop = size if stop is None else self.scalar(stop)
                if not 0 <= start <= stop <= size:
                    raise InterpError(f"slice [{start}:{stop}] out of bounds for dim of size {size}")
                out.append(slice(start, stop))
            else:
                i = self.scalar(item)
                if not 0 <= i < size:
                    raise InterpError(f"index {i} out of bounds for dim of size {size}")
                out.append(i)
        return tuple(out)

    # ------------------------------------------------------------ execution

    def block(self, b: Block) -> None:
        for op in b.ops:
            self.budget.charge(OP_OVERHEAD)
            self.op_reads = {}
            getattr(self, "op_" + op.kind, self.op_elementwise)(op)
            if op.kind in _WRITES_WHOLE or (op.kind == "alloc" and "init" in op.attrs):
                self.touch(op.outputs[0], (), write=True)
            for vid, m in self.op_reads.items():
                self.read[vid] = self.read[vid] | m if vid in self.read else m
            for vid in op.outputs:
                self.budget.charge(self.env[vid].size)

    def touch(self, vid: int, idx: tuple[Any, ...], write: bool, group: int | None = None) -> None:
        """Check a read or write of region `idx` of a shared tile against hazards since the last barrier."""
        if not self.shared(vid):
            return
        r = np.zeros(self.env[vid].shape, dtype=bool)
        r[idx] = True
        space = self.k.values[vid].space
        if any((m & r).any() for m in self.pending.get(vid, {}).values()):
            raise InterpError(f"race: %{vid} used while an async_copy into it is in flight (missing wait())")
        if not write:
            if vid in self.written and (self.written[vid] & r).any():
                raise InterpError(f"race: %{vid} ({space}) read after write with no barrier()")
            self.op_reads[vid] = self.op_reads[vid] | r if vid in self.op_reads else r
        elif vid in self.read and (self.read[vid] & r).any():
            raise InterpError(f"race: %{vid} ({space}) written after a read with no barrier()")
        elif group is not None:
            groups = self.pending.setdefault(vid, {})
            groups[group] = groups[group] | r if group in groups else r
        else:
            self.written[vid] = self.written[vid] | r if vid in self.written else r

    def own(self, vid: int, idx: tuple[Any, ...], write: bool) -> None:
        """GMEM: a CTA may not read or overwrite elements another CTA wrote in this launch."""
        if self.k.values[vid].space != "GMEM":
            return
        o = self.owners[vid][idx]
        other = np.unique(o[(o != -1) & (o != self.cta_id)])
        if other.size:
            what = "writes" if write else "reads"
            raise InterpError(f"race: CTA {self.cta_id} {what} GMEM %{vid} written by CTA {int(other[0])}")
        if write:
            self.owners[vid][idx] = self.cta_id

    def op_alloc(self, op: Op) -> None:
        vid = op.outputs[0]
        v = self.k.values[vid]
        shape = _concrete(v.shape, self.dims)
        if np.prod(shape, dtype=np.int64) > MAX_TILE_ELEMENTS:
            raise InterpError(f"alloc of {shape} exceeds {MAX_TILE_ELEMENTS} elements")
        if "init" in op.attrs:
            self.set(vid, np.full(shape, op.attrs["init"], dtype=self.dtype(vid)))
        else:
            self.set(vid, garbage(shape, v.dtype, self.garbage_rng))

    def _move(self, op: Op) -> None:
        s_vid, d_vid = int(op.inputs[0][1:]), op.outputs[0]
        src, dst = self.env[s_vid], self.env[d_vid]
        s_idx, d_idx = self.index(src, op.attrs.get("src_index")), self.index(dst, op.attrs.get("dst_index"))
        data = src[s_idx]
        if np.shape(data) != np.shape(dst[d_idx]):
            raise InterpError(f"{op.kind}: moving shape {np.shape(data)} into {np.shape(dst[d_idx])}")
        self.touch(s_vid, s_idx, write=False)  # region-level: disjoint parts of one tile never conflict
        self.touch(d_vid, d_idx, write=True, group=op.attrs["group"] if op.kind == "async_copy" else None)
        self.own(s_vid, s_idx, write=False)
        self.own(d_vid, d_idx, write=True)
        dst[d_idx] = data  # in place: GMEM writes are visible to later CTAs

    op_load = op_store = op_async_copy = _move

    def op_wait(self, op: Op) -> None:
        # Copies already happened; wait() makes them visible, still only after a barrier for other threads.
        for vid, groups in self.pending.items():
            m = groups.pop(op.attrs["group"], None)
            if m is not None:
                self.written[vid] = self.written[vid] | m if vid in self.written else m
        self.pending = {v: g for v, g in self.pending.items() if g}

    def op_barrier(self, op: Op) -> None:
        self.written.clear()
        self.read.clear()

    def op_mma(self, op: Op) -> None:
        a, b, acc = (self.get(x) for x in op.inputs)
        b = b.T if op.attrs.get("transpose_b") else b
        self.budget.charge(a.shape[0] * a.shape[1] * b.shape[1])
        a, b = a.astype(np.float64), b.astype(np.float64)
        step = self.mma_k_chunk or a.shape[1]
        out = acc.astype(np.float64)
        for k0 in range(0, a.shape[1], step):  # round into the accumulator dtype after each k-chunk
            out = (out + a[:, k0:k0 + step] @ b[k0:k0 + step, :]).astype(acc.dtype).astype(np.float64)
        acc[...] = out

    def op_mufu(self, op: Op) -> None:
        vid = op.outputs[0]
        dt = self.dtype(vid)
        y = _MUFU[op.attrs["op"]](self.get(op.inputs[0]).astype(np.float64)).astype(dt)
        if self.mufu_ulp_error:
            noise = self.mufu_rng.uniform(-self.mufu_ulp_error, self.mufu_ulp_error, size=y.shape)
            y = (y + noise * np.spacing(np.abs(y))).astype(dt)
        self.set(vid, y)

    def op_reduce(self, op: Op) -> None:
        # shortcut: NumPy pairwise sum in the tile dtype; model the chip's tree order if a task becomes sensitive.
        x = self.get(op.inputs[0])
        self.set(op.outputs[0], _REDUCE[op.attrs["op"]](x, axis=op.attrs["axis"],
                                                         keepdims=op.attrs.get("keepdims", False)))

    def op_convert(self, op: Op) -> None:
        x, vid = self.get(op.inputs[0]), op.outputs[0]
        dt = self.dtype(vid)
        if np.issubdtype(dt, np.integer) and np.issubdtype(x.dtype, np.floating):
            info = np.iinfo(dt)  # round toward zero, saturate, NaN -> 0 (like cvt.rzi.sat)
            x = np.nan_to_num(np.trunc(x.astype(np.float64)), nan=0.0, posinf=info.max, neginf=info.min)
            x = np.clip(x, info.min, info.max)
        self.set(vid, x.astype(dt))

    def op_bitcast(self, op: Op) -> None:
        self.set(op.outputs[0], np.ascontiguousarray(self.get(op.inputs[0])).view(self.dtype(op.outputs[0])))

    def op_cmp(self, op: Op) -> None:
        a, b = self._operands(op, op.inputs)
        self.set(op.outputs[0], _CMP[op.attrs["op"]](a, b))

    def op_select(self, op: Op) -> None:
        mask = self.get(op.inputs[0])
        a, b = self._operands(op, op.inputs[1:])
        self.set(op.outputs[0], np.where(mask, a, b))

    def _operands(self, op: Op, inputs: list[Any]) -> list[np.ndarray]:
        tiles = [self.get(x) for x in inputs if isinstance(x, str)]
        dt = tiles[0].dtype
        return [self.get(x, dt) for x in inputs]

    def op_elementwise(self, op: Op) -> None:
        if op.kind not in _ELEMENTWISE:
            raise InterpError(f"unknown op kind {op.kind!r}")
        xs = self._operands(op, op.inputs)
        self.set(op.outputs[0], _ELEMENTWISE[op.kind](*xs))

    def op_program_id(self, op: Op) -> None:
        self.set(op.outputs[0], self.pid[op.attrs["axis"]])

    def op_num_programs(self, op: Op) -> None:
        self.set(op.outputs[0], self.grid[op.attrs["axis"]])

    def op_for(self, op: Op) -> None:
        n = op.attrs["n"]
        n = self.dims[n] if isinstance(n, str) else n
        for i in range(n):
            self.set(op.outputs[0], i)
            self.block(op.blocks[0])

    def op_if(self, op: Op) -> None:
        if bool(self.get(op.inputs[0])):
            self.block(op.blocks[0])

    def op_gather_load(self, op: Op) -> None:
        src, idx = self.get(op.inputs[0]), self.get(op.inputs[1])
        dst = self.env[op.outputs[0]]
        rows = np.clip(idx.astype(np.int64), 0, src.shape[0] - 1)  # §8.3 group 15: indices are clamped
        self.own(int(op.inputs[0][1:]), (rows,), write=False)
        data = src[rows]
        if data.shape != dst.shape:
            raise InterpError(f"gather_load: gathered shape {data.shape} != dst {dst.shape}")
        dst[...] = data


def _fma(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    # f64 is exact for f16 products; for f32 the double rounding is off by 1 ulp only in rare ties.
    return a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)


def _shift(a: np.ndarray, b: np.ndarray, left: bool) -> np.ndarray:
    # PTX rule: shift amounts >= width clamp to width (shl -> 0, shr -> sign fill).
    bits = a.dtype.itemsize * 8
    s = np.minimum(np.clip(b.astype(np.int64), 0, bits), bits - 1)
    wide = a.astype(np.int64)
    if not left:
        return np.asarray(wide >> s).astype(a.dtype)
    out = np.asarray(np.where(b.astype(np.int64) >= bits, 0, wide << s) & ((1 << bits) - 1))
    return out.astype(f"u{a.dtype.itemsize}").view(a.dtype)  # wrap into the narrow signed type


_WRITES_WHOLE = ("mma", "gather_load")  # load/store/async_copy track their exact region in _move

_ELEMENTWISE = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "fma": _fma,
    "max": np.maximum, "min": np.minimum, "neg": np.negative, "abs": np.abs,
    "iadd": np.add, "isub": np.subtract, "imul": np.multiply,
    "and": np.bitwise_and, "or": np.bitwise_or, "xor": np.bitwise_xor,
    "shl": lambda a, b: _shift(a, b, True), "shr": lambda a, b: _shift(a, b, False),
}
