"""NumPy interpreter for Tile IR (§9): exact dtypes, deterministic, seeded.

- f16/f32 ops run in their dtype (NumPy rounds each op to nearest-even).
- fma and mma compute in float64 and round once to the result dtype.
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
    """The kernel did something illegal at run time (out-of-bounds, bad shapes)."""


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
               seed: int = 0, mufu_ulp_error: float = 0.0) -> dict[str, np.ndarray]:
    """Run `kernel` and return every GMEM param by name.

    Params missing from `inputs` start as garbage, so a kernel must write all of its outputs.
    """
    dims = resolve_dims(kernel, inputs, dims)
    rng = np.random.default_rng([seed, 0])
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
    _launch(kernel, gmem, dims, seed, 0, mufu_ulp_error)
    return gmem


def run_program(program: Program, inputs: dict[str, np.ndarray], dims: dict[str, int] | None = None, *,
                seed: int = 0, mufu_ulp_error: float = 0.0) -> dict[str, np.ndarray]:
    """Run each launch in order; returns the program's params (not its temporaries)."""
    # shortcut: kernels and program share one dims namespace, split it if names start to clash.
    dims = resolve_dims(program, inputs, dims)
    rng = np.random.default_rng([seed, 0])
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
    for i, launch in enumerate(program.launches):
        k = kernels[launch["kernel"]]
        bound = {p["name"]: bufs[a] for p, a in zip(k.params, launch["args"], strict=True)}
        for p in k.params:
            want = _concrete(k.values[p["value"]].shape, dims)
            if bound[p["name"]].shape != want:
                raise InterpError(f"launch {i} ({k.name}): {p['name']} shape {bound[p['name']].shape} != {want}")
        _launch(k, bound, dims, seed, i + 1, mufu_ulp_error)  # arrays are shared, so writes land in bufs
    return {p["name"]: bufs[p["name"]] for p in program.params}


def _launch(k: Kernel, gmem: dict[str, np.ndarray], dims: dict[str, int], seed: int, launch: int,
            mufu_ulp_error: float) -> None:
    grid = _concrete(k.grid, dims)
    for cta_index, pid in enumerate(itertools.product(*(range(g) for g in grid))):
        env: dict[int, np.ndarray] = {p["value"]: gmem[p["name"]] for p in k.params}
        cta = _CTA(k, env, dims, pid, grid, np.random.default_rng([seed, launch + 1, cta_index]), mufu_ulp_error)
        with np.errstate(all="ignore"):  # overflow, NaN and divide-by-zero follow IEEE / wraparound rules
            cta.block(k.body)


class _CTA:
    def __init__(self, k: Kernel, env: dict[int, np.ndarray], dims: dict[str, int], pid: tuple[int, ...],
                 grid: tuple[int, ...], rng: np.random.Generator, mufu_ulp_error: float):
        self.k, self.env, self.dims, self.pid, self.grid = k, env, dims, pid, grid
        self.rng, self.mufu_ulp_error = rng, mufu_ulp_error

    # ------------------------------------------------------------ helpers

    def dtype(self, vid: int) -> np.dtype:
        return np.dtype(NP_DTYPE[self.k.values[vid].dtype])

    def get(self, x: Any, dtype: np.dtype | None = None) -> np.ndarray:
        if isinstance(x, str):
            return self.env[int(x[1:])]
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
            getattr(self, "op_" + op.kind, self.op_elementwise)(op)

    def op_alloc(self, op: Op) -> None:
        vid = op.outputs[0]
        v = self.k.values[vid]
        shape = _concrete(v.shape, self.dims)
        if "init" in op.attrs:
            self.set(vid, np.full(shape, op.attrs["init"], dtype=self.dtype(vid)))
        else:
            self.set(vid, garbage(shape, v.dtype, self.rng))

    def _move(self, op: Op) -> None:
        src, dst = self.get(op.inputs[0]), self.env[op.outputs[0]]
        s_idx, d_idx = self.index(src, op.attrs.get("src_index")), self.index(dst, op.attrs.get("dst_index"))
        data = src[s_idx]
        if np.shape(data) != np.shape(dst[d_idx]):
            raise InterpError(f"{op.kind}: moving shape {np.shape(data)} into {np.shape(dst[d_idx])}")
        dst[d_idx] = data  # in place: GMEM writes are visible to later CTAs

    op_load = op_store = op_async_copy = _move

    def op_wait(self, op: Op) -> None:
        pass  # copies complete immediately here; ordering hazards are the cost model's job

    op_barrier = op_wait

    def op_mma(self, op: Op) -> None:
        a, b, acc = (self.get(x) for x in op.inputs)
        b = b.T if op.attrs.get("transpose_b") else b
        acc[...] = (acc.astype(np.float64) + a.astype(np.float64) @ b.astype(np.float64)).astype(acc.dtype)

    def op_mufu(self, op: Op) -> None:
        vid = op.outputs[0]
        dt = self.dtype(vid)
        y = _MUFU[op.attrs["op"]](self.get(op.inputs[0]).astype(np.float64)).astype(dt)
        if self.mufu_ulp_error:
            noise = self.rng.uniform(-self.mufu_ulp_error, self.mufu_ulp_error, size=y.shape)
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


_ELEMENTWISE = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "fma": _fma,
    "max": np.maximum, "min": np.minimum, "neg": np.negative, "abs": np.abs,
    "iadd": np.add, "isub": np.subtract, "imul": np.multiply,
    "and": np.bitwise_and, "or": np.bitwise_or, "xor": np.bitwise_xor,
    "shl": lambda a, b: _shift(a, b, True), "shr": lambda a, b: _shift(a, b, False),
}
