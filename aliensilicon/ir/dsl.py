"""Python DSL that builds Tile IR (§8.3, §8.4).

Kernel functions run once at build time; each primitive call appends an Op.
Primitive names are generic on purpose (rule 8.1.1): no trick is built in.
"""
from __future__ import annotations

import builtins
import inspect
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .core import DTYPES, Block, Dim, Kernel, Op, Program, Value

_WIDTH = {"f16": 2, "f32": 4, "i32": 4, "i8": 1, "bool": 1}
_FLOAT = ("f16", "f32")
_INT = ("i32", "i8")
MUFU_OPS = ("exp2", "log2", "rcp", "rsqrt", "sin", "cos")
REDUCE_OPS = ("max", "min", "sum")
REDUCE_SCOPES = ("thread", "warp", "cta")
CMP_OPS = ("lt", "le", "gt", "ge", "eq", "ne")


class DSLError(Exception):
    pass


# ---------------------------------------------------------------- types

f16, f32, i32, i8 = "f16", "f32", "i32", "i8"


@dataclass(frozen=True)
class ParamType:
    space: str
    dtype: str
    shape: tuple[Dim, ...]


class Space(str):
    """`GMEM` is both an alloc space and, subscripted, a param annotation."""

    def __getitem__(self, key: Any) -> ParamType:  # type: ignore[override]
        dtype, shape = key
        _check_dtype(dtype)
        return ParamType(str(self), dtype, _shape(shape))


GMEM, SMEM, RMEM, TMEM = Space("GMEM"), Space("SMEM"), Space("RMEM"), Space("TMEM")


class Tile:
    """Build-time handle for an IR value."""

    def __init__(self, value: Value):
        self.value = value

    @property
    def ref(self) -> str:
        return f"%{self.value.id}"

    def __getitem__(self, index: Any) -> View:
        index = index if isinstance(index, tuple) else (index,)
        if len(index) != len(self.value.shape):
            raise DSLError(f"index rank {len(index)} != tile rank {len(self.value.shape)}")
        return View(self, [_index_item(i) for i in index])


@dataclass
class View:
    tile: Tile
    index: list[Any]


def _index_item(i: Any) -> Any:
    if isinstance(i, Tile):
        if i.value.shape != () or i.value.dtype not in _INT:
            raise DSLError("tile index must be an integer scalar")
        return i.ref
    if isinstance(i, slice):
        if i.step not in (None, 1):
            raise DSLError("strided slices are not supported; use the stride layout attr")
        return {"slice": [_index_item(i.start) if i.start is not None else None,
                          _index_item(i.stop) if i.stop is not None else None]}
    if isinstance(i, int) and not isinstance(i, bool):
        return i
    raise DSLError(f"bad index item {i!r}")


def _check_dtype(dtype: str) -> None:
    if dtype not in DTYPES:
        raise DSLError(f"unknown dtype {dtype!r}")


# ---------------------------------------------------------------- builder

class _Builder:
    def __init__(self, grid: tuple[Dim, ...] = ()) -> None:
        self.grid = grid
        self.values: list[Value] = []
        self.blocks: list[Block] = [Block()]

    def new(self, dtype: str, shape: tuple[Dim, ...], space: str) -> Tile:
        v = Value(len(self.values), dtype, tuple(shape), space)
        self.values.append(v)
        return Tile(v)

    def emit(self, kind: str, inputs: list[Any], outputs: list[Tile], unit: str,
             attrs: dict[str, Any] | None = None, blocks: list[Block] | None = None) -> None:
        self.blocks[-1].ops.append(Op(kind, inputs, [t.value.id for t in outputs],
                                      attrs or {}, unit, blocks or []))


_builders: list[_Builder] = []
_programs: list[dict[str, Any]] = []


def _b() -> _Builder:
    if not _builders:
        raise DSLError("DSL primitives can only be called inside a @kernel body")
    return _builders[-1]


def _operand(x: Any) -> Any:
    if isinstance(x, Tile):
        return x.ref
    if isinstance(x, int) and not isinstance(x, bool):
        if not -(2**31) <= x < 2**31:
            raise DSLError(f"int literal {x} does not fit in i32")
        return x
    if isinstance(x, float):
        if not math.isfinite(x):
            raise DSLError(f"float literal {x} must be finite")
        return x
    raise DSLError(f"operand must be a tile or number, got {type(x).__name__}")


def _shape(shape: Any) -> tuple[Dim, ...]:
    shape = tuple(shape)
    for d in shape:
        ok = (isinstance(d, int) and not isinstance(d, bool) and d > 0) or (isinstance(d, str) and d.isidentifier())
        if not ok:
            raise DSLError(f"shape dim {d!r} must be a positive int or a symbolic name")
    return shape


def _ref(x: Tile | View) -> tuple[Tile, dict[str, Any]]:
    if isinstance(x, View):
        return x.tile, {"index": x.index}
    if isinstance(x, Tile):
        return x, {}
    raise DSLError(f"expected a tile or tile[index], got {type(x).__name__}")


def _layout(pad: Any, swizzle: Any, stride: Any) -> dict[str, Any]:
    return {k: v for k, v in (("pad", pad), ("swizzle", swizzle), ("stride", stride)) if v is not None}


# ---------------------------------------------------------------- 1 allocate

def alloc(space: str, shape: tuple[Dim, ...], dtype: str, *, init: float | None = None,
          pad: Any = None, swizzle: Any = None, stride: Any = None) -> Tile:
    """New tiles hold garbage (the interpreter fills them) unless `init` is given."""
    if space not in (SMEM, RMEM, TMEM):
        raise DSLError("alloc space must be SMEM, RMEM or TMEM (GMEM comes from params/buffers)")
    _check_dtype(dtype)
    t = _b().new(dtype, _shape(shape), str(space))
    attrs = _layout(pad, swizzle, stride)
    if init is not None:
        attrs["init"] = _operand(init)
    _b().emit("alloc", [], [t], "none", attrs)
    return t


# ---------------------------------------------------------------- 2 load/store, 3 async copy

def _mem_unit(a: str, b: str) -> str:
    for space in ("GMEM", "TMEM", "SMEM"):
        if space in (a, b):
            return space
    return "FMA"


def _move(kind: str, src: Tile | View, dst: Tile | View, unit: str | None, attrs: dict[str, Any]) -> None:
    s, s_attrs = _ref(src)
    d, d_attrs = _ref(dst)
    if s.value.dtype != d.value.dtype:
        raise DSLError(f"{kind}: dtype mismatch {s.value.dtype} -> {d.value.dtype}; use convert()")
    attrs = {**attrs, **{f"src_{k}": v for k, v in s_attrs.items()}, **{f"dst_{k}": v for k, v in d_attrs.items()}}
    _b().emit(kind, [s.ref], [d], unit or _mem_unit(s.value.space, d.value.space), attrs)


def load(src: Tile | View, dst: Tile | View, *, pad: Any = None, swizzle: Any = None, stride: Any = None) -> None:
    _move("load", src, dst, None, _layout(pad, swizzle, stride))


def store(src: Tile | View, dst: Tile | View, *, pad: Any = None, swizzle: Any = None, stride: Any = None) -> None:
    _move("store", src, dst, None, _layout(pad, swizzle, stride))


def async_copy(src: Tile | View, dst: Tile | View, group: int = 0) -> None:
    _move("async_copy", src, dst, "COPY", {"group": group})


def wait(group: int = 0) -> None:
    _b().emit("wait", [], [], "control", {"group": group})


# ---------------------------------------------------------------- 4 mma

def mma(a: Tile, b: Tile, acc: Tile, transpose_b: bool = False) -> Tile:
    """acc += a @ b (or a @ b.T), written in place. Matches T.gemm(transpose_B=...)."""
    for t in (a, b, acc):
        if not isinstance(t, Tile) or len(t.value.shape) != 2:
            raise DSLError("mma operands must be 2-D tiles")
    (m, k), (m2, n2) = a.value.shape, acc.value.shape
    n, k2 = b.value.shape if transpose_b else b.value.shape[::-1]
    if k != k2 or m != m2 or n != n2:
        raise DSLError(f"mma shape mismatch {a.value.shape} @ {b.value.shape}"
                       f"{'.T' if transpose_b else ''} -> {acc.value.shape}")
    _b().emit("mma", [a.ref, b.ref, acc.ref], [acc], "MMA", {"transpose_b": transpose_b})
    return acc


# ---------------------------------------------------------------- 5,7,9 elementwise

def broadcast_shape(kind: str, shapes: list[tuple[Dim, ...]]) -> tuple[Dim, ...]:
    """NumPy broadcasting; symbolic dims must match exactly (or meet a 1)."""
    out: list[Dim] = []
    for i in range(1, builtins.max((len(s) for s in shapes), default=0) + 1):
        dims = {s[-i] for s in shapes if len(s) >= i} - {1}
        if len(dims) > 1:
            raise DSLError(f"{kind}: shapes {[tuple(s) for s in shapes]} do not broadcast")
        out.append(dims.pop() if dims else 1)
    return tuple(reversed(out))


def _elementwise(kind: str, unit: str, allowed: tuple[str, ...], *xs: Any, out_dtype: str | None = None) -> Tile:
    tiles = [x for x in xs if isinstance(x, Tile)]
    if not tiles:
        raise DSLError(f"{kind}: at least one operand must be a tile")
    dtypes = {t.value.dtype for t in tiles}
    if len(dtypes) != 1:
        raise DSLError(f"{kind}: mixed dtypes {sorted(dtypes)}; use convert()")
    dtype = dtypes.pop()
    if dtype not in allowed:
        raise DSLError(f"{kind}: dtype {dtype} not allowed (needs one of {allowed})")
    out = _b().new(out_dtype or dtype, broadcast_shape(kind, [t.value.shape for t in tiles]), "RMEM")
    _b().emit(kind, [_operand(x) for x in xs], [out], unit)
    return out


def add(a: Any, b: Any) -> Tile: return _elementwise("add", "FMA", _FLOAT, a, b)
def sub(a: Any, b: Any) -> Tile: return _elementwise("sub", "FMA", _FLOAT, a, b)
def mul(a: Any, b: Any) -> Tile: return _elementwise("mul", "FMA", _FLOAT, a, b)
def fma(a: Any, b: Any, c: Any) -> Tile: return _elementwise("fma", "FMA", _FLOAT, a, b, c)
def neg(a: Tile) -> Tile: return _elementwise("neg", "FMA", _FLOAT, a)
def _max(a: Any, b: Any) -> Tile: return _elementwise("max", "FMA", _FLOAT, a, b)
def _min(a: Any, b: Any) -> Tile: return _elementwise("min", "FMA", _FLOAT, a, b)
def _abs(a: Tile) -> Tile: return _elementwise("abs", "FMA", _FLOAT, a)

def iadd(a: Any, b: Any) -> Tile: return _elementwise("iadd", "INT", _INT, a, b)
def isub(a: Any, b: Any) -> Tile: return _elementwise("isub", "INT", _INT, a, b)
def imul(a: Any, b: Any) -> Tile: return _elementwise("imul", "IMUL", _INT, a, b)
def and_(a: Any, b: Any) -> Tile: return _elementwise("and", "INT", _INT, a, b)
def or_(a: Any, b: Any) -> Tile: return _elementwise("or", "INT", _INT, a, b)
def xor(a: Any, b: Any) -> Tile: return _elementwise("xor", "INT", _INT, a, b)
def shl(a: Any, b: Any) -> Tile: return _elementwise("shl", "INT", _INT, a, b)
def shr(a: Any, b: Any) -> Tile: return _elementwise("shr", "INT", _INT, a, b)


def cmp(op: str, a: Any, b: Any) -> Tile:
    if op not in CMP_OPS:
        raise DSLError(f"cmp op must be one of {CMP_OPS}")
    t = _elementwise("cmp", "FMA", _FLOAT + _INT, a, b, out_dtype="bool")
    _b().blocks[-1].ops[-1].attrs["op"] = op
    return t


def select(mask: Tile, a: Any, b: Any) -> Tile:
    if not isinstance(mask, Tile) or mask.value.dtype != "bool":
        raise DSLError("select mask must be a bool tile from cmp()")
    out = _elementwise("select", "FMA", _FLOAT + _INT, a, b)
    out.value.shape = broadcast_shape("select", [mask.value.shape, out.value.shape])
    _b().blocks[-1].ops[-1].inputs.insert(0, mask.ref)
    return out


# ---------------------------------------------------------------- 6 special functions

def mufu(op: str, x: Tile) -> Tile:
    if op not in MUFU_OPS:
        raise DSLError(f"mufu op must be one of {MUFU_OPS}")
    t = _elementwise("mufu", "MUFU", _FLOAT, x)
    _b().blocks[-1].ops[-1].attrs["op"] = op
    return t


# ---------------------------------------------------------------- 8 reinterpret/convert

def convert(x: Tile, dtype: str) -> Tile:
    _check_dtype(dtype)
    out = _b().new(dtype, x.value.shape, "RMEM")
    _b().emit("convert", [x.ref], [out], "FMA", {"dtype": dtype})
    return out


def bitcast(x: Tile, dtype: str) -> Tile:
    _check_dtype(dtype)
    if _WIDTH[x.value.dtype] != _WIDTH[dtype]:
        raise DSLError(f"bitcast needs equal widths: {x.value.dtype} -> {dtype}")
    out = _b().new(dtype, x.value.shape, "RMEM")
    _b().emit("bitcast", [x.ref], [out], "INT", {"dtype": dtype})
    return out


# ---------------------------------------------------------------- 10 reductions

def reduce(op: str, x: Tile, axis: int, scope: str = "cta", keepdims: bool = False) -> Tile:
    if op not in REDUCE_OPS:
        raise DSLError(f"reduce op must be one of {REDUCE_OPS}")
    if scope not in REDUCE_SCOPES:
        raise DSLError(f"reduce scope must be one of {REDUCE_SCOPES}")
    shape = x.value.shape
    if not 0 <= axis < len(shape):
        raise DSLError(f"reduce axis {axis} out of range for shape {shape}")
    out_shape = shape[:axis] + ((1,) if keepdims else ()) + shape[axis + 1:]
    out = _b().new(x.value.dtype, out_shape, "RMEM")
    _b().emit("reduce", [x.ref], [out], "FMA" if scope == "thread" else "REDUCE",
              {"op": op, "axis": axis, "scope": scope, "keepdims": keepdims})
    return out


# ---------------------------------------------------------------- 11-14 control, sync, grid

def serial(n: Dim, unroll: bool = False) -> Iterator[Tile]:
    """`for i in serial(n):` records ONE loop op; the body is traced once."""
    n = _shape((n,))[0]  # static: a positive int or a symbolic dim
    b = _b()
    i = b.new("i32", (), "RMEM")
    b.blocks.append(Block())
    yield i
    body = b.blocks.pop()
    b.emit("for", [], [i], "control", {"n": n, "unroll": unroll}, [body])


@contextmanager
def if_(cond: Tile) -> Iterator[None]:
    # shortcut: no else branch, add one when a workload needs it (select() covers most cases).
    if not isinstance(cond, Tile) or cond.value.dtype != "bool" or cond.value.shape != ():
        raise DSLError("if_ condition must be a scalar bool from cmp()")
    b = _b()
    b.blocks.append(Block())
    yield
    body = b.blocks.pop()
    b.emit("if", [cond.ref], [], "control", {}, [body])


def barrier() -> None:
    _b().emit("barrier", [], [], "control")


def _axis(axis: Any) -> int:
    if not isinstance(axis, int) or isinstance(axis, bool) or not 0 <= axis < len(_b().grid):
        raise DSLError(f"grid axis {axis!r} out of range for a {len(_b().grid)}-D grid")
    return axis


def program_id(axis: int) -> Tile:
    _axis(axis)
    t = _b().new("i32", (), "RMEM")
    _b().emit("program_id", [], [t], "none", {"axis": axis})
    return t


def num_programs(axis: int) -> Tile:
    _axis(axis)
    t = _b().new("i32", (), "RMEM")
    _b().emit("num_programs", [], [t], "none", {"axis": axis})
    return t


# ---------------------------------------------------------------- 15 gather

def gather_load(src: Tile, indices: Tile, dst: Tile) -> None:
    """Rows of src at data-dependent indices; the interpreter clamps them."""
    if src.value.space != "GMEM" or indices.value.dtype not in _INT:
        raise DSLError("gather_load needs a GMEM source and integer indices")
    if src.value.dtype != dst.value.dtype:
        raise DSLError("gather_load dtype mismatch")
    _b().emit("gather_load", [src.ref, indices.ref], [dst], "GMEM", {"coalesced": False})


# ---------------------------------------------------------------- 16 kernels and programs

class KernelDef:
    def __init__(self, fn: Any, grid: tuple[Dim, ...], threads: int):
        if not isinstance(threads, int) or threads <= 0:
            raise DSLError("threads must be a positive int")
        self.fn, self.grid, self.threads = fn, _shape(grid), threads
        self.name = fn.__name__
        self._built: Kernel | None = None

    def param_types(self) -> list[tuple[str, ParamType]]:
        out = []
        for p in inspect.signature(self.fn).parameters.values():
            # Never eval string annotations (eval_str): that is an exec of arbitrary code.
            if not isinstance(p.annotation, ParamType) or p.annotation.space != "GMEM":
                raise DSLError(f"{self.name}: param {p.name} needs a GMEM[dtype, shape] annotation "
                               "(string annotations and `from __future__ import annotations` are not supported)")
            out.append((p.name, p.annotation))
        return out

    def build(self) -> Kernel:
        if self._built is None:
            b = _Builder(self.grid)
            args, params = [], []
            for name, pt in self.param_types():
                t = b.new(pt.dtype, pt.shape, "GMEM")
                args.append(t)
                params.append({"name": name, "value": t.value.id})
            _builders.append(b)
            try:
                if self.fn(*args) is not None:
                    raise DSLError(f"{self.name}: kernels write outputs with store(), they return nothing")
            finally:
                _builders.pop()
            self._built = Kernel(self.name, params, self.grid, self.threads, b.values, b.blocks[0], {})
        return self._built

    def __call__(self, *args: Buf) -> None:
        """Inside a @program body: record a launch."""
        if not _programs or _builders:
            raise DSLError(f"{self.name}: launch kernels only from a @program body; use .build() for IR")
        types = self.param_types()
        if len(args) != len(types):
            raise DSLError(f"{self.name}: expected {len(types)} args, got {len(args)}")
        for a, (pname, pt) in zip(args, types, strict=True):
            if not isinstance(a, Buf) or a.dtype != pt.dtype or len(a.shape) != len(pt.shape):
                raise DSLError(f"{self.name}: arg for {pname} must be a {pt.dtype} rank-{len(pt.shape)} buffer")
        prog = _programs[-1]
        if self.name not in prog["kernels"]:
            prog["kernels"][self.name] = self.build()
        prog["launches"].append({"kernel": self.name, "args": [a.name for a in args]})


@dataclass(frozen=True)
class Buf:
    name: str
    dtype: str
    shape: tuple[Dim, ...]


def kernel(grid: tuple[Dim, ...], threads: int) -> Any:
    return lambda fn: KernelDef(fn, grid, threads)


def buffer(shape: tuple[Dim, ...], dtype: str) -> Buf:
    """GMEM temporary passed between launches of a @program."""
    if not _programs:
        raise DSLError("buffer() is only allowed in a @program body")
    _check_dtype(dtype)
    prog = _programs[-1]
    b = Buf(f"tmp{len(prog['buffers'])}", dtype, _shape(shape))
    prog["buffers"].append({"name": b.name, "dtype": dtype, "shape": list(b.shape)})
    return b


class ProgramDef:
    def __init__(self, fn: Any):
        self.fn, self.name = fn, fn.__name__

    def build(self) -> Program:
        state: dict[str, Any] = {"kernels": {}, "launches": [], "buffers": []}
        args, params = [], []
        for p in inspect.signature(self.fn).parameters.values():
            pt = p.annotation
            if not isinstance(pt, ParamType) or pt.space != "GMEM":
                raise DSLError(f"{self.name}: param {p.name} needs a GMEM[dtype, shape] annotation")
            args.append(Buf(p.name, pt.dtype, pt.shape))
            params.append({"name": p.name, "dtype": pt.dtype, "shape": list(pt.shape)})
        _programs.append(state)
        try:
            self.fn(*args)
        finally:
            _programs.pop()
        if not state["launches"]:
            raise DSLError(f"{self.name}: a program needs at least one launch")
        return Program(self.name, params, state["buffers"], list(state["kernels"].values()), state["launches"])


def program(fn: Any) -> ProgramDef:
    return ProgramDef(fn)


# Spec names (§8.3) that shadow Python builtins; kernel code never sees the builtins anyway.
max, min, abs = _max, _min, _abs  # noqa: A001
