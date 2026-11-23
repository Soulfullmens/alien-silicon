"""Judge entry point for IR JSON (§8.6): the only form of a kernel the judge trusts.

The sandbox hands the judge IR JSON, which could be hand-crafted to skip every
builder check (wrong dtypes, unknown ops, dangling refs, extra attrs). So we
replay it through the DSL builder and accept it only if the rebuilt IR is
byte-identical. Anything the DSL could not have produced is rejected.
"""
from __future__ import annotations

import json
from typing import Any

from aliensilicon import ir
from aliensilicon.ir import Block, Kernel, Program
from aliensilicon.ir import dsl as _dsl

from .static_check import Rejected

MAX_IR_BYTES = 2_000_000

_ELEMENTWISE = {
    "add": ir.add, "sub": ir.sub, "mul": ir.mul, "fma": ir.fma, "neg": ir.neg,
    "max": ir.max, "min": ir.min, "abs": ir.abs,
    "iadd": ir.iadd, "isub": ir.isub, "imul": ir.imul,
    "and": ir.and_, "or": ir.or_, "xor": ir.xor, "shl": ir.shl, "shr": ir.shr,
}
_ERRORS = (ir.DSLError, KeyError, IndexError, TypeError, ValueError, AttributeError, RecursionError)


def load_kernel(text: str) -> Kernel:
    k = _parse(text, Kernel.from_json)
    _same(k, _rebuild(k), k.name)
    return k


def load_program(text: str) -> Program:
    p = _parse(text, Program.from_json)
    names = {k.name for k in p.kernels}
    if len(names) != len(p.kernels):
        raise Rejected("program has duplicate kernel names")
    for k in p.kernels:
        _same(k, _rebuild(k), k.name)
    buffers = {b["name"] for b in p.params + p.buffers}
    kernels = {k.name: k for k in p.kernels}
    for i, launch in enumerate(p.launches):
        k = kernels.get(launch.get("kernel"))
        args = launch.get("args")
        if k is None or not isinstance(args, list) or len(args) != len(k.params) or not set(args) <= buffers:
            raise Rejected(f"launch {i} is malformed")
    if not p.launches:
        raise Rejected("program has no launches")
    return p


def _parse(text: str, from_json: Any) -> Any:
    if len(text.encode()) > MAX_IR_BYTES:
        raise Rejected(f"IR larger than {MAX_IR_BYTES} bytes")
    try:
        return from_json(text)
    except (json.JSONDecodeError, *_ERRORS) as e:
        raise Rejected(f"malformed IR: {type(e).__name__}: {e}") from None


def _same(k: Kernel, rebuilt: Kernel, name: str) -> None:
    if rebuilt.to_json() != k.to_json():
        raise Rejected(f"{name}: IR is not what the DSL builds for these ops")


def _rebuild(k: Kernel) -> Kernel:
    try:
        b = _dsl._Builder(_dsl._shape(k.grid))
        env: dict[int, Any] = {}
        params = []
        for p in k.params:
            v = k.values[p["value"]]
            pt = ir.GMEM[v.dtype, v.shape]  # validates dtype and shape
            t = b.new(pt.dtype, pt.shape, "GMEM")
            env[p["value"]] = t
            params.append({"name": p["name"], "value": t.value.id})
        _dsl._builders.append(b)
        try:
            _walk(k, k.body, env)
        finally:
            _dsl._builders.pop()
        threads = k.threads
        if not isinstance(threads, int) or isinstance(threads, bool) or threads <= 0:
            raise ir.DSLError("threads must be a positive int")
        return Kernel(k.name, params, k.grid, threads, b.values, b.blocks[0], {})
    except _ERRORS as e:
        raise Rejected(f"{getattr(k, 'name', '?')}: invalid IR: {type(e).__name__}: {e}") from None


def _walk(k: Kernel, block: Block, env: dict[int, Any]) -> None:
    def ref(x: Any) -> Any:
        return env[int(x[1:])] if isinstance(x, str) and x.startswith("%") else x

    def idx(item: Any) -> Any:
        if isinstance(item, dict):
            start, stop = item["slice"]
            return slice(None if start is None else ref(start), None if stop is None else ref(stop))
        return ref(item)

    def view(tile: Any, index: list[Any] | None) -> Any:
        return tile if index is None else tile[tuple(idx(i) for i in index)]

    for op in block.ops:
        a, ins = op.attrs, [ref(x) for x in op.inputs]
        out = op.outputs[0] if op.outputs else None
        kind = op.kind
        if kind == "alloc":
            v = k.values[out]
            env[out] = ir.alloc(v.space, v.shape, v.dtype, init=a.get("init"), pad=a.get("pad"),
                                swizzle=a.get("swizzle"), stride=a.get("stride"))
        elif kind in ("load", "store", "async_copy"):
            src, dst = view(ins[0], a.get("src_index")), view(env[out], a.get("dst_index"))
            if kind == "async_copy":
                ir.async_copy(src, dst, a["group"])
            else:
                layout = {key: a.get(key) for key in ("pad", "swizzle", "stride")}
                getattr(ir, kind)(src, dst, **layout)
        elif kind == "wait":
            ir.wait(a["group"])
        elif kind == "barrier":
            ir.barrier()
        elif kind == "mma":
            ir.mma(*ins, transpose_b=a["transpose_b"], clear=a.get("clear", False))
        elif kind in _ELEMENTWISE:
            env[out] = _ELEMENTWISE[kind](*ins)
        elif kind == "cmp":
            env[out] = ir.cmp(a["op"], *ins)
        elif kind == "select":
            env[out] = ir.select(*ins)
        elif kind == "mufu":
            env[out] = ir.mufu(a["op"], *ins)
        elif kind in ("convert", "bitcast"):
            env[out] = getattr(ir, kind)(ins[0], a["dtype"])
        elif kind == "reduce":
            env[out] = ir.reduce(a["op"], ins[0], a["axis"], a["scope"], a["keepdims"])
        elif kind in ("program_id", "num_programs"):
            env[out] = getattr(ir, kind)(a["axis"])
        elif kind == "gather_load":
            ir.gather_load(ins[0], ins[1], env[out])
        elif kind == "for":
            for i in ir.serial(a["n"], a["unroll"]):
                env[out] = i
                _walk(k, op.blocks[0], env)
        elif kind == "if":
            with ir.if_(ins[0]):
                _walk(k, op.blocks[0], env)
        else:
            raise ir.DSLError(f"unknown op kind {kind!r}")
