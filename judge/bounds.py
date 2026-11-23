"""Judge step 2 (§14.1): prove index bounds for ALL valid shapes with Z3.

Runtime checks only cover the shapes we test. Here every index and slice of
every load/store/async_copy is checked symbolically: shape dims are unknown
positive integers bounded only by the task's constraints, program ids and loop
indices range over their whole domain. For each access we also prove the source
and destination extents agree, so a kernel cannot be right only for test shapes.

Result:
  "proven"   every obligation holds for every valid shape
  "failed"   a concrete counterexample exists (the kernel is wrong for some shape)
  "unproven" Z3 timed out, or an index depends on data or ops we do not model;
             the runtime checks over all task shapes still apply
"""
from __future__ import annotations

from typing import Any

import z3

from aliensilicon.ir import Block, Kernel
from aliensilicon.tasks import Task

TIMEOUT_MS = 2000
_INT_OPS = {"iadd": lambda a, b: a + b, "isub": lambda a, b: a - b, "imul": lambda a, b: a * b}
_CMP = {"lt": lambda a, b: a < b, "le": lambda a, b: a <= b, "gt": lambda a, b: a > b,
        "ge": lambda a, b: a >= b, "eq": lambda a, b: a == b, "ne": lambda a, b: a != b}


def prove_bounds(kernel: Kernel, task: Task, timeout_ms: int = TIMEOUT_MS) -> dict[str, Any]:
    p = _Prover(kernel, task)
    p.block(kernel.body, [])
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    solver.add(*p.facts)
    unproven = []
    for what, cond, path, opaque in p.obligations:
        if opaque:
            unproven.append(f"{what}: depends on values the prover does not model")
            continue
        solver.push()
        solver.add(*path, z3.Not(cond))
        r = solver.check()
        if r == z3.sat:
            m = solver.model()
            example = {str(d): m[d].as_long() for d in m.decls() if not str(d).startswith(("pid", "i_", "v_"))}
            solver.pop()
            return {"status": "failed", "detail": f"{what} for shape {example}"}
        if r == z3.unknown:
            unproven.append(f"{what}: solver timeout")
        solver.pop()
    if unproven:
        return {"status": "unproven", "detail": unproven[:5]}
    return {"status": "proven", "detail": f"{len(p.obligations)} obligations"}


class _Prover:
    def __init__(self, k: Kernel, task: Task):
        self.k = k
        self.syms: dict[str, z3.ArithRef] = {}
        self.facts: list[Any] = []
        self.obligations: list[tuple[str, Any, list[Any], bool]] = []
        self.ints: dict[int, Any] = {}  # value id -> z3 expr for i32 scalars we can model
        self.bools: dict[int, Any] = {}
        self.opaque: set[str] = set()  # names of fresh vars standing for unmodelled values
        shape_syms = {d: self.sym(d) for s in (*task.inputs.values(), *task.outputs.values()) for d in s
                      if isinstance(d, str)}
        self.facts += task.constraints(shape_syms)
        for name, expr in task.grid(shape_syms).items():
            self.facts.append(self.sym(name) == expr)
        # Link the kernel's own dim names to the task's, by parameter name and position.
        for p in k.params:
            spec = task.inputs.get(p["name"]) or task.outputs.get(p["name"])
            if spec is None:
                continue
            for kd, td in zip(k.values[p["value"]].shape, spec, strict=False):
                self.facts.append(self.dim(kd) == self.dim(td))

    def sym(self, name: str) -> z3.ArithRef:
        if name not in self.syms:
            self.syms[name] = z3.Int(name)
            self.facts.append(self.syms[name] > 0)
        return self.syms[name]

    def dim(self, d: Any) -> Any:
        return self.sym(d) if isinstance(d, str) else z3.IntVal(d)

    def fresh(self, prefix: str, opaque: bool) -> z3.ArithRef:
        v = z3.Int(f"{prefix}{len(self.ints)}_{len(self.opaque)}_{len(self.obligations)}_{id(object())}")
        if opaque:
            self.opaque.add(str(v))
        return v

    def val(self, x: Any) -> Any:
        if isinstance(x, str):
            vid = int(x[1:])
            if vid not in self.ints:
                self.ints[vid] = self.fresh("v_", opaque=True)
            return self.ints[vid]
        return z3.IntVal(int(x)) if isinstance(x, int) else None

    def depends_on_opaque(self, e: Any) -> bool:
        return any(str(v) in self.opaque for v in z3.z3util.get_vars(e))

    def block(self, b: Block, path: list[Any]) -> None:
        for op in b.ops:
            out = op.outputs[0] if op.outputs else None
            scalar_int = out is not None and self.k.values[out].shape == () and self.k.values[out].dtype == "i32"
            if op.kind == "program_id":
                pid = self.fresh("pid", opaque=False)
                self.facts += [pid >= 0, pid < self.dim(self.k.grid[op.attrs["axis"]])]
                self.ints[out] = pid
            elif op.kind == "num_programs":
                self.ints[out] = self.dim(self.k.grid[op.attrs["axis"]])
            elif op.kind in _INT_OPS and scalar_int:
                a, c = (self.val(x) for x in op.inputs)
                self.ints[out] = _INT_OPS[op.kind](a, c)
            elif op.kind == "cmp" and all(self.is_int_scalar(x) for x in op.inputs):
                a, c = (self.val(x) for x in op.inputs)
                self.bools[out] = _CMP[op.attrs["op"]](a, c)
            elif op.kind == "for":
                i = self.fresh("i_", opaque=False)
                self.ints[out] = i
                n = self.dim(op.attrs["n"])
                self.block(op.blocks[0], [*path, i >= 0, i < n])
            elif op.kind == "if":
                cond = self.bools.get(int(op.inputs[0][1:]))
                self.block(op.blocks[0], path if cond is None else [*path, cond])
            elif op.kind in ("load", "store", "async_copy"):
                self.access(op, path)
            elif scalar_int:
                self.ints[out] = self.fresh("v_", opaque=True)  # e.g. a value loaded from memory

    def is_int_scalar(self, x: Any) -> bool:
        if isinstance(x, int):
            return True
        if not isinstance(x, str):
            return False
        v = self.k.values[int(x[1:])]
        return v.shape == () and v.dtype in ("i32", "i8")

    def access(self, op: Any, path: list[Any]) -> None:
        src, dst = self.k.values[int(op.inputs[0][1:])], self.k.values[op.outputs[0]]
        extents = []
        for side, v, idx in (("src", src, op.attrs.get("src_index")), ("dst", dst, op.attrs.get("dst_index"))):
            ext = []
            items = idx if idx is not None else [{"slice": [None, None]}] * len(v.shape)
            for axis, (item, d) in enumerate(zip(items, v.shape, strict=True)):
                size = self.dim(d)
                where = f"{op.kind} {side} %{v.id} axis {axis}"
                if isinstance(item, dict):
                    lo, hi = item["slice"]
                    lo = z3.IntVal(0) if lo is None else self.val(lo)
                    hi = size if hi is None else self.val(hi)
                    self.oblige(f"{where}: slice in bounds", z3.And(0 <= lo, lo <= hi, hi <= size), path)
                    ext.append(hi - lo)
                else:
                    i = self.val(item)
                    self.oblige(f"{where}: index in bounds", z3.And(0 <= i, i < size), path)
            extents.append(ext)
        s_ext, d_ext = extents
        if len(s_ext) != len(d_ext):
            self.obligations.append((f"{op.kind}: rank {len(s_ext)} into rank {len(d_ext)}", z3.BoolVal(False),
                                     path, False))
            return
        for axis, (a, c) in enumerate(zip(s_ext, d_ext, strict=True)):
            self.oblige(f"{op.kind} %{src.id}->%{dst.id} axis {axis}: extents match", a == c, path)

    def oblige(self, what: str, cond: Any, path: list[Any]) -> None:
        opaque = self.depends_on_opaque(cond) or any(self.depends_on_opaque(c) for c in path)
        self.obligations.append((what, cond, list(path), opaque))
