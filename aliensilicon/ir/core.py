"""Tile IR data structures (§8.5) with canonical JSON and SHA-256."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

IR_VERSION = 1

DTYPES = ("f16", "f32", "i32", "i8")
SPACES = ("GMEM", "SMEM", "RMEM", "TMEM")

# Shape dims are ints or symbolic names bound at launch ("rows", "cols").
Dim = int | str


@dataclass
class Value:
    id: int
    dtype: str
    shape: tuple[Dim, ...]
    space: str


@dataclass
class Op:
    kind: str
    # Each input is a value ref "%<id>", a number literal, or {"slice": [start, stop]}.
    inputs: list[Any]
    outputs: list[int]
    attrs: dict[str, Any] = field(default_factory=dict)
    unit: str = "none"
    # Nested bodies for `for` / `if` ops.
    blocks: list[Block] = field(default_factory=list)


@dataclass
class Block:
    ops: list[Op] = field(default_factory=list)


@dataclass
class Kernel:
    name: str
    params: list[dict[str, Any]]  # {"name", "value"} where value is a Value id
    grid: tuple[Dim, ...]
    threads: int
    values: list[Value]
    body: Block
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"ir_version": IR_VERSION, "type": "kernel", **asdict(self)}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def sha256(self) -> str:
        return sha256_hex(self.to_json())

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Kernel:
        _check_header(d, "kernel")
        return Kernel(
            name=d["name"],
            params=[dict(p) for p in d["params"]],
            grid=tuple(d["grid"]),
            threads=d["threads"],
            values=[_value(v) for v in d["values"]],
            body=_block(d["body"]),
            meta=dict(d["meta"]),
        )

    @staticmethod
    def from_json(s: str) -> Kernel:
        return Kernel.from_dict(json.loads(s))


@dataclass
class Program:
    """Ordered kernel launches sharing GMEM buffers (primitive group 16)."""

    name: str
    params: list[dict[str, Any]]  # {"name", "dtype", "shape"}
    buffers: list[dict[str, Any]]  # temporaries: {"name", "dtype", "shape"}
    kernels: list[Kernel]
    launches: list[dict[str, Any]]  # {"kernel": name, "args": [buffer names]}

    def to_dict(self) -> dict[str, Any]:
        return {"ir_version": IR_VERSION, "type": "program", **asdict(self)}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def sha256(self) -> str:
        return sha256_hex(self.to_json())

    @staticmethod
    def from_json(s: str) -> Program:
        d = json.loads(s)
        _check_header(d, "program")
        kernels = [Kernel.from_dict({"ir_version": IR_VERSION, "type": "kernel", **k}) for k in d["kernels"]]
        return Program(d["name"], d["params"], d["buffers"], kernels, d["launches"])


def canonical_json(d: dict[str, Any]) -> str:
    # Sorted keys, no whitespace: the same IR always hashes the same.
    return json.dumps(d, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _check_header(d: dict[str, Any], kind: str) -> None:
    if d.get("ir_version") != IR_VERSION or d.get("type") != kind:
        raise ValueError(f"expected {kind} IR v{IR_VERSION}, got {d.get('type')} v{d.get('ir_version')}")


def _value(d: dict[str, Any]) -> Value:
    return Value(d["id"], d["dtype"], tuple(d["shape"]), d["space"])


def _block(d: dict[str, Any]) -> Block:
    return Block([_op(o) for o in d["ops"]])


def _op(d: dict[str, Any]) -> Op:
    return Op(d["kind"], d["inputs"], d["outputs"], d["attrs"], d["unit"], [_block(b) for b in d["blocks"]])
