"""Chip specs (§10): a declarative, bounded description of an AI chip.

Every numeric field has a [min, max] range in BOUNDS. The validator enforces it,
the alien generator samples inside it, and co-design search moves inside it, so
"edit the kernel" and "edit the chip" use the same contract.

Units: *_per_clk_per_sm rates are per streaming multiprocessor (SM) per clock;
gmem/l2 bandwidth is for the whole chip. MMA throughput counts FLOPs (2 per MAC).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROFILES = Path(__file__).parent / "profiles"
SCHEMA_VERSION = 1
MUFU_OPS = ("exp2", "log2", "rcp", "rsqrt", "sin", "cos")
FAMILIES = ("familiar", "control", "alien", "cliff", "future", "blackbox", "fallback")


@dataclass(frozen=True)
class Mma:
    shapes: tuple[tuple[int, int, int], ...]  # tile (m, n, k) multiples the tensor core accepts
    in_dtypes: tuple[str, ...]
    acc_dtypes: tuple[str, ...]
    ops_per_clk_per_sm: float  # FLOPs
    latency_clk: int


@dataclass(frozen=True)
class Fma:
    f32_ops_per_clk_per_sm: float
    f16_ops_per_clk_per_sm: float
    latency_clk: int


@dataclass(frozen=True)
class Mufu:
    ops: tuple[str, ...]  # may be empty: a chip with no special-function unit
    ops_per_clk_per_sm: float
    latency_clk: int
    ulp_error: float


@dataclass(frozen=True)
class Int:
    ops_per_clk_per_sm: float
    imul_ops_per_clk_per_sm: float
    latency_clk: int


@dataclass(frozen=True)
class Reduce:
    warp_shuffle_clk: int
    cta_reduce_clk_per_warp: int


@dataclass(frozen=True)
class ChipSpec:
    name: str
    family: str
    notes: str
    clock_ghz: float
    num_sm: int
    warp_size: int
    max_threads_per_cta: int
    max_threads_per_sm: int
    max_ctas_per_sm: int
    regs_per_sm: int
    max_regs_per_thread: int
    smem_per_sm_bytes: int
    smem_per_cta_max_bytes: int
    smem_banks: int
    smem_bank_width_bytes: int
    smem_read_bw_bytes_per_clk_per_sm: float
    smem_write_bw_bytes_per_clk_per_sm: float
    l2_bytes: int
    l2_bw_bytes_per_clk: float
    gmem_bw_bytes_per_clk: float
    gmem_latency_clk: int
    tmem_bytes_per_sm: int
    async_copy: bool
    max_pipeline_stages: int
    latency_hiding_warps: int  # resident warps needed for full overlap of units (fit in calibration)
    barrier_clk: int
    launch_overhead_clk: int
    mma: Mma
    fma: Fma
    mufu: Mufu
    int: Int
    reduce: Reduce

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def mma_k_chunk(self) -> int:
        """Accumulator rounding granularity for the interpreter: the smallest native K."""
        return min(k for _, _, k in self.mma.shapes)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ChipSpec:
        d = dict(d)
        if d.pop("schema_version", None) != SCHEMA_VERSION:
            raise ValueError(f"chip spec must have schema_version {SCHEMA_VERSION}")
        _check_keys("chip", d, ChipSpec)
        subs = {"mma": Mma, "fma": Fma, "mufu": Mufu, "int": Int, "reduce": Reduce}
        for key, cls in subs.items():
            _check_keys(key, d[key], cls)
        mma = dict(d["mma"], shapes=tuple(tuple(s) for s in d["mma"]["shapes"]),
                   in_dtypes=tuple(d["mma"]["in_dtypes"]), acc_dtypes=tuple(d["mma"]["acc_dtypes"]))
        chip = ChipSpec(**{**d, "mma": Mma(**mma), "fma": Fma(**d["fma"]),
                           "mufu": Mufu(**dict(d["mufu"], ops=tuple(d["mufu"]["ops"]))),
                           "int": Int(**d["int"]), "reduce": Reduce(**d["reduce"])})
        validate(chip)
        return chip

    @staticmethod
    def from_json(text: str) -> ChipSpec:
        return ChipSpec.from_dict(json.loads(text))


# [min, max] for every numeric field. Wide enough for real chips x0.25 .. x4 and the odd alien.
BOUNDS: dict[str, tuple[float, float]] = {
    "clock_ghz": (0.3, 4.0), "num_sm": (1, 1024), "warp_size": (4, 128),
    "max_threads_per_cta": (32, 4096), "max_threads_per_sm": (32, 8192), "max_ctas_per_sm": (1, 64),
    "regs_per_sm": (1024, 1 << 20), "max_regs_per_thread": (16, 1024),
    "smem_per_sm_bytes": (1024, 64 << 20), "smem_per_cta_max_bytes": (1024, 64 << 20),
    "smem_banks": (1, 128), "smem_bank_width_bytes": (1, 16),
    "smem_read_bw_bytes_per_clk_per_sm": (4, 4096), "smem_write_bw_bytes_per_clk_per_sm": (4, 4096),
    "l2_bytes": (0, 1 << 34), "l2_bw_bytes_per_clk": (16, 1 << 16), "gmem_bw_bytes_per_clk": (4, 1 << 16),
    "gmem_latency_clk": (1, 10000), "tmem_bytes_per_sm": (0, 64 << 20), "max_pipeline_stages": (1, 16),
    "latency_hiding_warps": (1, 128), "barrier_clk": (0, 1000), "launch_overhead_clk": (0, 1 << 20),
    "mma.ops_per_clk_per_sm": (16, 1 << 16), "mma.latency_clk": (1, 1000),
    "fma.f32_ops_per_clk_per_sm": (4, 4096), "fma.f16_ops_per_clk_per_sm": (4, 8192), "fma.latency_clk": (1, 100),
    "mufu.ops_per_clk_per_sm": (0, 1024), "mufu.latency_clk": (1, 1000), "mufu.ulp_error": (0, 64),
    "int.ops_per_clk_per_sm": (4, 4096), "int.imul_ops_per_clk_per_sm": (1, 4096), "int.latency_clk": (1, 100),
    "reduce.warp_shuffle_clk": (1, 1000), "reduce.cta_reduce_clk_per_warp": (1, 10000),
}


def validate(chip: ChipSpec) -> None:
    """Raise ValueError if the chip is out of bounds or physically inconsistent."""
    d = chip.to_dict()
    for path, (lo, hi) in BOUNDS.items():
        v = d
        for part in path.split("."):
            v = v[part]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
            raise ValueError(f"{chip.name}: {path} = {v!r} outside [{lo}, {hi}]")
    rules = [
        (chip.family in FAMILIES, f"family must be one of {FAMILIES}"),
        (chip.smem_per_cta_max_bytes <= chip.smem_per_sm_bytes, "smem_per_cta_max_bytes > smem_per_sm_bytes"),
        (chip.max_threads_per_cta <= chip.max_threads_per_sm, "max_threads_per_cta > max_threads_per_sm"),
        (chip.max_threads_per_cta % chip.warp_size == 0, "max_threads_per_cta must be a multiple of warp_size"),
        (chip.mma.ops_per_clk_per_sm >= chip.fma.f32_ops_per_clk_per_sm, "MMA slower than FMA"),
        (set(chip.mufu.ops) <= set(MUFU_OPS), f"mufu ops must be from {MUFU_OPS}"),
        (bool(chip.mufu.ops) == (chip.mufu.ops_per_clk_per_sm > 0), "mufu ops and mufu throughput disagree"),
        (bool(chip.mma.shapes) and all(len(s) == 3 and min(s) > 0 for s in chip.mma.shapes), "bad mma shapes"),
        (set(chip.mma.in_dtypes) <= {"f16", "i8"} and set(chip.mma.acc_dtypes) <= {"f16", "f32", "i32"},
         "bad mma dtypes"),
        (chip.async_copy or chip.max_pipeline_stages == 1, "pipeline stages > 1 need async_copy"),
    ]
    for ok, msg in rules:
        if not ok:
            raise ValueError(f"{chip.name}: {msg}")


def load_chip(name_or_path: str | Path) -> ChipSpec:
    p = Path(name_or_path)
    if not p.suffix:
        p = PROFILES / f"{name_or_path}.json"
    if not p.exists():
        raise ValueError(f"unknown chip {name_or_path!r}; profiles: {', '.join(list_chips())}")
    return ChipSpec.from_json(p.read_text())


def list_chips() -> list[str]:
    return sorted(p.stem for p in PROFILES.glob("*.json"))


def _check_keys(where: str, d: dict[str, Any], cls: type) -> None:
    want = set(cls.__dataclass_fields__)
    have = set(d)
    if have != want:
        raise ValueError(f"{where}: missing {sorted(want - have)}, unknown {sorted(have - want)}")
