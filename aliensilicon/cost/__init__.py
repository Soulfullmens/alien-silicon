"""Cost model v1 (§11.1 + §11.3 capacity effects + §12 physics): kernel x chip -> cycles + profile.

Work is counted by running the kernel in the interpreter (so loop trips and branches
are exact), then turned into time per unit and memory level:

  t[u]   = work on the busiest SM / per-SM throughput        u in mma, fma, mufu, int, imul, reduce
  t[m]   = bytes / bandwidth                                  m in gmem, l2 (chip-wide), smem_read, smem_write, spill
  core   = max(t) + (1 - overlap) * (sum(t) - max(t))         overlap grows with resident warps
  cycles = core + barriers + exposed gmem latency + launch overhead

Modelled: shared-memory bank conflicts for any bank count and padding, register
pressure by liveness analysis with spills, occupancy, MMA tile-shape and dtype support
(unsupported shapes fall back to FMA speed), missing units (reject), L2 reuse when tensors fit.
Not modelled yet (v2 scheduler, Weeks 4-5): instruction-level overlap and pipelining,
async copy hiding latency, warp scheduling, divergence.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from aliensilicon.chips import ChipSpec
from aliensilicon.interp import NP_DTYPE, resolve_dims, run_kernel
from aliensilicon.ir import Block, Kernel, Op

SAMPLE_CTAS = 8  # CTAs actually interpreted; work is scaled to the full grid
MAX_OVERLAP = 0.9  # even a fully occupied SM never overlaps units perfectly
REG_OVERHEAD = 16  # addresses, loop counters, predicates
GATHER_PENALTY = 2.0  # uncoalesced rows cost twice the bytes (sector waste)

_FMA = {"add", "sub", "mul", "fma", "max", "min", "neg", "abs", "cmp", "select", "convert"}
_INT = {"iadd", "isub", "and", "or", "xor", "shl", "shr"}


class ResourceError(ValueError):
    """The kernel cannot run on this chip at all (missing unit, too much SMEM, ...)."""


@dataclass
class Perf:
    chip: str
    cycles: float
    time_us: float
    bottleneck: str
    breakdown: dict[str, float]  # cycles per resource, plus barrier / latency / launch
    work: dict[str, float]  # totals over all CTAs: lane-ops, FLOPs, bytes
    occupancy: dict[str, Any]
    hotspots: list[dict[str, Any]]  # the ops that cost the most on the bottleneck unit
    flags: list[str] = field(default_factory=list)
    lower_bound_cycles: float = 0.0
    frac_lb: float = 0.0
    physics: str = "ok"  # ok | soft_flag | hard_violation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate(kernel: Kernel, chip: ChipSpec, inputs: dict[str, np.ndarray], dims: dict[str, int] | None = None,
             *, min_flops: float | None = None) -> Perf:
    """Run `kernel` on `inputs`, count its work, and price it on `chip`."""
    dims = resolve_dims(kernel, inputs, dims)
    res = resources(kernel, chip, dims)
    counter = _Counter(kernel, chip, dims)
    n_ctas = math.prod(_concrete(kernel.grid, dims))
    # shortcut: evenly spaced CTA sample assumes CTAs do similar work; fine for W1-W3, revisit for causal/MoE.
    sample = sorted({round(i * (n_ctas - 1) / max(1, SAMPLE_CTAS - 1)) for i in range(min(n_ctas, SAMPLE_CTAS))})
    out = run_kernel(kernel, inputs, dims, mufu_ulp_error=chip.mufu.ulp_error, mma_k_chunk=chip.mma_k_chunk(),
                     observer=counter, ctas=sample)
    w = {k: v * n_ctas / len(sample) for k, v in counter.work.items()}
    waves = math.ceil(n_ctas / chip.num_sm)  # CTAs on the busiest SM
    per_sm = {k: v / n_ctas * waves for k, v in w.items()}
    # L2 reuse (§11.3): if the tensors fit in L2, bytes beyond one pass over them are L2 hits, not DRAM.
    footprint = sum(np.asarray(a).nbytes for a in out.values())
    dram_bytes = w["gmem_bytes"] if footprint > chip.l2_bytes else min(w["gmem_bytes"], footprint)
    l2_bytes = w["gmem_bytes"] - dram_bytes

    t = {
        "mma": per_sm["mma_flops"] / chip.mma.ops_per_clk_per_sm,
        "fma": (per_sm["fma_f32"] / chip.fma.f32_ops_per_clk_per_sm
                + per_sm["fma_f16"] / chip.fma.f16_ops_per_clk_per_sm),
        "mufu": per_sm["mufu"] / chip.mufu.ops_per_clk_per_sm if per_sm["mufu"] else 0.0,
        "int": per_sm["int"] / chip.int.ops_per_clk_per_sm,
        "imul": per_sm["imul"] / chip.int.imul_ops_per_clk_per_sm,
        "reduce": per_sm["reduce_clk"] / max(1, min(res["ctas_per_sm"], waves)),
        "gmem": dram_bytes / chip.gmem_bw_bytes_per_clk,
        "l2": l2_bytes / chip.l2_bw_bytes_per_clk,
        "smem_read": per_sm["smem_read_bytes"] / chip.smem_read_bw_bytes_per_clk_per_sm,
        "smem_write": per_sm["smem_write_bytes"] / chip.smem_write_bw_bytes_per_clk_per_sm,
        "spill": w["spill_bytes"] / chip.l2_bw_bytes_per_clk,
    }
    resident_warps = min(res["ctas_per_sm"], waves) * math.ceil(kernel.threads / chip.warp_size)
    overlap = min(MAX_OVERLAP, resident_warps / chip.latency_hiding_warps)
    peak, total = max(t.values()), sum(t.values())
    core = peak + (1 - overlap) * (total - peak)
    concurrent = max(1, min(res["ctas_per_sm"], waves))
    extra = {
        "barrier": per_sm["barriers"] * chip.barrier_clk / concurrent,
        "latency": chip.gmem_latency_clk * math.ceil(waves / concurrent),
        "launch": chip.launch_overhead_clk,
    }
    cycles = core + sum(extra.values())
    bottleneck = max(t, key=t.__getitem__)

    lb_mem = footprint / chip.gmem_bw_bytes_per_clk  # every input read and output written once
    lb_compute = 0.0
    if min_flops:
        lb_compute = min_flops / (chip.num_sm * max(chip.mma.ops_per_clk_per_sm, 2 * chip.fma.f32_ops_per_clk_per_sm))
    physics = "ok"
    if cycles < 0.98 * lb_mem:
        physics = "hard_violation"  # cannot happen for honest kernels: inputs must be read once
    elif cycles < 0.98 * lb_compute:
        physics = "soft_flag"  # beat the naive compute bound: maybe a Level-C rewrite, send to review
    flags = sorted(counter.flags)
    if res["spill_regs"]:
        flags.append(f"register_spill:{res['spill_regs']}_regs")
    if n_ctas < chip.num_sm:
        flags.append(f"grid_underfills_chip:{n_ctas}_of_{chip.num_sm}_SMs")
    return Perf(
        chip=chip.name, cycles=cycles, time_us=cycles / (chip.clock_ghz * 1e3), bottleneck=bottleneck,
        breakdown={k: round(v, 1) for k, v in {**t, **extra}.items()},
        work={k: float(v) for k, v in w.items()},
        occupancy={**res, "resident_warps": resident_warps, "overlap": round(overlap, 3), "ctas": n_ctas,
                   "waves": waves, "sampled_ctas": len(sample)},
        hotspots=counter.hotspots(bottleneck),
        flags=flags, lower_bound_cycles=max(lb_mem, lb_compute), frac_lb=max(lb_mem, lb_compute) / cycles,
        physics=physics,
    )


def resources(kernel: Kernel, chip: ChipSpec, dims: dict[str, int]) -> dict[str, Any]:
    """Static per-CTA resource use and occupancy. Raises ResourceError if the kernel cannot run."""
    ops = list(_walk(kernel.body))
    errors = []
    if kernel.threads > chip.max_threads_per_cta:
        errors.append(f"{kernel.threads} threads > chip max {chip.max_threads_per_cta} per CTA")
    for op in ops:
        if op.kind == "mufu" and op.attrs["op"] not in chip.mufu.ops:
            errors.append(f"chip has no special-function unit for {op.attrs['op']}")
        if op.kind in ("async_copy", "wait") and not chip.async_copy:
            errors.append("chip has no async copy")
    smem = tmem = 0
    for op in ops:
        if op.kind == "alloc":
            v = kernel.values[op.outputs[0]]
            nbytes = _alloc_bytes(v, op, dims)
            smem += nbytes if v.space == "SMEM" else 0
            tmem += nbytes if v.space == "TMEM" else 0
    if smem > chip.smem_per_cta_max_bytes:
        errors.append(f"{smem} B shared memory > chip max {chip.smem_per_cta_max_bytes} B per CTA")
    if tmem > chip.tmem_bytes_per_sm:
        errors.append(f"{tmem} B tensor memory > chip's {chip.tmem_bytes_per_sm} B")
    if errors:
        raise ResourceError("; ".join(sorted(set(errors))))
    regs = register_pressure(kernel, dims) + REG_OVERHEAD
    used = min(regs, chip.max_regs_per_thread)
    threads = math.ceil(kernel.threads / chip.warp_size) * chip.warp_size
    limits = {
        "registers": chip.regs_per_sm // (used * threads),
        "shared_memory": chip.smem_per_sm_bytes // smem if smem else chip.max_ctas_per_sm,
        "threads": chip.max_threads_per_sm // threads,
        "ctas": chip.max_ctas_per_sm,
    }
    ctas_per_sm = min(limits.values())
    if ctas_per_sm == 0:
        raise ResourceError(f"one CTA does not fit on an SM (limits: {limits})")
    return {"regs_per_thread": regs, "spill_regs": max(0, regs - chip.max_regs_per_thread), "smem_bytes": smem,
            "tmem_bytes": tmem, "ctas_per_sm": ctas_per_sm, "occupancy_limiter": min(limits, key=limits.__getitem__)}


def register_pressure(kernel: Kernel, dims: dict[str, int]) -> int:
    """Peak live RMEM registers per thread (§8.2b: tiles are split evenly over the CTA's threads)."""
    born: dict[int, int] = {}
    last: dict[int, int] = {}
    pos = 0

    def refs(op: Op) -> list[int]:
        out = [int(x[1:]) for x in op.inputs if isinstance(x, str)]
        for key in ("src_index", "dst_index"):
            for item in op.attrs.get(key) or []:
                for x in (item["slice"] if isinstance(item, dict) else [item]):
                    if isinstance(x, str):
                        out.append(int(x[1:]))
        if op.kind in ("load", "store", "async_copy", "mma", "gather_load"):
            out += op.outputs  # writes into an existing tile keep it alive
        return out

    def walk(b: Block) -> None:
        nonlocal pos
        for op in b.ops:
            pos += 1
            if op.kind == "for":
                start = pos
                born.setdefault(op.outputs[0], pos)
                walk(op.blocks[0])
                for v, p in list(last.items()):  # values from outside the loop stay live through it
                    if p > start and born.get(v, 0) < start:
                        last[v] = pos
                continue
            if op.kind == "if":
                walk(op.blocks[0])
            for v in refs(op):
                last[v] = pos
            if op.kind not in ("load", "store", "async_copy", "mma", "gather_load"):
                for v in op.outputs:
                    born.setdefault(v, pos)

    walk(kernel.body)
    events: list[tuple[int, int]] = []
    for v, b in born.items():
        val = kernel.values[v]
        if val.space != "RMEM":
            continue
        n = math.prod(_concrete(val.shape, dims))
        per_thread = 1 if n == 1 else math.ceil(n / kernel.threads)
        regs = math.ceil(per_thread * np.dtype(NP_DTYPE[val.dtype]).itemsize / 4)
        events += [(b, regs), (last.get(v, b) + 1, -regs)]
    live = peak = 0
    for _, d in sorted(events):
        live += d
        peak = max(peak, live)
    return peak


class _Counter:
    """Interpreter observer: adds up the work of every executed op, per unit and memory level."""

    def __init__(self, kernel: Kernel, chip: ChipSpec, dims: dict[str, int]):
        self.k, self.chip, self.dims = kernel, chip, dims
        self.work = dict.fromkeys(("mma_flops", "fma_f32", "fma_f16", "mufu", "int", "imul", "reduce_clk",
                                   "gmem_bytes", "smem_read_bytes", "smem_write_bytes", "spill_bytes",
                                   "barriers"), 0.0)
        self.flags: set[str] = set()
        self.pad = {op.outputs[0]: op.attrs.get("pad", 0) for op in _walk(kernel.body) if op.kind == "alloc"}
        self.index = {id(op): i for i, op in enumerate(_walk(kernel.body))}
        self.per_op: dict[tuple[int, str], float] = {}
        self._conflict_cache: dict[tuple[int, str], float] = {}
        res = resources(kernel, chip, dims)
        self.spill_per_op = res["spill_regs"] * 4 * kernel.threads  # bytes moved per op when spilling

    def lanes(self, n: int) -> int:
        """Lane-ops an n-element op issues: whole warps only, every thread for uniform scalars."""
        threads, warp = self.k.threads, self.chip.warp_size
        if n >= threads or n <= 1:
            return math.ceil(max(n, 1) / threads) * threads
        return math.ceil(n / warp) * warp

    def add(self, op: Op, key: str, amount: float) -> None:
        self.work[key] += amount
        slot = (self.index[id(op)], key)
        self.per_op[slot] = self.per_op.get(slot, 0.0) + amount

    def __call__(self, cta: Any, op: Op) -> None:
        kind = op.kind
        out = cta.env[op.outputs[0]] if op.outputs else None
        if self.spill_per_op and kind not in ("for", "if", "program_id", "num_programs", "barrier", "wait"):
            self.add(op, "spill_bytes", self.spill_per_op)
        if kind in _FMA:
            self.add(op, "fma_f16" if out.dtype == np.float16 else "fma_f32", self.lanes(out.size))
        elif kind in _INT:
            self.add(op, "int", self.lanes(out.size))
        elif kind == "imul":
            self.add(op, "imul", self.lanes(out.size))
        elif kind == "mufu":
            self.add(op, "mufu", self.lanes(out.size))
        elif kind == "reduce":
            src = cta.env[int(op.inputs[0][1:])]
            self.add(op, "fma_f16" if src.dtype == np.float16 else "fma_f32", self.lanes(src.size))
            self.add(op, "reduce_clk", self.reduce_latency(src.shape, op.attrs["axis"]))
        elif kind == "mma":
            self.mma(cta, op)
        elif kind in ("load", "store", "async_copy"):
            self.move(cta, op)
        elif kind == "gather_load":
            self.add(op, "gmem_bytes", out.nbytes * GATHER_PENALTY)
        elif kind == "alloc" and "init" in op.attrs:
            self.add(op, "fma_f16" if out.dtype == np.float16 else "fma_f32", self.lanes(out.size))
            if self.k.values[op.outputs[0]].space == "SMEM":
                self.add(op, "smem_write_bytes", out.nbytes)
        elif kind == "barrier":
            self.add(op, "barriers", 1)

    def reduce_latency(self, shape: tuple[int, ...], axis: int) -> float:
        """Shuffle / cross-warp cost of one reduction; all outputs reduce in parallel.

        With the default row-major mapping (§8.2b) each thread holds e consecutive elements, so one
        output's elements are spread over `tpo` threads. Cost follows that mapping, not the declared scope.
        """
        warp = self.chip.warp_size
        e = math.ceil(math.prod(shape) / self.k.threads)
        inner = math.prod(shape[axis + 1:])
        tpo = min(self.k.threads, math.ceil(shape[axis] / max(1, e // inner)))
        if tpo <= 1:
            return 0.0
        shuffle = self.chip.reduce.warp_shuffle_clk
        if tpo <= warp:
            return math.ceil(math.log2(tpo)) * shuffle
        return math.log2(warp) * shuffle + self.chip.reduce.cta_reduce_clk_per_warp * math.ceil(tpo / warp)

    def mma(self, cta: Any, op: Op) -> None:
        a, b, acc = (cta.env[int(x[1:])] for x in op.inputs)
        m, k = a.shape
        n = acc.shape[1]
        flops = 2 * m * n * k
        dt = {np.dtype(np.float16): "f16", np.dtype(np.int8): "i8", np.dtype(np.float32): "f32"}
        ok_dtype = dt.get(a.dtype) in self.chip.mma.in_dtypes and dt.get(acc.dtype) in self.chip.mma.acc_dtypes
        ok_shape = any(m % sm == 0 and n % sn == 0 and k % sk == 0 for sm, sn, sk in self.chip.mma.shapes)
        if ok_dtype and ok_shape:
            self.add(op, "mma_flops", flops)
        else:
            self.flags.add("mma_fallback_to_fma")  # silently slow, like a real tensor-core miss
            self.add(op, "fma_f32", flops / 2)
        for x, arr in zip(op.inputs[:2], (a, b), strict=True):
            if self.k.values[int(x[1:])].space == "SMEM":
                self.add(op, "smem_read_bytes", arr.nbytes)  # operand loads are conflict-free (ldmatrix-style)

    def move(self, cta: Any, op: Op) -> None:
        s_vid, s_idx, d_vid, d_idx = cta.last_move
        nbytes = cta.env[s_vid][s_idx].nbytes
        s_space, d_space = self.k.values[s_vid].space, self.k.values[d_vid].space
        if "GMEM" in (s_space, d_space):
            self.add(op, "gmem_bytes", nbytes)
        if s_space == "SMEM":
            self.add(op, "smem_read_bytes", self.bank_bytes(cta, s_vid, s_idx))
        if d_space == "SMEM":
            self.add(op, "smem_write_bytes", self.bank_bytes(cta, d_vid, d_idx))

    def bank_bytes(self, cta: Any, vid: int, idx: tuple[Any, ...]) -> float:
        """Bytes of SMEM bandwidth an access costs, including bank conflicts (any bank count, any padding)."""
        key = (vid, repr(idx))
        if key not in self._conflict_cache:
            self._conflict_cache[key] = smem_access_bytes(
                cta.env[vid].shape, cta.env[vid].dtype.itemsize, idx, self.pad.get(vid, 0),
                self.chip.smem_banks, self.chip.smem_bank_width_bytes, self.chip.warp_size)
        cost = self._conflict_cache[key]
        if cost > cta.env[vid][idx].nbytes * 1.01:
            self.flags.add("smem_bank_conflicts")
        return cost

    def hotspots(self, unit: str, top: int = 3) -> list[dict[str, Any]]:
        keys = {"mma": ["mma_flops"], "fma": ["fma_f32", "fma_f16"], "mufu": ["mufu"], "int": ["int"],
                "imul": ["imul"], "reduce": ["reduce_clk"], "gmem": ["gmem_bytes"], "l2": ["gmem_bytes"],
                "smem_read": ["smem_read_bytes"], "smem_write": ["smem_write_bytes"], "spill": ["spill_bytes"]}[unit]
        rows = [(i, key, v) for (i, key), v in self.per_op.items() if key in keys]
        total = sum(v for *_, v in rows) or 1.0
        ops = list(_walk(self.k.body))
        rows.sort(key=lambda r: -r[2])
        return [{"op": i, "kind": ops[i].kind, "detail": ops[i].attrs.get("op", ""), "share": round(v / total, 3)}
                for i, _, v in rows[:top]]


def smem_access_bytes(shape: tuple[int, ...], itemsize: int, idx: tuple[Any, ...], pad: int, banks: int,
                      bank_width: int, warp: int) -> float:
    """Cost of reading/writing region `idx` of a row-major SMEM tile, in bank-width bytes actually spent.

    Lanes take consecutive elements; one wavefront serves `warp` lanes. A wavefront costs as many
    bank cycles as the most-used bank has distinct words. Padding adds `pad` elements per row.
    """
    lin = np.arange(math.prod(shape)).reshape(shape)[idx].ravel()
    if lin.size == 0:
        return 0.0
    cols = shape[-1] if len(shape) > 1 else lin.size
    addr = ((lin // cols) * (cols + pad) + lin % cols) * itemsize if len(shape) > 1 else lin * itemsize
    words = addr // bank_width
    per_wave = warp * max(1, bank_width // itemsize)
    n_waves = math.ceil(words.size / per_wave)
    w = np.full(n_waves * per_wave, -1, dtype=np.int64)
    w[:words.size] = words
    w = np.sort(w.reshape(n_waves, per_wave), axis=1)
    distinct = np.ones_like(w, dtype=bool)
    distinct[:, 1:] = w[:, 1:] != w[:, :-1]
    distinct &= w >= 0
    rows, cols_ = np.nonzero(distinct)
    hits = np.zeros((n_waves, banks), dtype=np.int64)
    np.add.at(hits, (rows, w[rows, cols_] % banks), 1)
    return float(hits.max(axis=1).sum() * banks * bank_width)


def _walk(b: Block) -> Any:
    for op in b.ops:
        yield op
        for blk in op.blocks:
            yield from _walk(blk)


def _concrete(shape: tuple[Any, ...], dims: dict[str, int]) -> tuple[int, ...]:
    return tuple(dims[d] if isinstance(d, str) else d for d in shape)


def _alloc_bytes(v: Any, op: Op, dims: dict[str, int]) -> int:
    shape = _concrete(v.shape, dims)
    if op.attrs.get("pad") and len(shape) > 1:
        shape = (*shape[:-1], shape[-1] + op.attrs["pad"])
    return math.prod(shape) * np.dtype(NP_DTYPE[v.dtype]).itemsize
