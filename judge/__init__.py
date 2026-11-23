"""Deterministic judge (§14). Read-only for agents. No LLM ever decides pass/fail.

`evaluate(source, task)` is the full pipeline for one submission:
  1. build in the sandbox, AST allow-list inside it   (judge.sandbox, judge.static_check)
  2. re-validate the returned IR by replaying it       (judge.ir_check)
  3. prove index bounds for every valid shape          (judge.bounds)
  4. hidden tests over every task shape                (judge.correctness)
Every result carries the judge hash, so a number can always be traced to the code that made it.
"""
from __future__ import annotations

import hashlib
import platform
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
# Everything that can change a verdict.
_HASHED = ("judge", "aliensilicon/ir", "aliensilicon/interp", "aliensilicon/tasks")


def judge_hash() -> str:
    h = hashlib.sha256()
    for d in _HASHED:
        for f in sorted((_ROOT / d).rglob("*.py")):
            h.update(str(f.relative_to(_ROOT)).encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def provenance() -> dict[str, Any]:
    import numpy as np  # lazy: the sandbox child imports this package without site-packages

    from .sandbox import isolation

    return {"judge_sha256": judge_hash(), "numpy": np.__version__, "python": platform.python_version(),
            "sandbox": isolation()}


def evaluate(source: str, task: Any, chip: Any = None, *, seed: int = 0, mufu_ulp_error: float = 0.0,
             mma_k_chunk: int | None = None) -> dict[str, Any]:
    """Judge one submission (DSL source with exactly one kernel). Never raises on bad input.

    With a `chip` (ChipSpec), the kernel must also fit the chip (stage "resources"), correctness uses the
    chip's numerics (MUFU error, MMA accumulator rounding), and the record gains `perf`: cycles, time,
    bottleneck, per-unit breakdown, occupancy and hotspots on task.perf_shape, plus `cheat_flags`.
    """
    from .bounds import prove_bounds
    from .correctness import passed, run_tests
    from .sandbox import build_in_sandbox
    from .static_check import Rejected

    record: dict[str, Any] = {"task": task.id, "seed": seed, "provenance": provenance(), "correct": False}
    try:
        built = build_in_sandbox(source)
    except Rejected as e:
        return {**record, "stage": "static", "detail": str(e)}
    kernels = list(built.values())
    if len(kernels) != 1:
        return {**record, "stage": "static", "detail": f"expected exactly one kernel, got {len(kernels)}"}
    kernel = kernels[0]
    record["ir_sha256"] = kernel.sha256()
    bounds = prove_bounds(kernel, task)
    record["bounds"] = bounds
    if bounds["status"] == "failed":
        return {**record, "stage": "bounds", "detail": bounds["detail"]}
    if chip is not None:
        from aliensilicon.cost import ResourceError, resources

        record["chip"] = chip.name
        mufu_ulp_error, mma_k_chunk = chip.mufu.ulp_error, chip.mma_k_chunk()
        try:
            for shape in (*task.shapes, task.perf_shape):
                resources(kernel, chip, {**shape, **task.dims(shape)})
        except ResourceError as e:
            return {**record, "stage": "resources", "detail": str(e)}
    tests = run_tests(task, kernel, seed=seed, mufu_ulp_error=mufu_ulp_error, mma_k_chunk=mma_k_chunk)
    record["tests"] = tests
    ok = passed(tests)
    if ok and chip is not None:
        from aliensilicon.cost import estimate

        shape = task.perf_shape
        perf = estimate(kernel, chip, task.make_inputs("D1", seed, shape), task.dims(shape),
                        min_flops=task.min_flops(shape))
        record["perf"] = perf.to_dict()
        hard = perf.physics == "hard_violation"
        record["cheat_flags"] = ["physics: faster than reading the inputs once"] if hard else []
        ok = not record["cheat_flags"]
    return {**record, "stage": "done" if ok else "tests", "correct": ok}
