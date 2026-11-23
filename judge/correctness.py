"""Judge step 3 (§14.1): correctness on hidden tests, over every task shape.

Each (shape, test) runs the kernel twice: once in grid order, and once with
different garbage memory and a shuffled CTA order. An honest kernel gives
bit-identical output both times. A difference means it reads memory it never
wrote, or it depends on CTA order (a race on real hardware). The interpreter
separately raises on barrier, async-copy and cross-CTA races.

Result keys: "D1" … "S4" for the main shape, "D1@M=64,N=192,K=96" for the others.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from aliensilicon.interp import run_kernel
from aliensilicon.ir import Kernel
from aliensilicon.tasks import DISTRIBUTIONS, Task, check

PASS = "pass"


def run_tests(task: Task, kernel: Kernel, *, seed: int = 0, mufu_ulp_error: float = 0.0,
              mma_k_chunk: int | None = None) -> dict[str, str]:
    """Return {test name: "pass" or the failure reason}. Never raises on a bad kernel."""
    results = {}
    for i, shape in enumerate(task.shapes):
        suffix = "" if i == 0 else "@" + ",".join(f"{k}={v}" for k, v in shape.items())
        for dist in DISTRIBUTIONS:
            inputs = task.make_inputs(dist, seed, shape)
            results[dist + suffix] = _one(task, kernel, shape, inputs, seed,
                                          {"mufu_ulp_error": mufu_ulp_error, "mma_k_chunk": mma_k_chunk})
    return results


def passed(results: dict[str, str]) -> bool:
    return all(r == PASS for r in results.values())


def _one(task: Task, kernel: Kernel, shape: dict[str, int], inputs: dict[str, np.ndarray], seed: int,
         numerics: dict[str, Any]) -> str:
    run = dict(dims=task.dims(shape), seed=seed + 1, **numerics)
    try:
        a = run_kernel(kernel, inputs, **run)
        b = run_kernel(kernel, inputs, **run, garbage_seed=seed + 2, cta_order=seed + 3)
    except Exception as e:  # noqa: BLE001 - any failure of the kernel is a failed test, not a judge crash
        return f"error: {type(e).__name__}: {e}"[:300]
    for name, arr in inputs.items():
        if a[name].tobytes() != arr.tobytes():
            return f"modified input {name}"
    ref = task.expected(inputs)
    for name in task.outputs:
        if name not in a:
            return f"kernel has no output param {name}"
        if a[name].shape != ref[name].shape:
            return f"output {name} has shape {a[name].shape}, expected {ref[name].shape}"
        if a[name].tobytes() != b[name].tobytes():
            return f"output {name} depends on uninitialized memory or CTA order"
        if not check(a[name], ref[name]):
            return f"output {name} wrong (outside tolerance)"
    return PASS
