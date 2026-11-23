"""Judge step 3 (§14.1): correctness on hidden tests, D-tests for now.

Each test runs the kernel twice: once in grid order, and once with different
garbage memory and a shuffled CTA order. An honest kernel gives bit-identical
output both times. A difference means it reads memory it never wrote, or it
depends on CTA order (a race on real hardware, where CTAs run in any order).
"""
from __future__ import annotations

import numpy as np

from aliensilicon.interp import run_kernel
from aliensilicon.ir import Kernel
from aliensilicon.tasks import DISTRIBUTIONS, Task, check

PASS = "pass"


def run_tests(task: Task, kernel: Kernel, *, seed: int = 0, mufu_ulp_error: float = 0.0) -> dict[str, str]:
    """Return {test name: "pass" or the failure reason}. Never raises on a bad kernel."""
    results = {}
    for dist in DISTRIBUTIONS:
        inputs = task.make_inputs(dist, seed)
        results[dist] = _one(task, kernel, inputs, seed, mufu_ulp_error)
    return results


def passed(results: dict[str, str]) -> bool:
    return all(r == PASS for r in results.values())


def _one(task: Task, kernel: Kernel, inputs: dict[str, np.ndarray], seed: int, mufu_ulp_error: float) -> str:
    run = dict(dims=task.dims, seed=seed + 1, mufu_ulp_error=mufu_ulp_error)
    try:
        a = run_kernel(kernel, inputs, **run)
        b = run_kernel(kernel, inputs, **run, garbage_seed=seed + 2, cta_order=seed + 3)
    except Exception as e:  # noqa: BLE001 - any failure of the kernel is a failed test, not a judge crash
        return f"error: {type(e).__name__}: {e}"[:300]
    for name, arr in inputs.items():
        if a[name].tobytes() != arr.tobytes():
            return f"modified input {name}"
    for name in task.outputs:
        if a[name].tobytes() != b[name].tobytes():
            return f"output {name} depends on uninitialized memory or CTA order"
    ref = task.expected(inputs)
    for name in task.outputs:
        if not check(a[name], ref[name]):
            return f"output {name} wrong (outside tolerance)"
    return PASS
