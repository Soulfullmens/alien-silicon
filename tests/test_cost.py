"""Week 3: chip specs, cost model v1, physics bounds, and the time-machine check (Gate 1)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from aliensilicon.chips import BOUNDS, ChipSpec, list_chips, load_chip
from aliensilicon.cost import ResourceError, estimate, register_pressure, resources, smem_access_bytes
from aliensilicon.tasks import TASKS
from judge import evaluate
from judge.static_check import build_submission

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "aliensilicon" / "tasks" / "examples"


def kernel(src):
    return next(iter(build_submission(src).values()))


def example(name):
    return build_submission((EXAMPLES / f"{name}.py").read_text())[name]


def baseline(tid):
    t = TASKS[tid]
    return build_submission(t.baseline_source())[t.baseline]


# ------------------------------------------------------------ chip specs

@pytest.mark.parametrize("name", list_chips())
def test_profiles_load_and_round_trip(name):
    c = load_chip(name)
    assert ChipSpec.from_json(c.to_json()) == c


def test_chip_validation():
    d = json.loads(load_chip("hopper-like").to_json())
    bad = dict(d, smem_per_cta_max_bytes=d["smem_per_sm_bytes"] + 1)
    with pytest.raises(ValueError, match="smem_per_cta_max_bytes"):
        ChipSpec.from_dict(bad)
    with pytest.raises(ValueError, match="outside"):
        ChipSpec.from_dict(dict(d, num_sm=0))
    with pytest.raises(ValueError, match="unknown"):
        ChipSpec.from_dict(dict(d, turbo=True))
    with pytest.raises(ValueError, match="mufu ops and mufu throughput"):
        ChipSpec.from_dict(dict(d, mufu=dict(d["mufu"], ops=[])))
    with pytest.raises(ValueError, match="unknown chip"):
        load_chip("no-such-chip")
    assert all(lo <= hi for lo, hi in BOUNDS.values())


# ------------------------------------------------------------ capacity effects

def test_bank_conflicts_any_bank_count_and_padding():
    col = (slice(None), 0)  # one column of a 32x32 f32 tile: every element in the same bank
    contiguous = smem_access_bytes((32, 32), 4, (0, slice(None)), 0, 32, 4, 32)
    assert contiguous == 128  # one conflict-free wavefront
    assert smem_access_bytes((32, 32), 4, col, 0, 32, 4, 32) == 32 * 128  # 32-way conflict
    assert smem_access_bytes((32, 32), 4, col, 1, 32, 4, 32) == 128  # padding one element fixes it
    # 23 banks: even a contiguous 32-lane access needs two cycles, and power-of-two padding does not help.
    assert smem_access_bytes((32, 32), 4, (0, slice(None)), 0, 23, 4, 32) == 2 * 23 * 4


def test_mma_tile_shape_support():
    t, x = TASKS["w3"], TASKS["w3"].make_inputs("D1", 0)
    naive = estimate(baseline("w3"), load_chip("hopper-like"), x, t.dims())
    assert "mma_fallback_to_fma" in naive.flags  # 16-row tiles cannot use a 64-row tensor core
    fa2 = estimate(example("attention_fa2"), load_chip("hopper-like"), x, t.dims())
    assert "mma_fallback_to_fma" not in fa2.flags and fa2.work["mma_flops"] > 0


def test_resource_rejections():
    t = TASKS["w3"]
    dims = {"seq": 256, **t.dims()}
    with pytest.raises(ResourceError, match="exp2"):
        resources(baseline("w3"), load_chip("alien-0001"), dims)
    with pytest.raises(ResourceError, match="shared memory"):
        resources(baseline("w3"), load_chip("t4"), dims)
    async_k = kernel('''from aliensilicon.ir import GMEM, SMEM, alloc, async_copy, f32, kernel, wait
@kernel(grid=(1,), threads=32)
def k(X: GMEM[f32, (8,)]):
    s = alloc(SMEM, (8,), f32)
    async_copy(X[:], s, 0)
    wait(0)
''')
    with pytest.raises(ResourceError, match="async"):
        resources(async_k, load_chip("t4"), {})
    resources(async_k, load_chip("hopper-like"), {})


def test_register_pressure_and_spills():
    big = kernel('''from aliensilicon.ir import GMEM, RMEM, add, alloc, f32, kernel, load, store
@kernel(grid=(1,), threads=32)
def k(X: GMEM[f32, (32, 256)], Y: GMEM[f32, (32, 256)]):
    a = alloc(RMEM, (32, 256), f32)
    load(X[:, :], a)
    b = add(a, 1.0)
    c = add(b, a)
    store(c, Y[:, :])
''')
    assert register_pressure(big, {}) == 3 * 256  # a, b and c are all live at once: 256 regs each
    p = estimate(big, load_chip("t4"), {"X": np.zeros((32, 256), np.float32)})
    assert any(f.startswith("register_spill") for f in p.flags) and p.breakdown["spill"] > 0


# ------------------------------------------------------------ physics and determinism

@pytest.mark.parametrize("tid", ["w1", "w2"])
def test_physics_lower_bound_holds_for_baselines(tid):
    t = TASKS[tid]
    p = estimate(baseline(tid), load_chip("hopper-like"), t.make_inputs("D1", 0, t.perf_shape), t.dims(t.perf_shape),
                 min_flops=t.min_flops(t.perf_shape))
    assert p.physics == "ok" and 0 < p.frac_lb <= 1 and p.cycles >= p.lower_bound_cycles


def test_estimate_is_deterministic():
    t, x = TASKS["w1"], TASKS["w1"].make_inputs("D1", 0)
    a = estimate(baseline("w1"), load_chip("t4"), x, t.dims())
    b = estimate(baseline("w1"), load_chip("t4"), x, t.dims())
    assert a == b


# ------------------------------------------------------------ judge with a chip

def test_evaluate_with_chip_prices_and_rejects():
    t = TASKS["w3"]
    ok = evaluate((EXAMPLES / "attention_fa4.py").read_text(), t, load_chip("blackwell-like"))
    assert ok["correct"] and ok["perf"]["bottleneck"] and ok["perf"]["hotspots"] and ok["cheat_flags"] == []
    alien = evaluate(t.baseline_source(), t, load_chip("alien-0001"))
    assert alien["stage"] == "resources" and "exp2" in alien["detail"]


@pytest.mark.parametrize("name", ["attention_fa2", "attention_fa4"])
def test_example_kernels_are_correct(name):
    from judge.correctness import passed, run_tests

    assert passed(run_tests(TASKS["w3"], example(name), mufu_ulp_error=2.0, mma_k_chunk=16))


# ------------------------------------------------------------ Gate 1: the time-machine check

def test_time_machine_gate1():
    sys.path.insert(0, str(ROOT / "analysis"))
    import time_machine

    ok, speedup = time_machine.gate1(time_machine.run(seq=2048))
    assert ok, speedup
    assert speedup["hopper-like"] < 1.03  # on Hopper the tensor cores are as busy as MUFU: no gain
