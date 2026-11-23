import time

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aliensilicon.interp import InterpError, run_kernel, run_program
from aliensilicon.ir import (
    GMEM,
    RMEM,
    add,
    alloc,
    bitcast,
    buffer,
    convert,
    f16,
    f32,
    fma,
    gather_load,
    i8,
    i32,
    iadd,
    imul,
    kernel,
    load,
    mufu,
    mul,
    program,
    program_id,
    shl,
    shr,
    store,
)
from aliensilicon.tasks import DISTRIBUTIONS, TASKS, check
from judge.static_check import build_submission


def baseline(task):
    return build_submission(task.baseline_source())[task.baseline]


# ------------------------------------------------------------ Week 2 done-check

@pytest.mark.parametrize("dist", sorted(DISTRIBUTIONS))
@pytest.mark.parametrize("tid", sorted(TASKS))
def test_baselines_correct_at_tau(tid, dist):
    task = TASKS[tid]
    inputs = task.make_inputs(dist, seed=0)
    out = run_kernel(baseline(task), inputs, task.dims, seed=1)
    ref = task.expected(inputs)
    for name in task.outputs:
        assert check(out[name], ref[name]), f"{tid}/{dist}: max err {np.max(np.abs(out[name] - ref[name]))}"


@pytest.mark.parametrize("tid", sorted(TASKS))
def test_deterministic_bit_identical(tid):
    task = TASKS[tid]
    inputs = task.make_inputs("D1", seed=3)
    k = baseline(task)
    a = run_kernel(k, inputs, task.dims, seed=7, mufu_ulp_error=2.0)
    b = run_kernel(k, inputs, task.dims, seed=7, mufu_ulp_error=2.0)
    for name in task.outputs:
        assert a[name].tobytes() == b[name].tobytes()


def test_attention_256_under_2s():
    task = TASKS["w3"]
    inputs = task.make_inputs("D1", seed=0)
    k = baseline(task)
    t = time.perf_counter()
    run_kernel(k, inputs, task.dims)
    assert time.perf_counter() - t < 2.0


def test_reference_catches_a_wrong_kernel():
    task = TASKS["w2"]
    inputs = task.make_inputs("D1", seed=0)
    out = run_kernel(baseline(task), inputs, task.dims)["Y"]
    ref = task.expected(inputs)["Y"]
    assert check(out, ref)
    assert not check(out * np.float16(1.01), ref)
    assert not check(np.zeros_like(out), ref)


# ------------------------------------------------------------ §9 unit tests

def one_op(fn, in_dtype, out_dtype, n=8):
    """Kernel computing Y = fn(X) elementwise on one n-vector."""
    @kernel(grid=(1,), threads=32)
    def k(X: GMEM[in_dtype, (n,)], Y: GMEM[out_dtype, (n,)]):
        x = alloc(RMEM, (n,), in_dtype)
        load(X[:], x)
        store(fn(x), Y[:])
    return lambda X: run_kernel(k.build(), {"X": X})["Y"]


def test_f16_rounds_after_each_op():
    # 2048 + 1 is not representable in f16 (spacing 2): ties-to-even gives 2048.
    f = one_op(lambda x: add(add(x, 1.0), 1.0), f16, f16, n=1)
    assert f(np.array([2048], np.float16))[0] == 2048
    assert one_op(lambda x: add(x, 2.0), f16, f16, n=1)(np.array([2048], np.float16))[0] == 2050


def test_fma_rounds_once():
    # x*x - 1 with x = 1 + 2^-12: the product's low bits survive only with a single rounding.
    x = np.float32(1 + 2**-12)
    f = one_op(lambda v: fma(v, v, -1.0), f32, f32, n=1)
    assert f(np.array([x], np.float32))[0] == np.float32(2**-11 + 2**-24)
    g = one_op(lambda v: add(mul(v, v), -1.0), f32, f32, n=1)
    assert g(np.array([x], np.float32))[0] == np.float32(2**-11)


@settings(max_examples=50, deadline=None)
@given(st.lists(st.floats(width=32, allow_nan=False), min_size=4, max_size=4))
def test_bitcast_round_trip(vals):
    x = np.array(vals, np.float32)
    f = one_op(lambda v: bitcast(bitcast(v, i32), f32), f32, f32, n=4)
    assert f(x).tobytes() == x.tobytes()
    bits = one_op(lambda v: bitcast(v, i32), f32, i32, n=4)(x)
    assert bits.tobytes() == x.view(np.int32).tobytes()


def test_integer_overflow_wraps():
    f = one_op(lambda v: iadd(v, 1), i32, i32, n=1)
    assert f(np.array([2**31 - 1], np.int32))[0] == -(2**31)
    g = one_op(lambda v: imul(v, 3), i8, i8, n=1)
    assert g(np.array([100], np.int8))[0] == np.int8(300 - 256)


def test_shifts_follow_ptx_clamp():
    x = np.array([1, -8, 1, -8], np.int32)
    assert one_op(lambda v: shl(v, 31), i32, i32, n=4)(x).tolist() == [-(2**31), 0, -(2**31), 0]
    assert one_op(lambda v: shl(v, 40), i32, i32, n=4)(x).tolist() == [0, 0, 0, 0]
    assert one_op(lambda v: shr(v, 40), i32, i32, n=4)(x).tolist() == [0, -1, 0, -1]


def test_float_to_int_truncates_and_saturates():
    x = np.array([1.9, -1.9, 1e10, np.nan], np.float32)
    assert one_op(lambda v: convert(v, i32), f32, i32, n=4)(x).tolist() == [1, -1, 2**31 - 1, 0]


def test_f16_overflow_goes_to_inf():
    assert np.isinf(one_op(lambda v: convert(v, f16), f32, f16, n=1)(np.array([1e6], np.float32))[0])


def test_mufu_error_is_seeded_and_bounded():
    x = np.linspace(-4, 4, 64).astype(np.float32)

    @kernel(grid=(1,), threads=32)
    def k(X: GMEM[f32, (64,)], Y: GMEM[f32, (64,)]):
        v = alloc(RMEM, (64,), f32)
        load(X[:], v)
        store(mufu("exp2", v), Y[:])

    exact = np.exp2(x.astype(np.float64)).astype(np.float32)
    noisy = run_kernel(k.build(), {"X": x}, seed=1, mufu_ulp_error=4.0)["Y"]
    ulps = np.abs(noisy.astype(np.float64) - exact) / np.spacing(exact)
    assert ulps.max() <= 4.5 and ulps.max() > 0.5
    assert run_kernel(k.build(), {"X": x}, seed=2, mufu_ulp_error=4.0)["Y"].tobytes() != noisy.tobytes()
    assert run_kernel(k.build(), {"X": x})["Y"].tobytes() == exact.tobytes()


def test_unwritten_output_is_garbage():
    @kernel(grid=(1,), threads=32)
    def lazy(X: GMEM[f32, (8,)], Y: GMEM[f32, (8,)]):
        x = alloc(RMEM, (4,), f32)
        load(X[0:4], x)
        store(x, Y[0:4])  # forgets Y[4:8]

    k = lazy.build()
    a = run_kernel(k, {"X": np.ones(8, np.float32)}, seed=1)["Y"]
    b = run_kernel(k, {"X": np.ones(8, np.float32)}, seed=2)["Y"]
    assert a[:4].tolist() == [1, 1, 1, 1]
    assert a[4:].tobytes() != b[4:].tobytes()


def test_out_of_bounds_load_fails():
    @kernel(grid=(3,), threads=32)
    def oob(X: GMEM[f32, (8,)], Y: GMEM[f32, (8,)]):
        x = alloc(RMEM, (4,), f32)
        s = imul(program_id(0), 4)
        load(X[s:iadd(s, 4)], x)  # third CTA reads X[8:12]
        store(x, Y[0:4])

    with pytest.raises(InterpError, match="out of bounds"):
        run_kernel(oob.build(), {"X": np.ones(8, np.float32)})


def test_gather_load_clamps_indices():
    @kernel(grid=(1,), threads=32)
    def g(X: GMEM[f32, (4, 2)], I: GMEM[i32, (3,)], Y: GMEM[f32, (3, 2)]):
        idx = alloc(RMEM, (3,), i32)
        load(I[:], idx)
        rows = alloc(RMEM, (3, 2), f32)
        gather_load(X, idx, rows)
        store(rows, Y[:, :])

    x = np.arange(8, dtype=np.float32).reshape(4, 2)
    y = run_kernel(g.build(), {"X": x, "I": np.array([2, -5, 99], np.int32)})["Y"]
    assert y.tolist() == [[4, 5], [0, 1], [6, 7]]


def test_input_validation():
    k = baseline(TASKS["w2"])
    with pytest.raises(InterpError, match="dtype"):
        run_kernel(k, {"X": np.zeros((8, 256), np.float32)})
    with pytest.raises(InterpError, match="unknown input"):
        run_kernel(k, {"Z": np.zeros((8, 256), np.float16)})


def test_program_runs_launches_in_order():
    @kernel(grid=("n",), threads=32)
    def double(X: GMEM[f32, ("n",)], Y: GMEM[f32, ("n",)]):
        i = program_id(0)
        x = alloc(RMEM, (), f32)
        load(X[i], x)
        store(mul(x, 2.0), Y[i])

    @program
    def quad(X: GMEM[f32, ("n",)], Y: GMEM[f32, ("n",)]):
        t = buffer(("n",), f32)
        double(X, t)
        double(t, Y)

    y = run_program(quad.build(), {"X": np.arange(5, dtype=np.float32)})["Y"]
    assert y.tolist() == [0, 4, 8, 12, 16]
