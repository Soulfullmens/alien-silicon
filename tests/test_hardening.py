"""Round-2 hardening: sandbox, bounds proofs, multi-shape tests, chip numerics, regions, mutation score."""
import os

import numpy as np
import pytest

from aliensilicon.interp import InterpError, run_kernel
from aliensilicon.tasks import TASKS
from judge import evaluate, judge_hash
from judge import sandbox as sb
from judge.bounds import prove_bounds
from judge.correctness import passed, run_tests
from judge.mutate import score
from judge.static_check import Rejected, build_submission


def baseline(tid):
    t = TASKS[tid]
    return build_submission(t.baseline_source())[t.baseline]


def variant(tid, old, new):
    t = TASKS[tid]
    src = t.baseline_source()
    assert old in src
    return build_submission(src.replace(old, new))[t.baseline]


# ------------------------------------------------------------ sandbox

needs_isolation = pytest.mark.skipif(not sb.isolation()["network_namespace"], reason="no namespaces here")


@pytest.mark.parametrize("tid", sorted(TASKS))
def test_sandbox_build_matches_in_process(tid):
    t = TASKS[tid]
    assert sb.build_in_sandbox(t.baseline_source())[t.baseline] == baseline(tid)


def test_sandbox_passes_rejections_through():
    with pytest.raises(Rejected, match="top level may only hold"):
        sb.build_in_sandbox("import os\n")


@needs_isolation
@pytest.mark.parametrize("code, sign", [
    ("f = open('/tmp/as_probe', 'w'); f.write('x'); f.close()", "File too large"),  # 0 bytes ever land
    ("import socket; socket.create_connection(('1.1.1.1', 80), timeout=2)", "unreachable"),
    ("import os; print(os.system('true'))", None),
    ("x = bytearray(4 << 30)", "MemoryError"),
])
def test_sandbox_contains_escapes(code, sign):
    r = sb.run_isolated(code)
    if os.path.exists("/tmp/as_probe"):
        assert os.path.getsize("/tmp/as_probe") == 0
        os.remove("/tmp/as_probe")
    if sign:
        assert sign in r.stderr
    else:
        assert r.stdout.strip() != "0"  # fork is refused, so the shell never runs


@needs_isolation
def test_sandbox_cpu_limit(monkeypatch):
    monkeypatch.setattr(sb, "CPU_SECONDS", 1)
    assert sb.run_isolated("while True:\n    pass", timeout=20).returncode != 0


# ------------------------------------------------------------ bounds proofs

@pytest.mark.parametrize("tid", sorted(TASKS))
def test_baselines_proven_in_bounds_for_all_shapes(tid):
    assert prove_bounds(baseline(tid), TASKS[tid])["status"] == "proven"


@pytest.mark.parametrize("old, new", [
    ('for k in serial("k_tiles"):', "for k in serial(4):"),  # only right for K = 128
    ("c1 = iadd(c0, 64)", "c1 = iadd(c0, 65)"),  # off by one
])
def test_bounds_counterexample_for_shape_specific_kernels(old, new):
    r = prove_bounds(variant("w1", old, new), TASKS["w1"])
    assert r["status"] == "failed" and "for shape" in r["detail"]


def test_data_dependent_index_is_unproven_not_failed():
    k = next(iter(build_submission('''from aliensilicon.ir import GMEM, RMEM, alloc, f16, i32, kernel, load, store
@kernel(grid=(1,), threads=32)
def k(X: GMEM[f16, ("rows", "cols")], I: GMEM[i32, (1,)], Y: GMEM[f16, ("rows", "cols")]):
    i = alloc(RMEM, (), i32)
    load(I[0], i)
    x = alloc(RMEM, ("cols",), f16)
    load(X[i, :], x)
    store(x, Y[i, :])
''').values()))
    assert prove_bounds(k, TASKS["w2"])["status"] == "unproven"


# ------------------------------------------------------------ multi-shape (S5) and output shapes

def test_hard_coded_shape_fails_other_shapes():
    k = variant("w2", 'x16 = alloc(RMEM, ("cols",), f16)\n    load(X[r, :], x16)',
                'x16 = alloc(RMEM, (256,), f16)\n    load(X[r, 0:256], x16)')
    res = run_tests(TASKS["w2"], k)
    assert res["D1"] == "pass" and not passed(res)


def test_wrong_output_shape_detected():
    k = next(iter(build_submission('''from aliensilicon.ir import GMEM, RMEM, alloc, f16, kernel, load
from aliensilicon.ir import program_id, store
@kernel(grid=("rows",), threads=32)
def k(X: GMEM[f16, ("rows", "cols")], Y: GMEM[f16, ("rows", 1)]):
    r = program_id(0)
    x = alloc(RMEM, (1,), f16)
    load(X[r, 0:1], x)
    store(x, Y[r, :])
''').values()))
    assert "shape" in run_tests(TASKS["w2"], k)["D1"]


# ------------------------------------------------------------ chip numerics

def test_mma_k_chunk_models_tensor_core_rounding():
    """One rounding per mma flatters f16 accumulation; per-k-chunk rounding (like real MMA) shows the error."""
    k = next(iter(build_submission('''from aliensilicon.ir import GMEM, RMEM, alloc, f16, kernel, load, mma, store
@kernel(grid=(1,), threads=32)
def k(A: GMEM[f16, (16, 512)], B: GMEM[f16, (512, 16)], C: GMEM[f16, (16, 16)]):
    a = alloc(RMEM, (16, 512), f16)
    b = alloc(RMEM, (512, 16), f16)
    load(A[:, :], a)
    load(B[:, :], b)
    acc = alloc(RMEM, (16, 16), f16, init=0.0)
    mma(a, b, acc)
    store(acc, C[:, :])
''').values()))
    rng = np.random.default_rng(0)
    x = {"A": rng.random((16, 512)).astype(np.float16), "B": rng.random((512, 16)).astype(np.float16)}
    ref = x["A"].astype(np.float64) @ x["B"].astype(np.float64)
    err = {c: np.abs(run_kernel(k, x, mma_k_chunk=c)["C"] - ref).mean() for c in (None, 8)}
    assert err[8] > 3 * err[None]
    # The f32-accumulating GEMM baseline stays correct with realistic rounding.
    assert passed(run_tests(TASKS["w1"], baseline("w1"), mma_k_chunk=8))


# ------------------------------------------------------------ region-level races

RACE = '''from aliensilicon.ir import GMEM, RMEM, SMEM, alloc, barrier, f32, kernel, load, store
@kernel(grid=(1,), threads=32)
def k(X: GMEM[f32, (8,)], Y: GMEM[f32, (4,)]):
    s = alloc(SMEM, (8,), f32)
    load(X[0:4], s[0:4])
    r = alloc(RMEM, (4,), f32)
    load(s[{read}], r)
    store(r, Y[:])
'''


def test_disjoint_regions_do_not_conflict():
    x = {"X": np.arange(8, dtype=np.float32)}
    ok = next(iter(build_submission(RACE.format(read="4:8")).values()))
    run_kernel(ok, x)  # reads the half nobody wrote: no barrier needed
    bad = next(iter(build_submission(RACE.format(read="2:6")).values()))
    with pytest.raises(InterpError, match="read after write"):
        run_kernel(bad, x)


# ------------------------------------------------------------ pipeline and mutation score

def test_evaluate_record():
    r = evaluate(TASKS["w2"].baseline_source(), TASKS["w2"])
    assert r["correct"] and r["stage"] == "done" and r["bounds"]["status"] == "proven"
    assert len(r["tests"]) == 8 * len(TASKS["w2"].shapes)
    assert r["provenance"]["judge_sha256"] == judge_hash() and len(r["ir_sha256"]) == 64
    bad = evaluate(TASKS["w2"].baseline_source().replace("    x = sub(x, m)\n", ""), TASKS["w2"])
    assert not bad["correct"] and bad["stage"] == "tests"
    assert evaluate("import os\n", TASKS["w2"])["stage"] == "static"


def test_full_judge_kills_every_softmax_mutant():
    r = score(TASKS["w2"])
    assert r["mutants"] >= 15 and r["survivors"] == [], r["survivors"]
