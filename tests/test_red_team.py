"""Every attack from docs/red_team.md stays fixed."""
import ast
import json

import numpy as np
import pytest

from aliensilicon import ir
from aliensilicon.interp import InterpError, run_kernel
from aliensilicon.tasks import TASKS
from judge.correctness import passed, run_tests
from judge.ir_check import load_kernel
from judge.static_check import Rejected, build_submission, check_source

IMPORTS = ("from aliensilicon.ir import GMEM, RMEM, alloc, cmp, convert, f16, f32, iadd, if_, kernel, "
           "load, mufu, mul, program_id, reduce, serial, store, sub\n")


def build(body):
    return next(iter(build_submission(IMPORTS + body).values()))


def baseline(tid):
    t = TASKS[tid]
    return build_submission(t.baseline_source())[t.baseline]


# A1: code in a string annotation passed the allow-list and could be eval'd.
ESCAPE = '''
@kernel(grid=(1,), threads=32)
def k(X: "[c for c in ().__class__.__base__.__subclasses__()][0]"):
    program_id(0)
'''


def test_a1_string_annotation_rejected(tmp_path):
    with pytest.raises(Rejected, match="annotated as GMEM"):
        check_source(IMPORTS + ESCAPE)


def test_a1_dsl_never_evals_annotations():
    env = {n: getattr(ir, n) for n in ir.DSL_NAMES} | {"__builtins__": {}}
    tree = ast.parse(ESCAPE)
    exec(compile(tree, "<t>", "exec", dont_inherit=True), env)
    with pytest.raises(ir.DSLError, match="annotation"):
        env["k"].build()


@pytest.mark.parametrize("ann", ['GMEM[f16, "rows"]', "GMEM[f16, (rows,)]", 'GMEM[float, ("n",)]', "GMEM"])
def test_a1_only_literal_gmem_annotations(ann):
    with pytest.raises(Rejected):
        check_source(IMPORTS + f"@kernel(grid=(1,), threads=32)\ndef k(X: {ann}):\n    program_id(0)\n")


# A2 / A3: memory and time denial of service.
def test_a2_huge_alloc_fails_cleanly():
    k = build('''
@kernel(grid=(1,), threads=32)
def k(X: GMEM[f32, (8,)]):
    a = alloc(RMEM, (100000, 100000), f32)
''')
    with pytest.raises(InterpError, match="exceeds"):
        run_kernel(k, {"X": np.zeros(8, np.float32)})


def test_a3_endless_loops_hit_the_budget():
    k = build('''
@kernel(grid=(1,), threads=32)
def k(X: GMEM[f32, (8,)]):
    for i in serial(100000):
        for j in serial(100000):
            x = alloc(RMEM, (), f32)
''')
    with pytest.raises(InterpError, match="budget"):
        run_kernel(k, {"X": np.zeros(8, np.float32)}, work_budget=10**8)


# A4: a kernel that is right only if CTA 0 runs first (a race on real GPUs).
def test_a4_cta_order_dependence_detected():
    k = build('''
@kernel(grid=("rows",), threads=128)
def softmax_naive(X: GMEM[f16, ("rows", "cols")], Y: GMEM[f16, ("rows", "cols")]):
    r = program_id(0)
    with if_(cmp("eq", r, 0)):
        z = alloc(RMEM, ("rows", "cols"), f16, init=0.0)
        store(z, Y[:, :])
    x16 = alloc(RMEM, ("cols",), f16)
    load(X[r, :], x16)
    x = convert(x16, f32)
    e = mufu("exp2", mul(sub(x, reduce("max", x, axis=0)), 1.4426950408889634))
    store(convert(mul(e, mufu("rcp", reduce("sum", e, axis=0))), f16), Y[r, :])
''')
    t = TASKS["w2"]
    with pytest.raises(InterpError, match="race: CTA 1 writes GMEM"):  # right answer in grid order, still caught
        run_kernel(k, t.make_inputs("D1", 0))
    assert "race" in run_tests(t, k)["D1"]


# A5: a kernel that scribbles over its input.
def test_a5_input_mutation_detected():
    k = build('''
@kernel(grid=("rows",), threads=128)
def softmax_naive(X: GMEM[f16, ("rows", "cols")], Y: GMEM[f16, ("rows", "cols")]):
    r = program_id(0)
    x16 = alloc(RMEM, ("cols",), f16)
    load(X[r, :], x16)
    x = convert(x16, f32)
    e = mufu("exp2", mul(sub(x, reduce("max", x, axis=0)), 1.4426950408889634))
    y = convert(mul(e, mufu("rcp", reduce("sum", e, axis=0))), f16)
    store(y, X[r, :])
    store(y, Y[r, :])
''')
    assert run_tests(TASKS["w2"], k)["D1"] == "modified input X"


def test_reading_uninitialized_memory_detected():
    k = build('''
@kernel(grid=("rows",), threads=128)
def softmax_naive(X: GMEM[f16, ("rows", "cols")], Y: GMEM[f16, ("rows", "cols")]):
    r = program_id(0)
    y = alloc(RMEM, ("cols",), f16)
    store(y, Y[r, :])
''')
    assert "uninitialized" in run_tests(TASKS["w2"], k)["D1"]


@pytest.mark.parametrize("tid", sorted(TASKS))
def test_honest_baselines_pass_the_judge(tid):
    assert passed(run_tests(TASKS[tid], baseline(tid), mufu_ulp_error=2.0))


# A6: hand-crafted IR JSON that skips the builder.
def tamper(fn):
    d = json.loads(baseline("w2").to_json())
    fn(d)
    return json.dumps(d)


def _ops(d, kind):
    return [op for op in d["body"]["ops"] if op["kind"] == kind]


TAMPERS = {
    "forbidden mufu op": lambda d: _ops(d, "mufu")[0]["attrs"].update(op="fast_exp"),
    "dangling value ref": lambda d: d["body"]["ops"][1].update(outputs=[999]),
    "lie about dtype": lambda d: d["values"][_ops(d, "sub")[0]["outputs"][0]].update(dtype="f16"),
    "extra attr": lambda d: _ops(d, "load")[0]["attrs"].update(coalesced=True),
    "unknown op": lambda d: d["body"]["ops"].append({"kind": "fast_exp", "inputs": [], "outputs": [],
                                                     "attrs": {}, "unit": "none", "blocks": []}),
    "wrong unit": lambda d: _ops(d, "mufu")[0].update(unit="FMA"),
    "bad grid axis": lambda d: d["body"]["ops"][0]["attrs"].update(axis=5),
    "bad threads": lambda d: d.update(threads=-1),
    "huge literal": lambda d: _ops(d, "mul")[0]["inputs"].__setitem__(1, 10**30),
    "missing field": lambda d: d.pop("grid"),
}


@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_a6_tampered_ir_rejected(name):
    with pytest.raises(Rejected):
        load_kernel(tamper(TAMPERS[name]))


def test_a6_deep_or_huge_json_rejected():
    with pytest.raises(Rejected):
        load_kernel('{"a":' * 100000 + "1" + "}" * 100000)
    with pytest.raises(Rejected, match="larger"):
        load_kernel(" " * 3_000_000)


@pytest.mark.parametrize("tid", sorted(TASKS))
def test_a6_honest_ir_loads(tid):
    k = baseline(tid)
    assert load_kernel(k.to_json()) == k


def test_raw_python_errors_become_rejections():
    with pytest.raises(Rejected, match="build failed"):
        build("@kernel(grid=(1,), threads=32)\ndef k(X: GMEM[f32, (8,)]):\n    r = GMEM[1]\n")


# B1: barriers removed. Lanes run in lockstep here, so without tracking this looks correct.
def test_b1_missing_barrier_detected():
    t = TASKS["w1"]
    no_barriers = t.baseline_source().replace("        barrier()\n", "")
    k = build_submission(no_barriers)[t.baseline]
    assert "read after write with no barrier" in run_tests(t, k)["D1"]
    after_mma = "        mma(a, b, acc)\n        barrier()\n"
    one_barrier = t.baseline_source().replace(after_mma, "        mma(a, b, acc)\n", 1)
    k = build_submission(one_barrier)[t.baseline]
    assert "written after a read with no barrier" in run_tests(t, k)["D1"]


def test_b1_async_copy_needs_wait_and_barrier():
    body = '''
@kernel(grid=(1,), threads=32)
def k(X: GMEM[f32, (8,)], Y: GMEM[f32, (8,)]):
    s = alloc(SMEM, (8,), f32)
    async_copy(X[:], s, 0)
    {sync}
    r = alloc(RMEM, (8,), f32)
    load(s, r)
    store(r, Y[:])
'''
    src = "from aliensilicon.ir import GMEM, RMEM, SMEM, alloc, async_copy, barrier, f32, kernel, load, store, wait\n"
    x = {"X": np.arange(8, dtype=np.float32)}
    for sync, err in [("barrier()", "before wait"), ("wait(0)", "no barrier"), ("wait(0)\n    barrier()", None)]:
        k = next(iter(build_submission(src + body.format(sync=sync)).values()))
        if err:
            with pytest.raises(InterpError, match=err):
                run_kernel(k, x)
        else:
            assert run_kernel(k, x)["Y"].tolist() == list(range(8))


# B2: softmax without max-subtraction passes every D-test (inputs <= 3) but overflows on large inputs.
def test_b2_unstable_softmax_caught_by_s3():
    t = TASKS["w2"]
    k = build_submission(t.baseline_source().replace("    x = sub(x, m)\n", ""))[t.baseline]
    res = run_tests(t, k)
    assert all(res[d] == "pass" for d in ("D1", "D2", "D3", "D4")) and res["S3"] != "pass"


# A7 / A8 / A9: bad literals and axes fail at build time with a clean DSLError.
@pytest.mark.parametrize("expr", ["mul(x, 1e999)", "iadd(program_id(0), 99999999999999999999)", "program_id(5)",
                                  "alloc(RMEM, (0,), f32)", "alloc(RMEM, (-4,), f32)"])
def test_a7_a9_bad_literals_rejected_at_build(expr):
    with pytest.raises(ir.DSLError):
        build(f'''
@kernel(grid=(1,), threads=32)
def k(X: GMEM[f32, (8,)]):
    x = alloc(RMEM, (8,), f32)
    r = {expr}
''')
