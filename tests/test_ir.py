import json
from pathlib import Path

import pytest

from aliensilicon.ir import (
    GMEM,
    RMEM,
    SMEM,
    DSLError,
    Kernel,
    Program,
    alloc,
    bitcast,
    buffer,
    cmp,
    f16,
    f32,
    i32,
    if_,
    kernel,
    load,
    mma,
    mufu,
    mul,
    program,
    program_id,
    reduce,
    select,
    serial,
    store,
)
from judge.static_check import build_submission

BASELINES = Path(__file__).parent.parent / "aliensilicon" / "tasks" / "baselines"


def build(name):
    return build_submission((BASELINES / f"{name}.py").read_text())[name]


def ops(block):
    for op in block.ops:
        yield op
        for b in op.blocks:
            yield from ops(b)


@pytest.mark.parametrize("name", ["copy_rows", "softmax_naive", "gemm_naive"])
def test_baselines_build_and_round_trip(name):
    k = build(name)
    s = k.to_json()
    k2 = Kernel.from_json(s)
    assert k2 == k
    assert k2.to_json() == s and k2.sha256() == k.sha256()
    assert len(k.sha256()) == 64


def test_softmax_ir_shape():
    k = build("softmax_naive")
    kinds = [op.kind for op in ops(k.body)]
    assert kinds == ["program_id", "alloc", "load", "convert", "reduce", "sub", "mul", "mufu",
                     "reduce", "mufu", "mul", "convert", "store"]
    units = {op.kind: op.unit for op in ops(k.body)}
    assert units["mufu"] == "MUFU" and units["load"] == "GMEM" and units["reduce"] == "REDUCE"
    reduces = [op for op in ops(k.body) if op.kind == "reduce"]
    assert k.values[reduces[0].outputs[0]].shape == ()  # row max is a scalar
    assert json.loads(k.to_json())["grid"] == ["rows"]


def test_gemm_loop_is_one_op_with_body():
    k = build("gemm_naive")
    loops = [op for op in k.body.ops if op.kind == "for"]
    assert len(loops) == 1 and loops[0].attrs["n"] == "k_tiles"
    assert [op.kind for op in loops[0].blocks[0].ops].count("mma") == 1
    assert [op for op in k.body.ops if op.kind == "alloc"][0].attrs["init"] == 0.0


def test_hash_changes_with_any_edit():
    k = build("softmax_naive")
    d = json.loads(k.to_json())
    d["threads"] = 256
    assert Kernel.from_json(json.dumps(d)).sha256() != k.sha256()


def test_from_json_rejects_wrong_version():
    d = json.loads(build("copy_rows").to_json())
    d["ir_version"] = 99
    with pytest.raises(ValueError):
        Kernel.from_json(json.dumps(d))


def test_type_errors_are_caught_at_build():
    @kernel(grid=(1,), threads=32)
    def bad_mix(X: GMEM[f16, (8,)]):
        a = alloc(RMEM, (8,), f16)
        load(X[:], a)
        b = alloc(RMEM, (8,), f32)
        mul(a, b)

    with pytest.raises(DSLError, match="mixed dtypes"):
        bad_mix.build()

    @kernel(grid=(1,), threads=32)
    def bad_bitcast(X: GMEM[f16, (8,)]):
        a = alloc(RMEM, (8,), f16)
        bitcast(a, i32)

    with pytest.raises(DSLError, match="equal widths"):
        bad_bitcast.build()

    @kernel(grid=(1,), threads=32)
    def bad_mma(X: GMEM[f16, (8,)]):
        mma(alloc(SMEM, (16, 8), f16), alloc(SMEM, (16, 8), f16), alloc(RMEM, (16, 8), f32))

    with pytest.raises(DSLError, match="mma shape"):
        bad_mma.build()

    with pytest.raises(DSLError, match="inside a @kernel"):
        program_id(0)


def test_branch_select_and_mufu():
    @kernel(grid=(1,), threads=32)
    def k(X: GMEM[f32, (32,)]):
        x = alloc(RMEM, (32,), f32)
        load(X[:], x)
        s = reduce("sum", x, axis=0, scope="warp")
        with if_(cmp("gt", s, 0.0)):
            x = select(cmp("lt", x, 0.0), 0.0, x)
            store(x, X[:])
        for _i in serial(4, unroll=True):
            x = mufu("exp2", x)

    built = k.build()
    branch = [op for op in built.body.ops if op.kind == "if"][0]
    sel = [op for op in branch.blocks[0].ops if op.kind == "select"][0]
    assert len(sel.inputs) == 3 and sel.inputs[1] == 0.0
    assert Kernel.from_json(built.to_json()) == built
    with pytest.raises(DSLError, match="mufu op"):
        @kernel(grid=(1,), threads=32)
        def k2(X: GMEM[f32, (32,)]):
            mufu("fast_exp", X)
        k2.build()


def test_program_records_launches():
    @kernel(grid=("rows",), threads=128)
    def scale(X: GMEM[f16, ("rows", "cols")], Y: GMEM[f16, ("rows", "cols")]):
        r = program_id(0)
        x = alloc(RMEM, ("cols",), f16)
        load(X[r, :], x)
        store(x, Y[r, :])

    @program
    def two_pass(X: GMEM[f16, ("rows", "cols")], Y: GMEM[f16, ("rows", "cols")]):
        t = buffer(("rows", "cols"), f16)
        scale(X, t)
        scale(t, Y)

    p = two_pass.build()
    assert [launch["args"] for launch in p.launches] == [["X", "tmp0"], ["tmp0", "Y"]]
    assert len(p.kernels) == 1
    p2 = Program.from_json(p.to_json())
    assert p2.to_json() == p.to_json()
