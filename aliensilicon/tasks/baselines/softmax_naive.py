# W2 naive row softmax, exactly the §8.4 example. Must pass judge/static_check.py as-is.
from aliensilicon.ir import (
    GMEM,
    RMEM,
    alloc,
    convert,
    f16,
    f32,
    kernel,
    load,
    mufu,
    mul,
    program_id,
    reduce,
    store,
    sub,
)


@kernel(grid=("rows",), threads=128)
def softmax_naive(X: GMEM[f16, ("rows", "cols")], Y: GMEM[f16, ("rows", "cols")]):
    r = program_id(0)
    x16 = alloc(RMEM, ("cols",), f16)
    load(X[r, :], x16)
    x = convert(x16, f32)
    m = reduce("max", x, axis=0, scope="cta")
    x = sub(x, m)
    x = mul(x, 1.4426950408889634)  # log2(e)
    e = mufu("exp2", x)
    s = reduce("sum", e, axis=0, scope="cta")
    y = mul(e, mufu("rcp", s))
    store(convert(y, f16), Y[r, :])
