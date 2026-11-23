# W3 naive attention forward, non-causal, head dim 64: 16 query rows per CTA, full score row
# in registers, plain softmax (no online rescale). Must pass judge/static_check.py as-is.
from aliensilicon.ir import (
    GMEM,
    RMEM,
    SMEM,
    alloc,
    barrier,
    convert,
    f16,
    f32,
    iadd,
    imul,
    kernel,
    load,
    mma,
    mufu,
    mul,
    program_id,
    reduce,
    store,
    sub,
)


@kernel(grid=("q_blocks",), threads=128)
def attention_naive(Q: GMEM[f16, ("seq", 64)], K: GMEM[f16, ("seq", 64)], V: GMEM[f16, ("seq", 64)],
                    O: GMEM[f16, ("seq", 64)]):
    r0 = imul(program_id(0), 16)
    r1 = iadd(r0, 16)
    q = alloc(SMEM, (16, 64), f16)
    k = alloc(SMEM, ("seq", 64), f16)
    v = alloc(SMEM, ("seq", 64), f16)
    load(Q[r0:r1, :], q)
    load(K[:, :], k)
    load(V[:, :], v)
    barrier()
    s = alloc(RMEM, (16, "seq"), f32, init=0.0)
    mma(q, k, s, transpose_b=True)
    m = reduce("max", s, axis=1, scope="cta", keepdims=True)
    p = mufu("exp2", mul(sub(s, m), 0.18033688011112042))  # log2(e) / sqrt(64)
    l = reduce("sum", p, axis=1, scope="cta", keepdims=True)
    p16 = convert(mul(p, mufu("rcp", l)), f16)
    o = alloc(RMEM, (16, 64), f32, init=0.0)
    mma(p16, v, o)
    store(convert(o, f16), O[r0:r1, :])
