# W3, FlashAttention-2 style: 128 query rows per CTA, 64-wide KV blocks, online softmax.
# Every exponential runs on the MUFU (special-function) unit.
from aliensilicon.ir import (
    GMEM,
    RMEM,
    SMEM,
    add,
    alloc,
    barrier,
    convert,
    f16,
    f32,
    fma,
    iadd,
    imul,
    kernel,
    load,
    max,
    mma,
    mufu,
    mul,
    program_id,
    reduce,
    serial,
    store,
)


@kernel(grid=("q_tiles",), threads=512)
def attention_fa2(Q: GMEM[f16, ("seq", 64)], K: GMEM[f16, ("seq", 64)], V: GMEM[f16, ("seq", 64)],
                  O: GMEM[f16, ("seq", 64)]):
    r0 = imul(program_id(0), 128)
    r1 = iadd(r0, 128)
    q = alloc(SMEM, (128, 64), f16)
    k = alloc(SMEM, (64, 64), f16)
    v = alloc(SMEM, (64, 64), f16)
    load(Q[r0:r1, :], q)
    m = alloc(RMEM, (128, 1), f32, init=-1e30)
    l = alloc(RMEM, (128, 1), f32, init=0.0)
    o = alloc(RMEM, (128, 64), f32, init=0.0)
    s = alloc(RMEM, (128, 64), f32)
    for j in serial("kv_tiles"):
        c0 = imul(j, 64)
        c1 = iadd(c0, 64)
        barrier()
        load(K[c0:c1, :], k)
        load(V[c0:c1, :], v)
        barrier()
        mma(q, k, s, transpose_b=True, clear=True)
        m_new = max(m, reduce("max", s, axis=1, scope="warp", keepdims=True))
        nm = mul(m_new, -0.18033688011112042)  # -max * log2(e) / sqrt(64)
        p = mufu("exp2", fma(s, 0.18033688011112042, nm))
        alpha = mufu("exp2", fma(m, 0.18033688011112042, nm))
        load(add(mul(l, alpha), reduce("sum", p, axis=1, scope="warp", keepdims=True)), l)
        load(mul(o, alpha), o)
        mma(convert(p, f16), v, o)
        load(m_new, m)
    store(convert(mul(o, mufu("rcp", l)), f16), O[r0:r1, :])
