# W3, FlashAttention-4 style tricks on top of attention_fa2 (Zadouri et al. 2026, §3.1.3):
#  1. software exp2 on the FMA/INT units for 16 of every 64 columns, built from generic ops only:
#     2^x = 2^floor(x) * 2^frac(x); 2^frac by a cubic polynomial, 2^floor by adding into the exponent bits;
#  2. conditional rescaling: only rescale O and l when the running max grows by more than 2^8.
from aliensilicon.ir import (
    GMEM,
    RMEM,
    SMEM,
    add,
    alloc,
    barrier,
    bitcast,
    cmp,
    convert,
    f16,
    f32,
    fma,
    i32,
    iadd,
    if_,
    imul,
    isub,
    kernel,
    load,
    max,
    mma,
    mufu,
    mul,
    program_id,
    reduce,
    select,
    serial,
    shl,
    store,
    sub,
)


@kernel(grid=("q_tiles",), threads=512)
def attention_fa4(Q: GMEM[f16, ("seq", 64)], K: GMEM[f16, ("seq", 64)], V: GMEM[f16, ("seq", 64)],
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
    p = alloc(RMEM, (128, 64), f32)
    xa = alloc(RMEM, (128, 48), f32)
    xb = alloc(RMEM, (128, 16), f32)
    for j in serial("kv_tiles"):
        c0 = imul(j, 64)
        c1 = iadd(c0, 64)
        barrier()
        load(K[c0:c1, :], k)
        load(V[c0:c1, :], v)
        barrier()
        mma(q, k, s, transpose_b=True, clear=True)
        m_new = max(m, reduce("max", s, axis=1, scope="warp", keepdims=True))
        grow = reduce("max", reduce("max", sub(m_new, m), axis=0, scope="cta"), axis=0, scope="thread")
        with if_(cmp("gt", grow, 44.361419555836500)):  # 8 / (log2(e) / sqrt(64)): max grew by > 2^8
            alpha = mufu("exp2", mul(sub(m, m_new), 0.18033688011112042))
            load(mul(o, alpha), o)
            load(mul(l, alpha), l)
            load(m_new, m)
        x = fma(s, 0.18033688011112042, mul(m, -0.18033688011112042))
        load(x[:, 0:48], xa)
        load(x[:, 48:64], xb)
        load(mufu("exp2", xa), p[:, 0:48])
        xc = max(xb, -126.0)
        xi = convert(xc, i32)
        xi = select(cmp("lt", xc, convert(xi, f32)), isub(xi, 1), xi)
        frac = sub(xc, convert(xi, f32))
        poly = fma(fma(fma(frac, 0.07705975192565022, 0.22764372212750406), frac, 0.6951233115822963), frac, 1.0)
        load(bitcast(iadd(bitcast(poly, i32), shl(xi, 23)), f32), p[:, 48:64])
        load(add(l, reduce("sum", p, axis=1, scope="warp", keepdims=True)), l)
        mma(convert(p, f16), v, o)
    store(convert(mul(o, mufu("rcp", l)), f16), O[r0:r1, :])
