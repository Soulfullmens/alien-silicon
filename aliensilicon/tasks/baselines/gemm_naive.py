# W1 naive tiled GEMM (64x64 output tile, K step 32). Must pass judge/static_check.py as-is.
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
    program_id,
    serial,
    store,
)


@kernel(grid=("m_tiles", "n_tiles"), threads=128)
def gemm_naive(A: GMEM[f16, ("M", "K")], B: GMEM[f16, ("K", "N")], C: GMEM[f16, ("M", "N")]):
    r0 = imul(program_id(0), 64)
    r1 = iadd(r0, 64)
    c0 = imul(program_id(1), 64)
    c1 = iadd(c0, 64)
    acc = alloc(RMEM, (64, 64), f32, init=0.0)
    a = alloc(SMEM, (64, 32), f16)
    b = alloc(SMEM, (32, 64), f16)
    for k in serial("k_tiles"):
        k0 = imul(k, 32)
        k1 = iadd(k0, 32)
        load(A[r0:r1, k0:k1], a)
        load(B[k0:k1, c0:c1], b)
        barrier()
        mma(a, b, acc)
        barrier()
    store(convert(acc, f16), C[r0:r1, c0:c1])
