# Week 1 paper-check kernel: row copy. Must pass judge/static_check.py as-is.
from aliensilicon.ir import GMEM, RMEM, alloc, f16, kernel, load, program_id, store


@kernel(grid=("rows",), threads=128)
def copy_rows(X: GMEM[f16, ("rows", "cols")], Y: GMEM[f16, ("rows", "cols")]):
    r = program_id(0)
    x = alloc(RMEM, ("cols",), f16)
    load(X[r, :], x)
    store(x, Y[r, :])
