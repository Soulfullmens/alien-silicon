"""Workloads (§13): spec, float64 reference, input generator and naive baseline per task.

Week 2 covers W1 GEMM, W2 row softmax and W3 attention (non-causal, head dim 64).
Every task has several shapes (§14.1 S5) and a kernel must pass all of them, so
hard-coding one shape fails.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Correctness tolerance, relative + absolute. Locked by the pre-registration: never change it.
TAU = 1e-3

# §14.1 hidden tests. D: scaled uniform [0, 1). S: structured inputs that break shortcuts
# (S1 constant rows, S2 zeroed 16-wide blocks, S3 large values, S4 sorted with alternating sign).
DISTRIBUTIONS = ("D1", "D2", "D3", "D4", "S1", "S2", "S3", "S4")
_D_SCALE = {"D1": 1.0, "D2": 3.0, "D3": 0.01, "D4": -1.0}

BASELINES = Path(__file__).parent / "baselines"


@dataclass(frozen=True)
class Task:
    id: str
    spec: str
    inputs: dict[str, tuple[str | int, ...]]  # name -> symbolic shape, all f16
    outputs: dict[str, tuple[str | int, ...]]
    shapes: tuple[dict[str, int], ...]  # every one must pass; shapes[0] is the main shape
    grid: Callable[[dict[str, Any]], dict[str, Any]]  # grid / loop symbols; works on ints and z3 Ints
    constraints: Callable[[dict[str, Any]], list[Any]]  # what every valid shape satisfies (for the bounds proof)
    reference: Callable[..., dict[str, np.ndarray]]  # float64 in, float64 out
    baseline: str
    large: float  # S3 scale: as large as the task stays finite in f16 ("where safe")

    def dims(self, shape: dict[str, int] | None = None) -> dict[str, int]:
        return self.grid(shape or self.shapes[0])

    def __post_init__(self) -> None:
        for s in self.shapes:
            assert all(s[d] > 0 for d in s) and all(self.constraints(s)), f"{self.id}: shape {s} breaks constraints"

    def make_inputs(self, dist: str, seed: int, shape: dict[str, int] | None = None) -> dict[str, np.ndarray]:
        shape = shape or self.shapes[0]
        rng = np.random.default_rng([seed, DISTRIBUTIONS.index(dist), *shape.values()])
        return {n: _sample(dist, tuple(shape.get(d, d) for d in s), rng, self.large).astype(np.float16)
                for n, s in self.inputs.items()}

    def expected(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return self.reference(**{n: a.astype(np.float64) for n, a in inputs.items()})

    def baseline_source(self) -> str:
        return (BASELINES / f"{self.baseline}.py").read_text()


def _sample(dist: str, shape: tuple[int, ...], rng: np.random.Generator, large: float) -> np.ndarray:
    u = rng.random(shape)
    if dist in _D_SCALE:
        return u * _D_SCALE[dist]
    if dist == "S1":
        return np.broadcast_to(rng.random((*shape[:-1], 1)), shape)
    if dist == "S2":
        keep = rng.random((*shape[:-1], -(-shape[-1] // 16))) < 0.5
        return u * np.repeat(keep, 16, axis=-1)[..., :shape[-1]]
    if dist == "S3":
        return u * large
    if dist == "S4":
        return np.sort(u, axis=-1) * np.where(np.arange(shape[-1]) % 2, -1.0, 1.0)
    raise ValueError(f"unknown test distribution {dist!r}")


def check(out: np.ndarray, ref: np.ndarray, tau: float = TAU) -> bool:
    """Pass iff all finite and |out - ref| <= tau * (|ref| + max|ref|) elementwise.

    The absolute part scales with the output: a fixed atol of tau would let an all-zero
    softmax (values ~ 1/cols) pass.
    """
    out = out.astype(np.float64)
    bound = tau * (np.abs(ref) + np.max(np.abs(ref)))
    return bool(np.all(np.isfinite(out)) and np.all(np.abs(out - ref) <= bound))


def div(a: Any, b: int) -> Any:
    """Floor division on ints and on z3 Ints (where `/` is integer division)."""
    return a // b if isinstance(a, int) else a / b


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


D = 64  # W3 head dim; 32 comes later

# shortcut: GEMM and attention shapes stay multiples of the baseline tiles (64 / 32 / 16), because
# the IR has no masked load yet. Add non-multiple shapes once a masking primitive exists.
TASKS = {
    "w1": Task(
        "w1", "C = A @ B. A: (M,K) f16, B: (K,N) f16, C: (M,N) f16, f32 accumulation. "
              "M, N multiples of 64 and K a multiple of 32. Grid symbols: m_tiles = M/64, n_tiles = N/64, "
              "k_tiles = K/32.",
        {"A": ("M", "K"), "B": ("K", "N")}, {"C": ("M", "N")},
        ({"M": 128, "N": 128, "K": 128}, {"M": 64, "N": 192, "K": 96}, {"M": 192, "N": 64, "K": 32}),
        lambda s: {"m_tiles": div(s["M"], 64), "n_tiles": div(s["N"], 64), "k_tiles": div(s["K"], 32)},
        lambda s: [s["M"] % 64 == 0, s["N"] % 64 == 0, s["K"] % 32 == 0],
        lambda A, B: {"C": A @ B}, "gemm_naive", large=10.0),
    "w2": Task(
        "w2", "Y = softmax(X) over each row. X, Y: (rows, cols) f16, any cols.",
        {"X": ("rows", "cols")}, {"Y": ("rows", "cols")},
        ({"rows": 8, "cols": 256}, {"rows": 3, "cols": 200}, {"rows": 2, "cols": 1000}),
        lambda s: {},
        lambda s: [],
        lambda X: {"Y": _softmax(X)}, "softmax_naive", large=100.0),
    "w3": Task(
        "w3", f"O = softmax(Q K^T / sqrt({D})) V, non-causal, one head. Q, K, V, O: (seq, {D}) f16, "
              "seq a multiple of 16. Grid symbol: q_blocks = seq/16.",
        {"Q": ("seq", D), "K": ("seq", D), "V": ("seq", D)}, {"O": ("seq", D)},
        ({"seq": 256}, {"seq": 128}, {"seq": 48}),
        lambda s: {"q_blocks": div(s["seq"], 16)},
        lambda s: [s["seq"] % 16 == 0],
        lambda Q, K, V: {"O": _softmax(Q @ K.T / np.sqrt(D)) @ V}, "attention_naive", large=100.0),
}
