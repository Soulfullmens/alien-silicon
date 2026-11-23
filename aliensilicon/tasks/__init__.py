"""Workloads (§13): spec, float64 reference, input generator and naive baseline per task.

Week 2 covers W1 GEMM, W2 row softmax and W3 attention (non-causal, head dim 64).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Correctness tolerance, relative + absolute. Locked by the pre-registration: never change it.
TAU = 1e-3

# §14.1 D-tests: scale applied to uniform [0, 1).
DISTRIBUTIONS = {"D1": 1.0, "D2": 3.0, "D3": 0.01, "D4": -1.0}

BASELINES = Path(__file__).parent / "baselines"


@dataclass(frozen=True)
class Task:
    id: str
    spec: str
    inputs: dict[str, tuple[int, ...]]  # name -> shape, all f16
    outputs: tuple[str, ...]
    dims: dict[str, int]  # grid / loop symbols the baseline needs
    reference: Callable[..., dict[str, np.ndarray]]  # float64 in, float64 out
    baseline: str

    def make_inputs(self, dist: str, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng([seed, int(dist[1:])])
        scale = DISTRIBUTIONS[dist]
        return {n: (rng.random(s) * scale).astype(np.float16) for n, s in self.inputs.items()}

    def expected(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return self.reference(**{n: a.astype(np.float64) for n, a in inputs.items()})

    def baseline_source(self) -> str:
        return (BASELINES / f"{self.baseline}.py").read_text()


def check(out: np.ndarray, ref: np.ndarray, tau: float = TAU) -> bool:
    """Pass iff all finite and |out - ref| <= tau * (|ref| + max|ref|) elementwise.

    The absolute part scales with the output: a fixed atol of tau would let an all-zero
    softmax (values ~ 1/cols) pass.
    """
    out = out.astype(np.float64)
    bound = tau * (np.abs(ref) + np.max(np.abs(ref)))
    return bool(np.all(np.isfinite(out)) and np.all(np.abs(out - ref) <= bound))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


M = N = K = 128
ROWS, COLS = 8, 256
SEQ, D = 256, 64

TASKS = {
    "w1": Task(
        "w1", f"C = A @ B. A: ({M},{K}) f16, B: ({K},{N}) f16, C: ({M},{N}) f16. Accumulate in f32.",
        {"A": (M, K), "B": (K, N)}, ("C",), {"m_tiles": M // 64, "n_tiles": N // 64, "k_tiles": K // 32},
        lambda A, B: {"C": A @ B}, "gemm_naive"),
    "w2": Task(
        "w2", f"Y = softmax(X) over each row. X, Y: ({ROWS},{COLS}) f16.",
        {"X": (ROWS, COLS)}, ("Y",), {},
        lambda X: {"Y": _softmax(X)}, "softmax_naive"),
    "w3": Task(
        "w3", f"O = softmax(Q K^T / sqrt({D})) V, non-causal, one head. Q, K, V, O: ({SEQ},{D}) f16.",
        {"Q": (SEQ, D), "K": (SEQ, D), "V": (SEQ, D)}, ("O",), {"q_blocks": SEQ // 16},
        lambda Q, K, V: {"O": _softmax(Q @ K.T / np.sqrt(D)) @ V}, "attention_naive"),
}
