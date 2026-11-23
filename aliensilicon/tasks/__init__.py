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

# §14.1 hidden tests. D: scaled uniform [0, 1). S: structured inputs that break shortcuts
# (S1 constant rows, S2 zeroed 16-wide blocks, S3 large values, S4 sorted with alternating sign).
DISTRIBUTIONS = ("D1", "D2", "D3", "D4", "S1", "S2", "S3", "S4")
_D_SCALE = {"D1": 1.0, "D2": 3.0, "D3": 0.01, "D4": -1.0}

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
    large: float  # S3 scale: as large as the task stays finite in f16 ("where safe")

    def make_inputs(self, dist: str, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng([seed, DISTRIBUTIONS.index(dist)])
        return {n: _sample(dist, s, rng, self.large).astype(np.float16) for n, s in self.inputs.items()}

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
        assert shape[-1] % 16 == 0, "S2 needs the last dim to be a multiple of 16"
        keep = rng.random((*shape[:-1], shape[-1] // 16)) < 0.5
        return u * np.repeat(keep, 16, axis=-1)
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
        lambda A, B: {"C": A @ B}, "gemm_naive", large=10.0),
    "w2": Task(
        "w2", f"Y = softmax(X) over each row. X, Y: ({ROWS},{COLS}) f16.",
        {"X": (ROWS, COLS)}, ("Y",), {},
        lambda X: {"Y": _softmax(X)}, "softmax_naive", large=100.0),
    "w3": Task(
        "w3", f"O = softmax(Q K^T / sqrt({D})) V, non-causal, one head. Q, K, V, O: ({SEQ},{D}) f16.",
        {"Q": (SEQ, D), "K": (SEQ, D), "V": (SEQ, D)}, ("O",), {"q_blocks": SEQ // 16},
        lambda Q, K, V: {"O": _softmax(Q @ K.T / np.sqrt(D)) @ V}, "attention_naive", large=100.0),
}
