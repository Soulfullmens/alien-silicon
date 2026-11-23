"""Time-machine check (Gate 1, §I-9): do FlashAttention-4's tricks help where FA4 says they should?

FA4 (Zadouri et al., 2026) moves part of softmax's exponentials from the MUFU unit to FMA units and
skips most rescaling, because Blackwell doubled tensor-core throughput but not MUFU or shared memory.
This script prices an FA2-style and an FA4-style attention kernel on chip profiles built only from
public spec sheets, and reports where the simulator says the tricks pay off.

Gate 1 passes if FA4 is >= 10% faster on blackwell-like and gains <= 3% on the balanced control.

    python analysis/time_machine.py              # table + docs/time_machine.md (+ .png with matplotlib)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aliensilicon.chips import load_chip  # noqa: E402
from aliensilicon.cost import estimate  # noqa: E402
from aliensilicon.tasks import TASKS  # noqa: E402
from judge.static_check import build_submission  # noqa: E402

CHIPS = ("t4", "hopper-like", "blackwell-like", "balanced")
KERNELS = ("attention_fa2", "attention_fa4")
SEQ = 4096
EXAMPLES = ROOT / "aliensilicon" / "tasks" / "examples"


def run(seq: int = SEQ) -> dict[str, dict[str, object]]:
    task = TASKS["w3"]
    shape = {"seq": seq}
    inputs, dims = task.make_inputs("D1", 0, shape), task.dims(shape)
    kernels = {n: build_submission((EXAMPLES / f"{n}.py").read_text())[n] for n in KERNELS}
    return {c: {n: estimate(k, load_chip(c), inputs, dims) for n, k in kernels.items()} for c in CHIPS}


def gate1(results: dict[str, dict[str, object]]) -> tuple[bool, dict[str, float]]:
    speedup = {c: r["attention_fa2"].cycles / r["attention_fa4"].cycles for c, r in results.items()}
    return speedup["blackwell-like"] >= 1.10 and speedup["balanced"] <= 1.03, speedup


def report(results: dict[str, dict[str, object]]) -> str:
    ok, speedup = gate1(results)
    lines = [
        f"# Time-machine check: attention, seq = {SEQ}, head dim 64",
        "",
        "Simulated with cost model v1 on chip profiles built from public specs (no calibration yet).",
        "",
        "| Chip | FA2-style cycles (bottleneck) | FA4-style cycles (bottleneck) | FA4 speedup |",
        "|---|---|---|---|",
    ]
    for c, r in results.items():
        a, b = r["attention_fa2"], r["attention_fa4"]
        lines.append(f"| {c} | {a.cycles:,.0f} ({a.bottleneck}) | {b.cycles:,.0f} ({b.bottleneck}) | "
                     f"**{speedup[c]:.2f}x** |")
    verdict = "PASS" if ok else "FAIL"
    lines += ["", f"**Gate 1: {verdict}** (needs >= 1.10x on blackwell-like and <= 1.03x on balanced)."]
    return "\n".join(lines) + "\n"


def chart(results: dict[str, dict[str, object]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, speedup = gate1(results)
    names = list(speedup)
    vals = [speedup[c] for c in names]
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    bars = ax.bar(names, vals, color=["#9aa5b1" if v < 1.03 else "#2f6fde" for v in vals])
    ax.axhline(1.0, color="#333", linewidth=0.8)
    for bar, v in zip(bars, vals, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}x", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("FA4-style speedup over FA2-style")
    ax.set_title("Simulator predicts where FlashAttention-4's tricks pay off", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(vals) * 1.2)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    res = run()
    text = report(res)
    print(text)
    (ROOT / "docs" / "time_machine.md").write_text(text)
    try:
        chart(res, ROOT / "docs" / "time_machine.png")
    except ImportError:
        print("(install matplotlib for the chart: pip install -e '.[analysis]')")
