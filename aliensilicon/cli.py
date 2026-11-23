"""`aliensilicon` command line (I-15.5). Later: `chips generate`, `optimum`, `bench`, `studio`, `lower`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aliensilicon", description="AI kernels for chips nobody has programmed yet.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tasks", help="list workloads")
    sub.add_parser("chips", help="list chip profiles")
    j = sub.add_parser("judge", help="judge a kernel: sandbox build, IR check, Z3 bounds proof, hidden tests")
    j.add_argument("kernel", type=Path, help="DSL source file with one @kernel")
    j.add_argument("--task", required=True, help="task id, e.g. w2 (see `aliensilicon tasks`)")
    j.add_argument("--chip", default=None, help="price the kernel on a chip profile (see `aliensilicon chips`)")
    j.add_argument("--mufu-ulp", type=float, default=0.0, help="inject special-function error (ulps)")
    j.add_argument("--mma-k-chunk", type=int, default=None, help="round the MMA accumulator every k elements")
    j.add_argument("--json", action="store_true", help="print the full result record")
    m = sub.add_parser("mutate", help="judge mutation score: how many broken baselines does the judge catch?")
    m.add_argument("--task", default=None)
    a = p.parse_args(argv)

    from aliensilicon.tasks import TASKS

    if a.cmd == "chips":
        from aliensilicon.chips import list_chips, load_chip

        for name in list_chips():
            c = load_chip(name)
            print(f"{name:15} [{c.family}] {c.notes}")
        return 0
    if a.cmd == "tasks":
        for t in TASKS.values():
            print(f"{t.id}  {t.spec}\n     baseline: aliensilicon/tasks/baselines/{t.baseline}.py\n")
        return 0
    if a.cmd == "mutate":
        from judge.mutate import score

        for t in [TASKS[a.task]] if a.task else TASKS.values():
            r = score(t)
            print(f"{t.id}: {r['killed']}/{r['mutants']} mutants caught ({r['score']:.0%})")
            for s in r["survivors"]:
                print(f"   survived: {s}")
        return 0

    from judge import evaluate

    if a.task not in TASKS:
        p.error(f"unknown task {a.task!r}; choose from {', '.join(TASKS)}")
    chip = None
    if a.chip:
        from aliensilicon.chips import load_chip

        try:
            chip = load_chip(a.chip)
        except ValueError as e:
            p.error(str(e))
    r = evaluate(a.kernel.read_text(), TASKS[a.task], chip, mufu_ulp_error=a.mufu_ulp, mma_k_chunk=a.mma_k_chunk)
    if a.json:
        print(json.dumps(r, indent=2))
    else:
        print(f"{'CORRECT' if r['correct'] else 'REJECTED'}  (stage: {r['stage']})")
        if "detail" in r:
            print(f"  {r['detail']}")
        if "bounds" in r:
            print(f"  bounds: {r['bounds']['status']}")
        if "tests" in r:
            bad = {k: v for k, v in r["tests"].items() if v != "pass"}
            print(f"  tests: {len(r['tests']) - len(bad)}/{len(r['tests'])} passed")
            for k, v in list(bad.items())[:5]:
                print(f"    {k}: {v}")
        if "perf" in r:
            pf = r["perf"]
            shape = ", ".join(f"{k}={v}" for k, v in TASKS[a.task].perf_shape.items())
            print(f"  on {pf['chip']} at {shape}: {pf['cycles']:,.0f} cycles ({pf['time_us']:.1f} us), "
                  f"bottleneck: {pf['bottleneck']}, {pf['frac_lb']:.0%} of the lower bound")
            busy = sorted(((v, k) for k, v in pf["breakdown"].items() if v > 0), reverse=True)[:5]
            print("  busiest: " + ", ".join(f"{k} {v:,.0f}" for v, k in busy))
            for h in pf["hotspots"]:
                print(f"  hotspot: op #{h['op']} {h['kind']} {h['detail']} = {h['share']:.0%} of {pf['bottleneck']}")
            for f in pf["flags"]:
                print(f"  note: {f}")
        print(f"  judge {r['provenance']['judge_sha256'][:12]}")
    return 0 if r["correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
