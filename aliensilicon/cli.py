"""`aliensilicon` command line (I-15.5). Week 3+ adds `run --chip`, `profile`, `chips`, `optimum`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aliensilicon", description="AI kernels for chips nobody has programmed yet.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tasks", help="list workloads")
    j = sub.add_parser("judge", help="judge a kernel: sandbox build, IR check, Z3 bounds proof, hidden tests")
    j.add_argument("kernel", type=Path, help="DSL source file with one @kernel")
    j.add_argument("--task", required=True, help="task id, e.g. w2 (see `aliensilicon tasks`)")
    j.add_argument("--mufu-ulp", type=float, default=0.0, help="inject special-function error (ulps)")
    j.add_argument("--mma-k-chunk", type=int, default=None, help="round the MMA accumulator every k elements")
    j.add_argument("--json", action="store_true", help="print the full result record")
    m = sub.add_parser("mutate", help="judge mutation score: how many broken baselines does the judge catch?")
    m.add_argument("--task", default=None)
    a = p.parse_args(argv)

    from aliensilicon.tasks import TASKS

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
    r = evaluate(a.kernel.read_text(), TASKS[a.task], mufu_ulp_error=a.mufu_ulp, mma_k_chunk=a.mma_k_chunk)
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
        print(f"  judge {r['provenance']['judge_sha256'][:12]}")
    return 0 if r["correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
