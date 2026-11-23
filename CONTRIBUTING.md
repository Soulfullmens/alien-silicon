# Contributing

Thanks for helping. Three kinds of contribution matter most right now.

## 1. Break the judge (most wanted)

If you find a kernel that is wrong, or would race on a real GPU, but the judge still says `CORRECT`, that's a real bug. It's the most valuable thing you can send.

1. Check it: `aliensilicon judge your_kernel.py --task w2` prints `CORRECT`.
2. Open an issue with the **Judge break** template, and include the kernel plus one sentence on why it is wrong.
3. If you like, send a PR that adds it to `tests/test_red_team.py` as a failing test. We fix the judge, the test goes green, and you go in the README hall of fame.

Judge breaks that count include wrong results, reading uninitialized memory, depending on CTA order, missing barriers, writing to inputs, escaping the sandbox, and anything that only passes because of how the simulator works.

## 2. Kernels and workloads

- Faster correct kernels for `w1`–`w3`, written in the DSL (`aliensilicon/tasks/baselines/` shows the style).
- New workloads (§13 of `docs/master_plan.md`). Each needs a float64 reference, 3 shapes, shape constraints and a naive baseline.

## 3. Hardware knowledge

Week 3 adds the chip schema. We'd like measured numbers from real hardware (MMA shapes, MUFU throughput, shared-memory banks, special-function accuracy), and their sources.

## Rules for code

- `make test` and `make lint` must pass. New logic needs a test.
- The judge is deterministic. No LLM ever decides pass or fail.
- The tolerance τ = 1e-3 is locked. Don't change it.
- Record a decision that changes the spec in `docs/decision_log.md`.
- Primitives stay generic. No primitive may be named after a known trick (`fast_exp`, `online_softmax`, …).
