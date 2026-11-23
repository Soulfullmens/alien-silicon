# Launch plan: how this gets seen

Stars follow one thing: a result people want to share, which they can check in a minute. Launching too early with nothing striking spends the one shot you get on each channel. So there are two stages.

## Stage 1 (now): "the judge nobody can fool"

The judge is finished and stands on its own, and it addresses a known pain: AI kernel generators that cheat their benchmarks.

- **Artifact:** the README, `aliensilicon judge`, the "Break the judge" challenge, and `docs/red_team.md`, rewritten as a blog post: *"13 ways an AI can cheat a GPU kernel judge, and how we stopped each one."* The sandbox escape hidden in a type annotation, and kernels that only pass because simulators run threads in lockstep, are the attention-grabbers.
- **Where:**
  - GPU MODE Discord (the kernel community)
  - r/MachineLearning, flaired [P]
  - one thread on X tagging kernel and benchmark people
  - the TileLang community, since we map onto it
- **Ask:** "try to break it". A challenge pulls in contributors as well as visitors.
- **Don't** post to Hacker News yet. Save it for Stage 2.

## Stage 2 (Gate 1 / Phase 2): a headline result

One of these, with a figure:
- *"A model trained before FlashAttention-4 re-invented FA4's trick from raw chip specs"* (or, just as shareable, *"…it couldn't"*);
- *"AI kernels lose X% of the optimum on chips they've never seen"*;
- *"Our simulator ranks real T4 kernels with ρ = 0.8x"*.

Launch: arXiv paper, plus a Show HN with a browser demo of the alien chip explorer, plus a thread with the figure. Post on a Tuesday–Thursday morning, US time.

## Every time

- The README's first screen says what it is, why it matters, and a one-command demo. ✔
- Badges, a license, and CI that passes. ✔
- Topics on the repo: `gpu`, `cuda`, `kernels`, `compilers`, `llm-agents`, `benchmark`, `formal-verification`, `hardware-design`, `tilelang`.
- Issues labelled `good first issue` and `judge-break`, so visitors have something to do.
- Answer every issue within a day in the first two weeks.

## Before going public with results

- `main` must hold the code, not just the plan. Today the code lives on a branch.
- `docs/master_plan.md` contains private material: the business model, the M.Tech guide, budgets. Decide whether it stays public.
