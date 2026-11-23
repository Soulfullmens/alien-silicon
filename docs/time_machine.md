# Time-machine check: attention, seq = 4096, head dim 64

Simulated with cost model v1 on chip profiles built from public specs (no calibration yet).

| Chip | FA2-style cycles (bottleneck) | FA4-style cycles (bottleneck) | FA4 speedup |
|---|---|---|---|
| t4 | 155,847 (mma) | 157,806 (mma) | **0.99x** |
| hopper-like | 50,762 (mufu) | 51,581 (mma) | **0.98x** |
| blackwell-like | 48,509 (mufu) | 42,910 (fma) | **1.13x** |
| balanced | 32,673 (fma) | 40,446 (fma) | **0.81x** |

**Gate 1: PASS** (needs >= 1.10x on blackwell-like and <= 1.03x on balanced).
