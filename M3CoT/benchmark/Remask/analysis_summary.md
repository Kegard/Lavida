# M3CoT Remask Experiment Summary

Setting:
- Dataset split: test
- Sampling: random, seed 42
- Samples: 400
- Generation: max_new_tokens=64, block_length=64, step_ratio=0.5
- Native total denoising steps: 32

## Baselines

| Run | Acc (%) | Notes |
|---|---:|---|
| native_random42_m64_spb32_cot | 40.50 | Official eval jsonl |
| proposal_only_random42_sr0p5_fixed_p32_rr0_r0 | 40.75 | proposal_step=32, remask_ratio=0, refine_steps=0 |

The proposal-only degeneration is close to native final, with a +0.25 point difference.

## Fixed Proposal Step, Fixed Total Budget

| Run | Proposal step | Remask ratio | Refine steps | Proposal acc (%) | Final acc (%) | Delta vs native | Improved | Worsened | Mean sec/sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| budget32_p16_rr0p50_r16 | 16 | 0.500 | 16 | 41.25 | 41.25 | +0.75 | 14 | 14 | 8.98 |
| budget32_p20_rr0p375_r12 | 20 | 0.375 | 12 | 40.00 | 40.75 | +0.25 | 8 | 5 | 9.17 |
| budget32_p24_rr0p25_r8 | 24 | 0.250 | 8 | 40.50 | 41.00 | +0.50 | 4 | 2 | 9.37 |
| budget32_p28_rr0p125_r4 | 28 | 0.125 | 4 | 41.25 | 40.50 | +0.00 | 2 | 5 | 9.47 |
| budget32_p32_rr0_r0 | 32 | 0.000 | 0 | 40.75 | 40.75 | +0.25 | 0 | 0 | 9.56 |

Observation:
- Best fixed-step result is p16, 41.25%.
- p24 is close at 41.00%.
- p28 hurts relative to its proposal state: improved=2, worsened=5.
- Differences are small; with 400 samples, sub-1 point gaps should be treated as weak evidence unless paired per-sample comparisons support them.

## Dynamic x0 Convergence, Native Total Budget

| Run | Threshold | Persistence | Proposal acc (%) | Final acc (%) | Delta vs native | Avg trigger | Trigger range | Avg remask tokens | Avg refine steps | Improved | Worsened | Mean sec/sample |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| budgetNative_x0conv_t0p90_s2 | 0.90 | 2 | 38.75 | 41.25 | +0.75 | 11.38 | 3-30 | 41.23 | 20.61 | 20 | 10 | 9.10 |
| budgetNative_x0conv_t0p90_s3 | 0.90 | 3 | 39.75 | 40.25 | -0.25 | 16.79 | 4-32 | 30.41 | 15.21 | 10 | 8 | 9.18 |
| budgetNative_x0conv_t0p95_s2 | 0.95 | 2 | 39.75 | 41.00 | +0.50 | 19.66 | 3-32 | 24.69 | 12.35 | 12 | 7 | 9.34 |

Observation:
- Best x0 convergence result is threshold=0.90, persistence=2, final acc=41.25%.
- It triggers early on average, around step 11.38, then uses more remaining budget for remasking/refinement.
- threshold=0.95 triggers later and is more conservative, but final accuracy is slightly lower.
- persistence=3 is worse in this run.

## Entropy Plateau Runs

| Run | Completed samples | Status |
|---|---:|---|
| budgetNative_entropy_d0p01_s2 | 1 | Failed: entropy_plateau did not trigger on a sample |
| budgetNative_entropy_d0p01_s3 | 30 | Failed: entropy_plateau did not trigger on a sample |
| budgetNative_entropy_d0p02_s2 | 1 | Failed: entropy_plateau did not trigger on a sample |

These runs are incomplete and should not be compared with the 400-sample runs.

Failure reason from logs:

```text
ValueError: Proposal policy entropy_plateau did not trigger within available native denoising steps (32).
```

Recommendation:
- Add a fallback for entropy_plateau: if no plateau is detected by the final native step, trigger proposal at the final native step.
- After fallback is added, rerun entropy experiments before comparing them with fixed-step and x0-convergence results.

## Takeaways

Current best completed runs:
- Fixed step: budget32_p16_rr0p50_r16, 41.25%.
- Dynamic: budgetNative_x0conv_t0p90_s2, 41.25%.

Compared with native baseline 40.50%, the best observed gain is +0.75 points. This is directionally positive but small. The next useful step is paired per-sample analysis against native final to see whether gains concentrate in specific domains/topics or are mostly noise.

## Topic Breakdown For Key Runs

| Topic | N | Native | Fixed p16 | Fixed p24 | x0conv 0.90/s2 | x0conv 0.95/s2 |
|---|---:|---:|---:|---:|---:|---:|
| commonsense/physical-commonsense | 14 | 78.57 | 78.57 | 71.43 | 78.57 | 71.43 |
| commonsense/social-commonsense | 43 | 65.12 | 62.79 | 62.79 | 62.79 | 62.79 |
| commonsense/temporal-commonsense | 23 | 73.91 | 73.91 | 73.91 | 73.91 | 78.26 |
| mathematics/algebra | 21 | 19.05 | 23.81 | 19.05 | 23.81 | 23.81 |
| mathematics/geometry | 12 | 16.67 | 16.67 | 16.67 | 8.33 | 8.33 |
| mathematics/theory | 4 | 75.00 | 75.00 | 75.00 | 75.00 | 75.00 |
| science/language-science | 44 | 50.00 | 56.82 | 56.82 | 56.82 | 56.82 |
| science/natural-science | 122 | 36.07 | 35.25 | 36.07 | 36.07 | 35.25 |
| science/social-science | 117 | 26.50 | 27.35 | 27.35 | 27.35 | 27.35 |

The largest consistent positive movement is on language-science. Social-commonsense drops for all key remask variants, and geometry drops for x0 convergence. This suggests the average gain is not uniform across task types.
