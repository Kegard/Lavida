# PostMaSK Results Summary

## Dynamic Results

| Experiment | Selection | Draft Acc | Final Acc | Improved | Worsened | Mean Sec |
|---|---|---:|---:|---:|---:|---:|
| `postmask_sr0p5_d32_p0_seed42_n400` | `baseline` | 0.4075 | 0.4075 | 0 | 0 | 9.085 |
| `postmask_sr0p5_d16_p16_conf_r4_seed42_n400` | `cached_confidence` | 0.3900 | 0.3925 | 6 | 5 | 8.916 |
| `postmask_sr0p5_d16_p16_histconf_r4_seed42_n400_new` | `history_confidence` | 0.3900 | 0.3925 | 1 | 0 | 13.324 |
| `postmask_sr0p5_d16_p16_topkmargin_r4_seed42_n400` | `topk_margin` | 0.3900 | 0.3900 | 0 | 0 | 14.438 |
| `postmask_sr0p5_d16_p16_kldiv_r4_seed42_n400` | `kl_divergence` | 0.3900 | 0.3900 | 0 | 0 | 14.077 |
| `postmask_sr0p5_d16_p16_histconf_r4_seed42_n400` | `history_confidence` | 0.3900 | 0.3900 | 0 | 0 | 13.092 |
| `postmask_sr0p5_d16_p16_histconf_r4_seed42_n400_new_log` | `history_confidence` | 0.3900 | 0.3900 | 0 | 0 | 13.295 |
| `postmask_sr0p5_d16_p16_meanafterfill_r4_seed42_n400` | `mean_after_fill` | 0.3900 | 0.3900 | 1 | 1 | 13.332 |
| `postmask_sr0p5_d16_p16_meanafterfill_r4_seed42_n400_new` | `mean_after_fill` | 0.3900 | 0.3900 | 1 | 1 | 12.745 |
| `postmask_sr0p5_d16_p16_meanafterfill_r4_seed42_n400_new_log` | `mean_after_fill` | 0.3900 | 0.3900 | 1 | 1 | 13.229 |
| `postmask_sr0p5_d16_p16_laststepconf_r4_seed42_n400` | `last_step_confidence` | 0.3900 | 0.3850 | 1 | 3 | 14.601 |
| `postmask_sr0p5_d16_p16_rand_r4_seed42_n400` | `random` | 0.3900 | 0.3825 | 2 | 5 | 8.782 |
| `postmask_sr0p5_d16_p16_conf_r2_seed42_n400` | `cached_confidence (r2)` | 0.3900 | 0.3800 | 0 | 4 | 8.862 |

## Fixed-Set Results

| Experiment | Selection | Fixed Set | Refill/Step | Draft Acc | Final Acc | Improved | Worsened | Mean Sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `postmask_sr0p5_d16_p16_topkmargin_fixed16_refill1_seed42_n400` | `topk_margin` | 16 | 1 | 0.3900 | 0.3975 | 5 | 2 | 14.510 |
| `postmask_sr0p5_d16_p16_meanafterfill_fixed16_refill1_seed42_n400` | `mean_after_fill` | 16 | 1 | 0.3900 | 0.3950 | 3 | 1 | 14.504 |
| `postmask_sr0p5_d16_p16_cachedconf_fixed16_refill1_seed42_n400` | `cached_confidence` | 16 | 1 | 0.3900 | 0.3925 | 5 | 4 | 14.476 |
| `postmask_sr0p5_d16_p16_histconf_fixed16_refill1_seed42_n400` | `history_confidence` | 16 | 1 | 0.3900 | 0.3925 | 1 | 0 | 14.501 |
| `postmask_sr0p5_d16_p16_kldiv_fixed16_refill1_seed42_n400` | `kl_divergence` | 16 | 1 | 0.3900 | 0.3925 | 2 | 1 | 14.397 |
| `postmask_sr0p5_d16_p16_random_fixed16_refill1_seed42_n400` | `random` | 16 | 1 | 0.3900 | 0.3875 | 1 | 2 | 14.685 |
| `postmask_sr0p5_d16_p16_laststepconf_fixed16_refill1_seed42_n400` | `last_step_confidence` | 16 | 1 | 0.3900 | 0.3850 | 3 | 5 | 14.369 |

## Key Takeaways

- Best overall PostMaSK result: `topk_margin + fixed_set(16) + refill1`, with `Final Acc = 0.3975`
- Second best: `mean_after_fill + fixed_set(16) + refill1`, with `Final Acc = 0.3950`
- `fixed_set` generally outperforms the corresponding `dynamic` version
- All current PostMaSK variants remain below the baseline `0.4075`
