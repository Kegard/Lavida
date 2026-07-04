# PostMaSK Error Source Summary

Definitions:

- `global_dominant_persistent`: draft is wrong, final is wrong, and no postmask state is ever judged correct.
- `local_degradation`: draft is correct, but final becomes wrong after postmask.
- `local_repair_of_draft_error`: draft is wrong, but final becomes correct after postmask.
- `mixed_repaired_then_lost`: draft is wrong, at least one postmask state becomes correct, but final is wrong.
- `stable_correct`: draft and all observed postmask/final states stay correct.
- `stable_or_recovered_correct`: draft and final are correct, but at least one intermediate postmask state is wrong.

The labels are trajectory-based, not semantic annotations. They use the same `judge_answer` rule as the PostMaSK runner.

## Experiment Table

| Experiment | N | Draft Acc | Final Acc | Global Persistent | Local Degrade | Draft Error Repaired | Mixed Lost |
|---|---:|---:|---:|---:|---:|---:|---:|
| `postmask_sr0p5_d16_p16_cachedconf_fixed16_refill1_seed42_n400` | 400 | 0.390 | 0.393 | 228 | 4 | 5 | 11 |
| `postmask_sr0p5_d16_p16_cachedconf_fixed32_refill2_seed42_n400` | 241 | 0.398 | 0.415 | 129 | 4 | 8 | 8 |
| `postmask_sr0p5_d16_p16_conf_r2_seed42_n400` | 400 | 0.390 | 0.380 | 244 | 4 | 0 | 0 |
| `postmask_sr0p5_d16_p16_conf_r4_seed42_n400` | 400 | 0.390 | 0.393 | 238 | 5 | 6 | 0 |
| `postmask_sr0p5_d16_p16_fullconf_r4_seed42_n400` | 2 | 0.500 | 0.500 | 1 | 0 | 0 | 0 |
| `postmask_sr0p5_d16_p16_histconf_fixed16_refill1_seed42_n400` | 400 | 0.390 | 0.393 | 243 | 0 | 1 | 0 |
| `postmask_sr0p5_d16_p16_histconf_fixed32_refill2_seed42_n400` | 241 | 0.398 | 0.398 | 142 | 1 | 1 | 2 |
| `postmask_sr0p5_d16_p16_histconf_r4_seed42_n400` | 400 | 0.390 | 0.390 | 244 | 0 | 0 | 0 |
| `postmask_sr0p5_d16_p16_histconf_r4_seed42_n400_new` | 400 | 0.390 | 0.393 | 243 | 0 | 1 | 0 |
| `postmask_sr0p5_d16_p16_histconf_r4_seed42_n400_new_log` | 400 | 0.390 | 0.390 | 244 | 0 | 0 | 0 |
| `postmask_sr0p5_d16_p16_kldiv_fixed16_refill1_seed42_n400` | 400 | 0.390 | 0.393 | 235 | 1 | 2 | 7 |
| `postmask_sr0p5_d16_p16_kldiv_fixed32_refill2_seed42_n400` | 240 | 0.396 | 0.396 | 133 | 5 | 5 | 7 |
| `postmask_sr0p5_d16_p16_kldiv_r4_seed42_n400` | 400 | 0.390 | 0.390 | 244 | 0 | 0 | 0 |
| `postmask_sr0p5_d16_p16_laststepconf_fixed16_refill1_seed42_n400` | 400 | 0.390 | 0.385 | 233 | 5 | 3 | 8 |
| `postmask_sr0p5_d16_p16_laststepconf_r4_seed42_n400` | 400 | 0.390 | 0.385 | 243 | 3 | 1 | 0 |
| `postmask_sr0p5_d16_p16_meanafterfill_fixed16_refill1_seed42_n400` | 400 | 0.390 | 0.395 | 231 | 1 | 3 | 10 |
| `postmask_sr0p5_d16_p16_meanafterfill_fixed32_refill2_seed42_n400` | 242 | 0.401 | 0.409 | 133 | 4 | 6 | 6 |
| `postmask_sr0p5_d16_p16_meanafterfill_r4_seed42_n400` | 400 | 0.390 | 0.390 | 243 | 1 | 1 | 0 |
| `postmask_sr0p5_d16_p16_meanafterfill_r4_seed42_n400_new` | 400 | 0.390 | 0.390 | 243 | 1 | 1 | 0 |
| `postmask_sr0p5_d16_p16_meanafterfill_r4_seed42_n400_new_log` | 400 | 0.390 | 0.390 | 243 | 1 | 1 | 0 |
| `postmask_sr0p5_d16_p16_proposalconf_fixed16_refill1_seed42_n400` | 400 | 0.390 | 0.390 | 234 | 3 | 3 | 7 |
| `postmask_sr0p5_d16_p16_proposalconf_fixed32_refill2_seed42_n400` | 240 | 0.396 | 0.421 | 128 | 4 | 10 | 7 |
| `postmask_sr0p5_d16_p16_rand_r4_seed42_n400` | 400 | 0.390 | 0.383 | 242 | 5 | 2 | 0 |
| `postmask_sr0p5_d16_p16_random_fixed16_refill1_seed42_n400` | 400 | 0.390 | 0.388 | 234 | 2 | 1 | 9 |
| `postmask_sr0p5_d16_p16_random_fixed32_refill2_seed42_n400` | 242 | 0.401 | 0.401 | 133 | 3 | 3 | 9 |
| `postmask_sr0p5_d16_p16_topkmargin_fixed16_refill1_seed42_n400` | 400 | 0.390 | 0.398 | 235 | 2 | 5 | 4 |
| `postmask_sr0p5_d16_p16_topkmargin_fixed32_refill2_seed42_n400` | 238 | 0.399 | 0.408 | 130 | 5 | 7 | 6 |
| `postmask_sr0p5_d16_p16_topkmargin_r4_seed42_n400` | 400 | 0.390 | 0.390 | 244 | 0 | 0 | 0 |
| `postmask_sr0p5_d32_p0_seed42_n400` | 400 | 0.407 | 0.407 | 237 | 0 | 0 | 0 |

## Final Error Attribution

| Experiment | Final Wrong | Global Persistent | Local Degrade | Mixed Lost |
|---|---:|---:|---:|---:|
| `postmask_sr0p5_d16_p16_cachedconf_fixed16_refill1_seed42_n400` | 243 | 228 (0.938) | 4 (0.016) | 11 (0.045) |
| `postmask_sr0p5_d16_p16_cachedconf_fixed32_refill2_seed42_n400` | 141 | 129 (0.915) | 4 (0.028) | 8 (0.057) |
| `postmask_sr0p5_d16_p16_conf_r2_seed42_n400` | 248 | 244 (0.984) | 4 (0.016) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_conf_r4_seed42_n400` | 243 | 238 (0.979) | 5 (0.021) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_fullconf_r4_seed42_n400` | 1 | 1 (1.000) | 0 (0.000) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_histconf_fixed16_refill1_seed42_n400` | 243 | 243 (1.000) | 0 (0.000) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_histconf_fixed32_refill2_seed42_n400` | 145 | 142 (0.979) | 1 (0.007) | 2 (0.014) |
| `postmask_sr0p5_d16_p16_histconf_r4_seed42_n400` | 244 | 244 (1.000) | 0 (0.000) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_histconf_r4_seed42_n400_new` | 243 | 243 (1.000) | 0 (0.000) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_histconf_r4_seed42_n400_new_log` | 244 | 244 (1.000) | 0 (0.000) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_kldiv_fixed16_refill1_seed42_n400` | 243 | 235 (0.967) | 1 (0.004) | 7 (0.029) |
| `postmask_sr0p5_d16_p16_kldiv_fixed32_refill2_seed42_n400` | 145 | 133 (0.917) | 5 (0.034) | 7 (0.048) |
| `postmask_sr0p5_d16_p16_kldiv_r4_seed42_n400` | 244 | 244 (1.000) | 0 (0.000) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_laststepconf_fixed16_refill1_seed42_n400` | 246 | 233 (0.947) | 5 (0.020) | 8 (0.033) |
| `postmask_sr0p5_d16_p16_laststepconf_r4_seed42_n400` | 246 | 243 (0.988) | 3 (0.012) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_meanafterfill_fixed16_refill1_seed42_n400` | 242 | 231 (0.955) | 1 (0.004) | 10 (0.041) |
| `postmask_sr0p5_d16_p16_meanafterfill_fixed32_refill2_seed42_n400` | 143 | 133 (0.930) | 4 (0.028) | 6 (0.042) |
| `postmask_sr0p5_d16_p16_meanafterfill_r4_seed42_n400` | 244 | 243 (0.996) | 1 (0.004) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_meanafterfill_r4_seed42_n400_new` | 244 | 243 (0.996) | 1 (0.004) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_meanafterfill_r4_seed42_n400_new_log` | 244 | 243 (0.996) | 1 (0.004) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_proposalconf_fixed16_refill1_seed42_n400` | 244 | 234 (0.959) | 3 (0.012) | 7 (0.029) |
| `postmask_sr0p5_d16_p16_proposalconf_fixed32_refill2_seed42_n400` | 139 | 128 (0.921) | 4 (0.029) | 7 (0.050) |
| `postmask_sr0p5_d16_p16_rand_r4_seed42_n400` | 247 | 242 (0.980) | 5 (0.020) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_random_fixed16_refill1_seed42_n400` | 245 | 234 (0.955) | 2 (0.008) | 9 (0.037) |
| `postmask_sr0p5_d16_p16_random_fixed32_refill2_seed42_n400` | 145 | 133 (0.917) | 3 (0.021) | 9 (0.062) |
| `postmask_sr0p5_d16_p16_topkmargin_fixed16_refill1_seed42_n400` | 241 | 235 (0.975) | 2 (0.008) | 4 (0.017) |
| `postmask_sr0p5_d16_p16_topkmargin_fixed32_refill2_seed42_n400` | 141 | 130 (0.922) | 5 (0.035) | 6 (0.043) |
| `postmask_sr0p5_d16_p16_topkmargin_r4_seed42_n400` | 244 | 244 (1.000) | 0 (0.000) | 0 (0.000) |
| `postmask_sr0p5_d32_p0_seed42_n400` | 237 | 237 (1.000) | 0 (0.000) | 0 (0.000) |

## Draft Error Attribution

| Experiment | Draft Wrong | Global Persistent | Repaired By Postmask | Mixed Lost |
|---|---:|---:|---:|---:|
| `postmask_sr0p5_d16_p16_cachedconf_fixed16_refill1_seed42_n400` | 244 | 228 (0.934) | 5 (0.020) | 11 (0.045) |
| `postmask_sr0p5_d16_p16_cachedconf_fixed32_refill2_seed42_n400` | 145 | 129 (0.890) | 8 (0.055) | 8 (0.055) |
| `postmask_sr0p5_d16_p16_conf_r2_seed42_n400` | 244 | 244 (1.000) | 0 (0.000) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_conf_r4_seed42_n400` | 244 | 238 (0.975) | 6 (0.025) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_fullconf_r4_seed42_n400` | 1 | 1 (1.000) | 0 (0.000) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_histconf_fixed16_refill1_seed42_n400` | 244 | 243 (0.996) | 1 (0.004) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_histconf_fixed32_refill2_seed42_n400` | 145 | 142 (0.979) | 1 (0.007) | 2 (0.014) |
| `postmask_sr0p5_d16_p16_histconf_r4_seed42_n400` | 244 | 244 (1.000) | 0 (0.000) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_histconf_r4_seed42_n400_new` | 244 | 243 (0.996) | 1 (0.004) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_histconf_r4_seed42_n400_new_log` | 244 | 244 (1.000) | 0 (0.000) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_kldiv_fixed16_refill1_seed42_n400` | 244 | 235 (0.963) | 2 (0.008) | 7 (0.029) |
| `postmask_sr0p5_d16_p16_kldiv_fixed32_refill2_seed42_n400` | 145 | 133 (0.917) | 5 (0.034) | 7 (0.048) |
| `postmask_sr0p5_d16_p16_kldiv_r4_seed42_n400` | 244 | 244 (1.000) | 0 (0.000) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_laststepconf_fixed16_refill1_seed42_n400` | 244 | 233 (0.955) | 3 (0.012) | 8 (0.033) |
| `postmask_sr0p5_d16_p16_laststepconf_r4_seed42_n400` | 244 | 243 (0.996) | 1 (0.004) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_meanafterfill_fixed16_refill1_seed42_n400` | 244 | 231 (0.947) | 3 (0.012) | 10 (0.041) |
| `postmask_sr0p5_d16_p16_meanafterfill_fixed32_refill2_seed42_n400` | 145 | 133 (0.917) | 6 (0.041) | 6 (0.041) |
| `postmask_sr0p5_d16_p16_meanafterfill_r4_seed42_n400` | 244 | 243 (0.996) | 1 (0.004) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_meanafterfill_r4_seed42_n400_new` | 244 | 243 (0.996) | 1 (0.004) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_meanafterfill_r4_seed42_n400_new_log` | 244 | 243 (0.996) | 1 (0.004) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_proposalconf_fixed16_refill1_seed42_n400` | 244 | 234 (0.959) | 3 (0.012) | 7 (0.029) |
| `postmask_sr0p5_d16_p16_proposalconf_fixed32_refill2_seed42_n400` | 145 | 128 (0.883) | 10 (0.069) | 7 (0.048) |
| `postmask_sr0p5_d16_p16_rand_r4_seed42_n400` | 244 | 242 (0.992) | 2 (0.008) | 0 (0.000) |
| `postmask_sr0p5_d16_p16_random_fixed16_refill1_seed42_n400` | 244 | 234 (0.959) | 1 (0.004) | 9 (0.037) |
| `postmask_sr0p5_d16_p16_random_fixed32_refill2_seed42_n400` | 145 | 133 (0.917) | 3 (0.021) | 9 (0.062) |
| `postmask_sr0p5_d16_p16_topkmargin_fixed16_refill1_seed42_n400` | 244 | 235 (0.963) | 5 (0.020) | 4 (0.016) |
| `postmask_sr0p5_d16_p16_topkmargin_fixed32_refill2_seed42_n400` | 143 | 130 (0.909) | 7 (0.049) | 6 (0.042) |
| `postmask_sr0p5_d16_p16_topkmargin_r4_seed42_n400` | 244 | 244 (1.000) | 0 (0.000) | 0 (0.000) |
| `postmask_sr0p5_d32_p0_seed42_n400` | 237 | 237 (1.000) | 0 (0.000) | 0 (0.000) |
