# StepMask FreeCorrection

This folder implements a M3CoT runner for the training-free self-correction idea from
`Training-Free Self-Correction for Multimodal Masked Diffusion Models`.

The implemented loop is:

1. Run the normal masked diffusion transfer for the current step.
2. For already generated tokens, temporarily mask one token at a time.
3. Score the original token by its leave-one-out log likelihood under the current context.
4. Remask the lowest-scoring generated tokens.
5. Continue denoising with the updated state.

The implementation supports these remask metrics:

- `--correction-metric confidence`: remask tokens with low leave-one-out confidence.
- `--correction-metric time_aggregation`: remask tokens with low accumulated leave-one-out log likelihood.
- `--correction-metric topk_margin`: remask tokens with low top-1/top-2 probability margin.
- `--correction-metric kl_divergence`: remask tokens whose leave-one-out distribution changes most from the previous evaluation.

It also supports:

- `--correction-rule deterministic`: remask the lowest-scoring tokens.
- `--correction-rule stochastic`: sample remasked tokens with probability proportional to low scores.

Example:

```bash
python M3CoT/StepMask/run_m3cot_free_correction.py \
  --limit 100 \
  --max-new-tokens 64 \
  --block-length 64 \
  --step-per-block 32 \
  --correction-score cumulated \
  --correction-rule deterministic \
  --transfer-per-step 4 \
  --remask-per-step 2 \
  --correction-metric time_aggregation \
  --loo-chunk-size 16
```

Notes:

- The leave-one-out scoring is batched with `--loo-chunk-size`, but it still adds extra forward passes.
- The transfer count is adjusted dynamically after remasking so that later steps can refill remasked tokens.
- The final step of each block skips remasking to avoid leaving `[MASK]` tokens in the final answer.
- To model "denoise 4 tokens, then remask 2 tokens per step", set `--transfer-per-step 4 --remask-per-step 2`.
- Each record stores `candidate_metrics` for every scored token, including confidence, time aggregation, top-k margin, entropy, and KL divergence.
