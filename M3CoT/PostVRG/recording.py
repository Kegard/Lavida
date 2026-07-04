import torch
import torch.nn.functional as F


def gather_token_logprobs(logits, token_ids):
    log_probs = F.log_softmax(logits.to(torch.float64), dim=-1)
    return log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


def gather_topk_margin(logits):
    probs = F.softmax(logits.to(torch.float64), dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def compute_tokenwise_kl_from_logprobs(current_log_probs, previous_log_probs):
    previous_log_probs = previous_log_probs.to(
        device=current_log_probs.device,
        dtype=current_log_probs.dtype,
    )
    current_probs = current_log_probs.exp()
    return (current_probs * (current_log_probs - previous_log_probs)).sum(dim=-1)


def build_mean_scores(value_sum, value_count):
    scores = torch.full_like(value_sum, float("inf"), dtype=torch.float64)
    seen = value_count > 0
    if seen.any():
        scores[seen] = value_sum[seen] / value_count[seen].to(torch.float64)
    return scores, seen


class PostVRGRecorder:
    def __init__(self, tokenizer, decode_answer, max_new_tokens, device):
        self.tokenizer = tokenizer
        self.decode_answer = decode_answer
        self.max_new_tokens = int(max_new_tokens)

        shape = (self.max_new_tokens,)
        self.history_conf_sum = torch.zeros(shape, dtype=torch.float64, device=device)
        self.history_conf_count = torch.zeros(shape, dtype=torch.long, device=device)
        self.after_fill_conf_sum = torch.zeros(shape, dtype=torch.float64, device=device)
        self.after_fill_conf_count = torch.zeros(shape, dtype=torch.long, device=device)
        self.margin_sum = torch.zeros(shape, dtype=torch.float64, device=device)
        self.margin_count = torch.zeros(shape, dtype=torch.long, device=device)
        self.kl_sum = torch.zeros(shape, dtype=torch.float64, device=device)
        self.kl_count = torch.zeros(shape, dtype=torch.long, device=device)
        self.previous_after_fill_log_probs = {}

        self.draft_records = []
        self.postmask_records = []

    def build_step_stats(self, logits, x0, answer_slice):
        return {
            "token_logprobs": gather_token_logprobs(logits, x0),
            "token_margin": gather_topk_margin(logits),
            "answer_log_probs": F.log_softmax(
                logits[0, answer_slice].to(torch.float64),
                dim=-1,
            ),
        }

    def observe_draft_before_fill(self, stats, x, answer_slice):
        answer_mask = (x == self.mask_token_id)[:, answer_slice][0]
        token_logprobs = stats["token_logprobs"]
        token_margin = stats["token_margin"]
        answer_log_probs = stats["answer_log_probs"]

        if answer_mask.any():
            self.history_conf_sum[answer_mask] += token_logprobs[0, answer_slice][answer_mask].to(torch.float64)
            self.history_conf_count[answer_mask] += 1

        filled_answer = ~answer_mask
        if not filled_answer.any():
            return

        filled_token_ids = x[0, answer_slice][filled_answer]
        filled_log_probs = answer_log_probs[filled_answer]
        filled_log_conf = filled_log_probs.gather(
            dim=-1,
            index=filled_token_ids.unsqueeze(-1),
        ).squeeze(-1)
        self.after_fill_conf_sum[filled_answer] += filled_log_conf
        self.after_fill_conf_count[filled_answer] += 1
        self.margin_sum[filled_answer] += token_margin[0, answer_slice][filled_answer].to(torch.float64)
        self.margin_count[filled_answer] += 1
        self._observe_kl_for_positions(filled_answer, answer_log_probs)

    def observe_selected_positions(self, stats, selected_seq_positions, prefix_length):
        token_logprobs = stats["token_logprobs"]
        token_margin = stats["token_margin"]
        answer_log_probs = stats["answer_log_probs"]
        for seq_pos in selected_seq_positions.detach().cpu().tolist():
            if seq_pos < prefix_length:
                continue
            answer_pos = int(seq_pos - prefix_length)
            self.after_fill_conf_sum[answer_pos] += token_logprobs[0, seq_pos].to(torch.float64)
            self.after_fill_conf_count[answer_pos] += 1
            self.margin_sum[answer_pos] += token_margin[0, seq_pos].to(torch.float64)
            self.margin_count[answer_pos] += 1
            self._observe_kl_for_position(answer_pos, answer_log_probs[answer_pos])

    def observe_refilled_positions(self, stats, refilled_seq_positions, prefix_length):
        token_logprobs = stats["token_logprobs"]
        token_margin = stats["token_margin"]
        answer_log_probs = stats["answer_log_probs"]
        answer_positions = []
        for seq_pos in refilled_seq_positions.detach().cpu().tolist():
            if seq_pos < prefix_length:
                continue
            answer_pos = int(seq_pos - prefix_length)
            regenerated_logprob = token_logprobs[0, seq_pos].to(torch.float64)
            self.history_conf_sum[answer_pos] += regenerated_logprob
            self.history_conf_count[answer_pos] += 1
            self.after_fill_conf_sum[answer_pos] += regenerated_logprob
            self.after_fill_conf_count[answer_pos] += 1
            self.margin_sum[answer_pos] += token_margin[0, seq_pos].to(torch.float64)
            self.margin_count[answer_pos] += 1
            self._observe_kl_for_position(answer_pos, answer_log_probs[answer_pos])
            answer_positions.append(answer_pos)
        return answer_positions

    def append_draft_record(
        self,
        step,
        guidance_used,
        num_filled,
        selected_seq_positions,
        answer_ids,
        num_masked_after_step,
    ):
        self.draft_records.append(
            {
                "step": int(step),
                "phase": "draft",
                "guidance_used": bool(guidance_used),
                "num_filled": int(num_filled),
                "selected_positions": [
                    int(pos) for pos in selected_seq_positions.detach().cpu().tolist()
                ],
                "state_text": self.decode_answer(self.tokenizer, answer_ids),
                "num_masked_after_step": int(num_masked_after_step),
            }
        )

    def append_postmask_record(
        self,
        step,
        remasked_answer_positions,
        remasked_token_ids,
        refilled_answer_positions,
        selection_scores,
        answer_ids,
    ):
        self.postmask_records.append(
            {
                "step": int(step),
                "phase": "postmask",
                "remasked_answer_positions": [
                    int(pos) for pos in remasked_answer_positions.detach().cpu().tolist()
                ],
                "remasked_token_ids": (
                    [int(token_id) for token_id in remasked_token_ids]
                    if remasked_token_ids is not None
                    else None
                ),
                "remasked_token_texts": self.decode_token_texts(remasked_token_ids),
                "remasked_rule_categories": None,
                "refilled_answer_positions": [
                    int(pos) for pos in refilled_answer_positions
                ],
                "selection_scores": [
                    float(score.item()) for score in selection_scores.detach().cpu()
                ],
                "state_text": self.decode_answer(self.tokenizer, answer_ids),
            }
        )

    def decode_token_texts(self, token_ids):
        if token_ids is None:
            return None
        return [
            self.tokenizer.decode([int(token_id)], skip_special_tokens=False)
            for token_id in token_ids
        ]

    def build_payload(self, proposal_confidence):
        history_confidence, seen_history = build_mean_scores(
            self.history_conf_sum,
            self.history_conf_count,
        )
        mean_after_fill, seen_after_fill = build_mean_scores(
            self.after_fill_conf_sum,
            self.after_fill_conf_count,
        )
        topk_margin, seen_margin = build_mean_scores(
            self.margin_sum,
            self.margin_count,
        )
        mean_kl_scores, seen_kl = build_mean_scores(self.kl_sum, self.kl_count)
        kl_divergence = torch.full_like(mean_kl_scores, float("inf"), dtype=torch.float64)
        if seen_kl.any():
            kl_divergence[seen_kl] = -mean_kl_scores[seen_kl]

        return {
            "draft_records": self.draft_records,
            "postmask_records": self.postmask_records,
            "history_confidence": history_confidence.detach().cpu().tolist(),
            "history_confidence_count": self.history_conf_count.detach().cpu().tolist(),
            "proposal_confidence": proposal_confidence.detach().cpu().tolist(),
            "mean_after_fill": mean_after_fill.detach().cpu().tolist(),
            "mean_after_fill_count": self.after_fill_conf_count.detach().cpu().tolist(),
            "topk_margin": topk_margin.detach().cpu().tolist(),
            "topk_margin_count": self.margin_count.detach().cpu().tolist(),
            "kl_divergence": kl_divergence.detach().cpu().tolist(),
            "kl_divergence_count": self.kl_count.detach().cpu().tolist(),
            "meta_stats": {
                "history_confidence_mean": (
                    float(history_confidence[seen_history].mean().item())
                    if seen_history.any()
                    else None
                ),
                "mean_after_fill_mean": (
                    float(mean_after_fill[seen_after_fill].mean().item())
                    if seen_after_fill.any()
                    else None
                ),
                "proposal_confidence_mean": float(proposal_confidence.mean().item()),
                "topk_margin_mean": (
                    float(topk_margin[seen_margin].mean().item())
                    if seen_margin.any()
                    else None
                ),
                "kl_divergence_mean": (
                    float((-kl_divergence[seen_kl]).mean().item())
                    if seen_kl.any()
                    else None
                ),
            },
        }

    @property
    def mask_token_id(self):
        from M3CoT.run_m3cot_stepwise_x0 import MASK_TOKEN_ID

        return MASK_TOKEN_ID

    def _observe_kl_for_positions(self, answer_positions_mask, answer_log_probs):
        for answer_pos in torch.nonzero(
            answer_positions_mask,
            as_tuple=False,
        ).view(-1).detach().cpu().tolist():
            self._observe_kl_for_position(answer_pos, answer_log_probs[answer_pos])

    def _observe_kl_for_position(self, answer_pos, current_pos_log_probs):
        previous_pos_log_probs = self.previous_after_fill_log_probs.get(int(answer_pos))
        if previous_pos_log_probs is not None:
            kl_value = compute_tokenwise_kl_from_logprobs(
                current_pos_log_probs.unsqueeze(0),
                previous_pos_log_probs.unsqueeze(0),
            )[0]
            self.kl_sum[answer_pos] += kl_value
            self.kl_count[answer_pos] += 1
        self.previous_after_fill_log_probs[int(answer_pos)] = current_pos_log_probs.detach().to(
            device="cpu",
            dtype=torch.float32,
        )
