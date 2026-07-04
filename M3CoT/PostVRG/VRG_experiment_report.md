# VRG on LaViDa / M3CoT — 实验整理与汇报材料

> 数据集: M3CoT test (n=2318, seed=42)，模型: LaViDa (LLaDA backbone, `weight/lavida-reason`)
> 所有数字均来自 `M3CoT/PostVRG/outputs/*/summary.json` 与 `vrg_added_change_summary/`，可复现。

---

## 一、一句话总览

我们想用**视觉引导 (Visual Reference Guidance, VRG)** 让扩散式多模态模型在推理时更"看图"，从而提升 M3CoT 上的多步多模态推理准确率。
**结论**：把推理拆成 *Proposal(草稿) → Remask(重掩码) → Refill(回填)* 三段后，**重掩码-回填本身**带来主要收益 (38.52 → 40.60)；在回填阶段叠加 **VRG 视觉引导**只带来很小的额外增益 (40.60 → 40.72)。诊断分析揭示了**为什么 VRG 收益不大**——它确实强烈扰动了候选 logits 并把它们推向视觉相关 token，但**很少改变最终被选中并写入的答案 token**。

---

## 二、研究动机与思路 (Why)

1. **问题**：扩散语言模型 (dLLM) 一次性并行解码多个 token，单步内 token 之间相互独立采样，容易产生"自洽但不看图"的草稿——推理链条通顺、但与视觉证据脱节。
2. **假设**：如果在解码过程中显式注入"有图 vs 无图"的对比信号 (VCD 风格的视觉引导)，可以把 logits 拉向视觉相关 token，纠正"没看图"的错误。
3. **机制设计 (Pipeline)**：
   - **Stage 1 — Proposal (草稿)**: 正常跑到第 k 步，读出一个完整的 x0 草稿。
   - **Stage 2 — Remask (重掩码)**: 用某种选择策略 (置信度 / margin / KL / 视觉增益…) 选出一批"可疑"位置重新打掩码。
   - **Stage 3 — Refill (回填)**: 对这批位置做短的精炼回填；**VRG 在这一步注入**：`guided_logit = cond_logit + α·(cond_logit − weak_visual_logit)`，weak visual 用扩散噪声 (`vcd-noise-step=500`) 或 null-image 实现。
4. **要回答的核心问题**：VRG 到底有没有在"看图"？收益不大是因为它没改 logits，还是改了 logits 但没改最终答案？

---

## 三、主结果 (Slide 1)

| Method | Overall | Commonsense | Mathematics | Science | 说明 |
|---|---:|---:|---:|---:|---|
| **Base** | **38.52** | 67.03 | 24.48 | 32.61 | 原始一次性解码，无 remask |
| Proposal | 40.60 | 68.79 | **31.12** | **34.09** | 草稿 + remask + 回填 (无 VRG) |
| PostVRG | **40.72** | 68.57 | 30.71 | 34.40 | Proposal 基础上回填阶段加 VRG (α=0.5, noise500) |
| Visual Select | 39.39 | 66.37 | 29.05 | 33.35 | 用 visual-gain 选 remask 位置 |

**读法**：
- 收益主要来自 **Proposal (remask→refill 结构本身)**：`+2.08` (38.52→40.60)，且在 **Mathematics (+6.6)** 和 **Science (+1.5)** 上最明显——这两个子域最依赖多步重算。
- VRG 叠加在上面只 **+0.12** (40.60→40.72)，几乎打平，部分子域 (Commonsense/Math) 甚至略降。
- 用"视觉增益"来挑 remask 位置 (Visual Select) **反而掉点** (39.39 < 40.60)：说明"视觉相关"≠"该被修正"，低置信≠与视觉有关。

> **Slide 1 一句话标题建议**：*"结构化的草稿-重掩码-回填带来主要增益；VRG 视觉引导只是锦上添花。"*

---

## 四、诊断一：Full-stage VRG 确实改变 logits、推向视觉 token (Slide 2)

对全流程 VRG 做 logits 追踪 (`analyze_fullstage_vrg_token_switches.py`，2318 样本 / 1.27M switch 事件)：

| 范围 | 切换率 | 含义 |
|---|---:|---|
| **Active masked positions** | **51.3%** (1,255,016 / 2,447,808) | 仍处于掩码的候选位置，top-1 候选被 VRG 改写的比例 |
| **Selected filled positions** | **8.9%** (13,214 / 148,352) | 真正被填入的位置，top-1 被改的比例 |

- VRG 把大量 active 位置的 top-1 从**通用语言 token** 推向 **颜色 / 数值 / 表格-value / 方向** 等视觉相关 token（如 `the→Value (+10.7)`, `the→blue`, `the→green`, `the→red`, `\n→0`）。
- 时序上 (Where switches happen across steps)：**active 切换在第 1 步最猛、随步数衰减**；**selected 切换在后期步骤才上升**——即 VRG 早期狂改候选，但只有后期少量变成真正写入。

> **结论**：VRG **确实在增强 diffusion decoding 对视觉信息的关注**，机制是生效的——它把候选分布拉向视觉证据。

> **Slide 2 标题建议**：*"VRG 把候选 logits 显著推向视觉 token —— 机制确实在'看图'。"*

---

## 五、诊断二：VRG 在 refill 阶段主要改"路径"，几乎不改"最终答案 token" (Slide 3)

`vrg_added_change_summary` / `refill_vrg_marginal_effect`：

| key metric | Proposal + VRG | Visual + VRG | (Full-stage) VRG |
|---|---:|---:|---:|
| Selected top1 changed | 905 (1.22%) | 0 (0.00%) | 13214 (8.91%) |
| Guided-only selected | 7591 (10.23%) | 1766 (2.38%) | 10134 (6.83%) |
| Selected rank changed | 15169 (20.45%) | 3691 (4.98%) | 11247 (7.58%) |
| Refill order changed (steps) | 19115 (51.5%) | 4936 (13.3%) | — |
| **Improved / Worsened / Net** | **28 / 25 / +3** | 0 / 0 / 0 | — |

**三个指标含义**：
- **Selected top1 changed**：最终被填入的位置上，cond 的 top1 与 guided/VRG 的 top1 是否不同 → 只有 1.22%。
- **Guided-only selected**：cond 排序本来进不了 top-k、但 VRG 把它推进了 top-k 的位置 → 10.23%（VRG 在改候选集合）。
- **Selected rank changed**：被选中位置上 VRG 是否改变了优先级/是否属于 top-k → 20.45%。

> **结论 (Slide 3 标题)**：*"VRG 在 refill 阶段主要改变'回填路径与候选排序'(51.5%/20.45%)，而不是直接替换最终答案 token (仅 1.22%)。"* 这就解释了为什么 net 只有 +3 (28 改对 / 25 改错)——扰动很大，但落到最终决策上的净效应被改对/改错几乎抵消。

---

## 六、诊断三：visual-gain 作为视觉引导，收益不显著 (Slide 4)

| Method | Weak visual | alpha | Acc |
|---|---|---:|---:|
| Baseline | none | – | 38.52 |
| PostMask (Proposal) | none | – | 40.60 |
| **PostVRG** | noise | 0.5 | **40.72** |

- 全量实验中，**refill 策略**和 **refill + visual** 都有正向收益，但 **visual 收益不够显著**。
- **visual 收益不大的原因 (推测)**：在 **math 子域**，模型无法严格遵循 question 产生结果——即使推理正确，最终选项也会写错；推测是 **VRG 视觉引导扰动了格式/答案 token 的稳定性**所致 (与诊断二的"改路径不改答案、且改对改错相抵"一致)。

> **Slide 4 标题建议**：*"visual-gain 引导收益有限：扰动落在格式/答案 token 上，改对与改错相互抵消。"*

---

## 七、把四页串成一条故事线 (汇报顺序)

1. **动机**：dLLM 并行解码 → 草稿"不看图" → 想用 VRG 视觉引导纠正。(口述)
2. **Slide 1 主结果**：结构 (Proposal) 是主要收益来源，VRG 仅 +0.12，Visual-Select 反而掉点 → 抛出疑问"VRG 到底在干什么？"
3. **Slide 2 机制验证**：VRG **确实**把 logits 推向视觉 token (active 51.3%) → 机制是对的。
4. **Slide 3 归因**：但它主要改**路径/候选排序** (20–51%)，**几乎不改最终写入的答案 token** (1.22%)，net 仅 +3 → 解释收益小。
5. **Slide 4 失败模式**：尤其在 math，VRG 扰动答案/格式 token 稳定性，改对改错相抵 → 收益被吃掉。
6. **收尾**：方向不是"加更强的视觉引导"，而是"让视觉引导精准作用在答案决策 token 上 / 只在确实需要看图的位置触发"。

---

## 八、研究启发与下一步 (Research Takeaways)

**核心洞察**：VRG 的瓶颈不在"强度"，而在"作用点"——它扰动了海量候选，但这些扰动很少落到决定答案的 token 上，且改对改错近乎抵消 (28 vs 25)。

可探索的方向：

1. **把引导限制在"决策 token"上**：仅对答案/选项位置 (`{A}{B}…`)、关键数值/实体 token 施加 VRG，而非全序列；或在 selected/filled 阶段 (而非 active 候选阶段) 才注入，避免无效扰动。
2. **门控触发 (gated VRG)**：用一个"是否需要看图"的判据 (如 cond/uncond 的 visual-gain 阈值) 决定**这个位置是否启用 VRG**，避免对"低置信但与视觉无关"的 token 强行视觉化——这正是 Visual-Select 掉点的教训。
3. **校准而非替换**：当前 net 收益被改对/改错抵消，说明 VRG 缺乏"何时该信"的判据。引入置信度校准 (refill-vrg-calibration) 或一致性投票，只在 cond 与 guided 一致或高置信时才接受 VRG 的改写。
4. **针对 math 的格式鲁棒性**：math 子域"推理对但答案写错"是收益吃掉的主因。可对答案抽取格式做约束解码 (constrained decoding on the answer slot)，把 VRG 的视觉收益与答案格式解耦。
5. **更早注入 vs 更晚注入**：诊断二显示 active 切换早期猛、selected 切换后期才升。可实验"两阶段调度"——早期用 VRG 重排候选、后期关闭 VRG 以稳定答案 token。
6. **重新定义"视觉相关性"指标**：Visual-Select 用 visual-gain 选位反而掉点，说明现有 visual-gain 不能区分"该改"与"不该改"。值得做一个更好的 saliency/causal 视觉归因指标 (项目里已有 `Causal_Analysis/`) 来挑 remask 位置。

---

## 附:关键文件索引

- 主结果 runs: `M3CoT/PostVRG/outputs/main_no_postmask_*` (Base), `main_proposal_postmask_*` (Proposal), `main_postvrg_alpha0p5_noise500_fixed32_refill2_*` (PostVRG), `M3CoT/PostMaSK/outputs/postmask_visualgain_*` (Visual Select)
- Slide 2 (logits 推向视觉): `analyze_fullstage_vrg_token_switches.py` → `outputs/main_fullstage_vrg_token_switches_full`
- Slide 3 (key metric 表): `summarize_vrg_added_changes.py` → `outputs/vrg_added_change_summary/{summary.md,proposal_key_summary.csv}`
- Slide 3 (marginal effect): `analyze_refill_vrg_marginal_effect.py` → `outputs/refill_vrg_marginal_effect_full.json`
- 改对/改错案例: `outputs/proposal_vrg_gain_case_analysis.md`
- 方法实现: `Proposal/run_m3cot_proposal_refine.py`, `PostMaSK/run_m3cot_postmask.py`, `PostVRG/run_m3cot_postvrg.py`, `PostVRG/run_m3cot_fullstage_vrg.py`
