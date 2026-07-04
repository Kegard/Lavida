# LaViDa 扩散多模态推理 · 视觉引导 (VRG) 系统性汇报

> **模型**: LaViDa (LLaDA backbone, `weight/lavida-reason`) — 扩散式多模态理解模型
> **主数据集**: M3CoT test (n=2318, seed=42)；辅助: TextVQA val (n=100~1000)
> 所有数字均来自 `*/outputs/*/summary.json`，已逐条核对，可复现。

---

## 0. 一页纸结论 (Executive Summary)

我们研究"如何让扩散式多模态模型在推理时更看图"。围绕一个统一的 **草稿→重掩码→回填 (Proposal → Remask → Refill)** 框架，做了 80+ 个全量/小样本实验。核心结论四条：

1. **结构带来温和但稳定的收益**：把单次解码改成"草稿+重掩码回填"，M3CoT 从 **38.52 → 40.60 (+2.1)**；TextVQA (n=1000, 相对真实 base) 从 **33.5 → 39.8 (+6.3)**。两个任务都为正、但都不算大;收益来源是"重算可疑位置"，不是视觉引导本身。
2. **VRG 视觉引导收益很小但为正**：在回填阶段叠加 VRG，M3CoT 仅 **40.60 → 40.72 (+0.12)**；且 **null-visual 控制 (0.4060) 与 noise-visual (0.4072) 几乎相同**，说明当前增益主要来自"扰动+重排"而非真正的视觉对比。
3. **机制确实在"看图"，但作用点错位**：logits 追踪显示 VRG 把 **51.3%** 的候选 top-1 推向视觉 token，但只有 **1.22%** 的最终写入 token 被改变，net 仅 **+3 (28 改对/25 改错)** — 扰动巨大、落点稀疏、改对改错相抵。
4. **"挑哪里重掩码"比"怎么引导"更关键**：remask 选择策略 (置信度/margin/KL/visual-gain) 的影响远大于 VRG α；用 visual-gain 选位反而掉点 (39.39)。

**研究启发**：方向不是"加更强的视觉引导",而是"**把引导精准作用在决策 token 上 + 用门控判断哪里真的需要看图**"。

---

## 1. 研究背景与动机 (Why)

### 1.1 问题
扩散语言模型 (dLLM) 一次性并行解码多个 token，单步内 token 相互独立采样。这带来两个隐患：
- **草稿"自洽但不看图"**：推理链通顺，却与视觉证据脱节；
- **早期低置信 token 被过早定稿**，后续无法纠正。

### 1.2 核心假设
如果在解码时显式注入"有图 vs 弱视觉"的对比信号 (VCD 风格)，可把 logits 拉向视觉相关 token，纠正"没看图"的错误。

### 1.3 统一框架 (本项目所有方法的母体)
```
Stage 1  Proposal  : 正常扩散解码到第 k 步，读出完整 x0 草稿
Stage 2  Remask    : 用选择策略挑出"可疑"位置重新打掩码 (固定/动态预算)
Stage 3  Refill    : 对这些位置做短回填精炼；VRG 在这一步注入：
                     guided = cond + α·(cond − weak_visual)
                     weak_visual ∈ {diffusion_noise(step500), null_image}
```
- **PostMaSK** = 上述框架但回填**不加** VRG (纯重算)
- **PostVRG** = 回填阶段**加** VRG
- **Full-stage VRG** = 从头到尾每步都加 VRG (不止回填)

---

## 2. 主结果 (M3CoT 全量 n=2318)

| Method | Overall | Commonsense | Math | Science | 说明 |
|---|---:|---:|---:|---:|---|
| **Base** (no remask) | **38.52** | 67.03 | 24.48 | 32.61 | 单次解码 |
| Proposal (PostMaSK) | 40.60 | 68.79 | **31.12** | **34.09** | 草稿+重掩码回填，无 VRG |
| **PostVRG** (α=0.5,noise) | **40.72** | 68.57 | 30.71 | 34.40 | 回填阶段加 VRG |
| Visual Select | 39.39 | 66.37 | 29.05 | 33.35 | 用 visual-gain 选 remask 位置 → **掉点** |

**关键读法**
- 主要增益来自**结构** (+2.08)，集中在 **Math (+6.6)** / **Science (+1.5)** — 最依赖多步重算的子域。
- VRG 仅 **+0.12**，Commonsense/Math 子域甚至略降。
- 用"视觉相关性"挑位置反而**掉 1.2 个点**："视觉相关"≠"该被修正"。

---

## 3. 全面消融与扫描 (这部分是新增、四页 PPT 没覆盖的)

### 3.1 弱视觉信号的来源:noise vs null（关键控制实验）
| 配置 | weak visual | Overall | 解读 |
|---|---|---:|---|
| PostVRG α=0.5 | diffusion noise(500) | **40.72** | 主报告数字 |
| PostVRG α=0.5 | **null image** | **40.60** | **几乎相同** → 增益≠真实视觉对比 |
| PostVRG α=1.0 | null image | 39.86 | α 过大反而掉 |
| Full-stage α=1.0 | null image | 40.60 | 全程引导也只打平 |
| Full-stage α=1.0 | noise(500) | 38.18 | 全程 + 噪声反而最差 |

> **结论**：null-visual 与 noise-visual 收益几乎一致 → 当前 VRG 的增益主要来自"对候选分布的扰动 + 重排序"，而非"看图 vs 不看图"的真实对比。这是对 VRG 有效性最重要的一条 caveat。

### 3.2 引导强度 α 与 top-k 范围
| 变体 | n | Overall |
|---|---:|---:|
| PostVRG k=4 α=0.5 | 2318 | 40.68 |
| PostVRG k=8 α=0.5 | 2318 | 40.64 |
| Full-stage α=0.5 | 400 | 39.25 |
| Full-stage α=1.0 | 400 | 40.50 |
| Full-stage α=2.0 | 400 | 39.75 |
| softconf / hardconf 校准 | 2318 | 40.55 / 40.51 |

> α 存在最优区间 (~0.5–1.0)，过大破坏稳定性；置信度校准 (soft/hard) 没带来额外收益。

### 3.3 Remask 选择策略大扫描 (n=400，draft 一律 0.39)
| 选择策略 | Final Acc | vs proposal-only(0.4075) |
|---|---:|---|
| **proposal_confidence** (fixed32,refill2) | **0.4175** | **+1.0** ✅ |
| topk_margin (fixed32) | 0.4125 | +0.5 |
| mean_after_fill (fixed32) | 0.4125 | +0.5 |
| cached_confidence (fixed32) | 0.4150 | +0.75 |
| kl_divergence | 0.3975 | −1.0 |
| visual_gain | 0.3925 | −1.5 ❌ |
| random | 0.3975 | −1.0 |
| last_step_confidence | 0.3850 | −2.3 ❌ |

> **核心洞察**：选哪些位置重掩码，决定成败。基于**自身置信度**的策略最稳；基于**视觉增益**的策略最差。这与主表 Visual-Select 掉点一致。

### 3.4 预算分配 (固定 budget=32: 保留 p 个 / 重掩码 r 个)
| p / r | proposal | final | gain |
|---|---:|---:|---:|
| native x0conv t0.90 s2 | 38.75 | **41.25** | **+2.5** ✅ |
| p16 / r16 (rr0.50) | 41.25 | 41.25 | 0 |
| p24 / r8 (rr0.25) | 40.50 | 41.00 | +0.5 |
| p28 / r4 (rr0.125) | 41.25 | 40.50 | −0.75 |
| p32 / r0 (no remask) | 40.75 | 40.75 | 0 |

> 重掩码比例适中 (~25%) 且从较弱的草稿出发 (native, 38.75→41.25) 时收益最大；草稿已经很好时再重掩码无益甚至有害。

### 3.5 Proposal 生成长度 / 步数研究 (n=400, 答案位准确率随步数)
| gen length | 步数 | 末步 acc | 峰值 acc |
|---|---:|---:|---:|
| 64 | 32 | 48.5 | 49.0 @step26 |
| **128** | 64 | **56.5** | **57.25 @step48** |
| 256 | 128 | 52.0 | 53.0 @step122 |
| 512 | 256 | 53.0 | 53.0 @step252 |

> **重要**:生成越长 ≠ 越准。len=128 (中等长度) 显著优于 256/512。而且**峰值常出现在中间步** (step48<64)，末步反而回落 → 提示"早停 / 选最优中间步"本身就是一个增益方向。

### 3.6 触发位置 / 时段研究
| 变体 | Overall | 解读 |
|---|---:|---|
| second-half VRG (后半程才引导) | 38.52 | 无收益 |
| position-boost (按位置加权) | 37.92–39.04 | 普遍掉点 |
| visual-warmup r0.1 | 40.03 | 轻微 |
| visual-warmup r0.5 α0.5 | 40.60 | 打平 proposal |
| rule-content (只引导规则部分) | 39.47–39.65 | 掉点 |
| prompt-contrast (混淆图对比) | 40.55 | 接近但不超 |

---

## 4. 机制诊断:VRG 到底在干什么? (对应原 PPT slide 2-3)

### 4.1 VRG 确实把 logits 推向视觉 token (slide 2)
全流程 VRG logits 追踪 (2318 样本 / 1.27M switch 事件)：
- **Active masked 位置**: top-1 切换率 **51.3%** — 被推向 颜色/数值/表格-value/方向 token (`the→Value +10.7`, `the→blue/green/red`)
- **Selected filled 位置**: 仅 **8.9%** 被改
- 时序：active 切换第 1 步最猛→衰减；selected 切换后期才上升

> 机制**是对的**:VRG 在增强 decoding 对视觉信息的关注。

### 4.2 但它改"路径"不改"答案" (slide 3)
| key metric | Proposal+VRG | Visual+VRG | Full-stage VRG |
|---|---:|---:|---:|
| Selected top1 changed | 905 (1.22%) | 0 (0.00%) | 13214 (8.91%) |
| Guided-only selected | 7591 (10.23%) | 1766 (2.38%) | 10134 (6.83%) |
| Selected rank changed | 15169 (20.45%) | 3691 (4.98%) | 11247 (7.58%) |
| Refill order changed (steps) | 19115 (51.5%) | 4936 (13.3%) | — |
| **Improved/Worsened/Net** | **28/25/+3** | 0/0/0 | — |

> VRG 主要改变**回填路径与候选排序** (20–51%)，**几乎不改最终写入答案** (1.22%)；net 仅 +3，改对改错近乎抵消。这解释了为什么收益这么小。

### 4.3 失败模式:math 子域 (slide 4)
- math: base 24.48 → proposal/softconf **31.12 (+6.6)**，是结构收益最大的子域；
- 但 VRG 在 math 上**不再加分** (30.71 ≤ 31.12)；
- 原因推测:即使推理正确，VRG 扰动了**答案/格式 token 的稳定性**，导致选项写错。

---

## 5. 跨任务对照:TextVQA 上 refine 收益略大于 M3CoT,但都温和

**只用相对真实 base 的全量数字** (n=1000, base = 不做任何 refine 的正常解码):

| 任务 | 真实 Base | with refine | 相对 base 的 gain |
|---|---:|---:|---:|
| M3CoT (n=2318) | 38.52 | 40.72 | **+2.2** |
| TextVQA (n=1000, proposal) | 33.48 | 39.65 | **+6.2** |
| TextVQA (n=1000, k4 remask) | 33.48 | 39.81 | **+6.3** |

> **结论 (已修正)**:refine 框架在两个任务上都是**温和但稳定的正收益** (M3CoT +2、TextVQA +6),**并非 ~10× 的量级差异**。
>
> ⚠️ **重要更正**:之前一版报告里写的 "TextVQA 31→50 (+19)、是 M3CoT 的近 10 倍" 是**错误对比**。那里的 "31" 是某次 sweep **自身的中间 proposal-step 草稿**(一个被人为削弱的早期解码),不是模型的真实 base;拿 final 去比"自己的弱草稿"会虚高收益。真实 base 是 33.5,best sweep 约 48→50,**净提升只有 2 个点左右**,和 M3CoT 同量级。
>
> 修正后的解读:TextVQA(纯视觉读取型) 收益 (+6) **略高于** M3CoT(多步推理型) 收益 (+2),方向上仍支持"草稿含可恢复视觉错误时 refine 更有效",但**不能把它当成最有力的辩护点** —— 两个任务的提升都不大,这本身就是当前方法的局限。

---

## 6. 周边支撑性分析 (项目里的诊断工具链)

| 模块 | 做了什么 | 价值 |
|---|---|---|
| `Entropy/` | M3CoT token 熵分析 | 定位"低置信但与视觉无关"的 token |
| `PDM/` | token 级视觉敏感度 (mask 图像看 logits 变化) | 给"哪些 token 真的依赖视觉"提供 ground-truth |
| `Sink/` | attention sink / KV-norm / query collapse | 解释 dLLM 视觉注意力为何弱 |
| `Scale_Attention/` | LASCD 层选择 / attention 重加权 (TACA) | 另一条"增强视觉"的正交路线 |
| `Causal_Analysis/` | 把视觉作为因果处理,结果变量=答案正确性 | 为"该不该看图"提供因果判据 |
| `AdaptRVRG/` | 自适应阈值触发 VRG (λ,thr) | 门控触发的早期实现 |

> 这些模块共同支撑一个判断:**"该看图的 token" 是少数且可识别的**(PDM/Entropy/Causal 都指向这一点),正好对应第 4 节的发现——VRG 应该只在这些 token 上触发。

---

## 7. 研究启发与下一步 (Takeaways)

**核心瓶颈**:VRG 不是"强度不够",而是"**作用点错位 + 缺乏门控**"。它扰动海量候选,却很少落到决定答案的 token 上,且改对改错相抵 (28 vs 25)。

可执行方向(按优先级):

1. **作用点定位 (highest impact)**:只对答案/选项位 `{A}{B}`、关键数值/实体 token 施加 VRG;用 `PDM/` 的视觉敏感度作为"是否施加"的 mask,而不是全序列扰动。
2. **门控触发 (gated VRG)**:用 cond/uncond 的 visual-gain + 置信度 双阈值决定"这个位置是否启用 VRG",避免对"低置信但与视觉无关"的 token 强行视觉化 (Visual-Select 掉点的直接教训)。`AdaptRVRG/` 已有雏形,可扩展。
3. **换任务验证 (诚实定位)**:在更"视觉读取型"的 benchmark (DocVQA/ChartQA/InfoVQA) 上做全量,看 refine 收益是否随"视觉读取占比"系统性增大。注意 TextVQA 当前也只有 +6 (33.5→39.8),并非压倒性证据 —— 目标是验证"任务越偏视觉读取、收益越大"这个趋势,而不是宣称大幅提升。
4. **proposal-step / 早停**:第 3.5 节显示峰值在中间步、长度 128 最优。做"选最优中间步作为草稿"或自适应早停,可能比 VRG 更省更准。
5. **校准接受准则**:net 被改对改错抵消 → 引入一致性投票/置信度校准,只在 cond 与 guided 高度一致或高置信时接受 VRG 改写。
6. **answer-slot 约束解码 (针对 math)**:把答案抽取格式做约束,解耦"视觉收益"与"答案格式稳定性",修复 math 上 VRG 不加分的失败模式。
7. **与 attention 路线结合**:`Sink/`+`Scale_Attention/` 表明 dLLM 视觉注意力本身偏弱;在 logits 引导(VRG)之外叠加 attention 重加权(TACA),可能是正交增益。

---

## 8. 汇报故事线 (建议讲 8-10 页)

1. **动机**:dLLM 并行解码 → 草稿不看图 (1 页)
2. **统一框架**:Proposal→Remask→Refill,VRG 注入点 (1 页)
3. **主结果**:结构 +2.1,VRG +0.12,Visual-Select 掉点 (1 页, slide1)
4. **关键控制**:null vs noise 几乎相同 → 抛疑问 (1 页, 新增)
5. **机制验证**:VRG 推 logits 向视觉 (51.3%) (1 页, slide2)
6. **归因**:改路径不改答案 (1.22%, net+3) (1 页, slide3)
7. **跨任务**:TextVQA +6 vs M3CoT +2 → refine 在偏视觉读取的任务上略好,但都温和 (1 页)
8. **消融全景**:remask 策略 + 预算 + α 一张大图 (1 页, 新增)
9. **失败模式 + 启发**:math 不加分 → 作用点/门控/换任务 (1-2 页, slide4+启发)

---

## 附:关键文件索引

- 主结果: `PostVRG/outputs/main_no_postmask_*`(Base), `main_proposal_postmask_*`(Proposal), `main_postvrg_alpha0p5_noise500_*`(PostVRG), `PostMaSK/outputs/postmask_visualgain_*`(Visual Select)
- null vs noise 控制: `main_postvrg_alpha0p5_nullvisual_*`, `main_fullstage_vrg_*_nullvisual_*`
- Remask 策略扫描: `PostMaSK/results_summary.md` + `PostMaSK/outputs/postmask_sr0p5_*`
- 预算分配: `benchmark/Remask/budget32_*`, `budgetNative_x0conv_*`
- Proposal 长度/步数: `Proposal/outputs/{64,128,256,512}_stepwise_x0_reason_cot/`
- logits 推向视觉: `PostVRG/analyze_fullstage_vrg_token_switches.py` → `main_fullstage_vrg_token_switches_full/`
- 改路径不改答案: `PostVRG/summarize_vrg_added_changes.py` → `vrg_added_change_summary/`
- 改对/改错案例: `PostVRG/outputs/proposal_vrg_gain_case_analysis.md`
- TextVQA refine 收益: `VRG/outputs/textvqa_proposal_refine_sweep/`, `textvqa_base_proposal_k4/`
- 视觉敏感度/因果: `PDM/`, `Causal_Analysis/lavida_causal_analysis.md`, `Entropy/`, `Sink/`, `Scale_Attention/`
