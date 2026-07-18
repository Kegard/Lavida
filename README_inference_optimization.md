# LaViDa 推理阶段优化调研

> 调研截止：2026-07-16（Asia/Shanghai）。本文只讨论**不更新 LaViDa 参数、不进行 SFT/RL/蒸馏训练**时可以采用的推理期优化；2026 年新出现但需要训练的工作会单独标注为“参考上界”，不列入直接实施方案。

## 1. 结论摘要

LaViDa 的推理瓶颈不是单一的“生成 token 太多”，而是以下乘积：

```text
总成本 ≈ 视觉 token 数 × 去噪步数 × 每步参与计算的 token 数
       + 视觉重编码/额外验证成本
```

LaViDa 原始实现已经提供了三个重要控制点：视觉编码后形成 multimodal prefix、prefix KV cache，以及基于 confidence 的 remasking 和 `step_ratio/schedule` 采样调度。仓库中已有的 `Attention/`、`PDM`、`Scale_Attention/`、`M3CoT/PostVRG/` 可直接作为观测和原型入口。

综合文献后，推荐优先级如下：

| 优先级 | 方向                                              | 是否严格满足“只推理” |                      预期收益 | 主要风险                     |
| ------ | ------------------------------------------------- | ---------------------: | ----------------------------: | ---------------------------- |
| P0     | 逐去噪步视觉注意力/熵/置信度遥测                  |                     是 |    找到真正可剪枝、可刷新阶段 | 观测指标不一定等价于视觉证据 |
| P1     | 文本条件视觉 token 剪枝 + 2D 保形/可恢复路由      |                     是 | 降低 prefill、KV 和每步注意力 | OCR、计数、小目标容易损失    |
| P1     | 全图到 ROI 的推理期渐进视觉增强                   |                     是 |          提升细粒度感知和 OCR | ROI 错误会造成自我强化       |
| P1     | remasking、步数和 transfer schedule 联合搜索      |                     是 |         直接改善速度-质量曲线 | 可能改变 LaViDa 的质量上限   |
| P2     | uncertain-step 视觉对比验证/VCD、草稿-验证-重采样 |                     是 |                  降低视觉幻觉 | 额外 forward，必须控制触发率 |
| P2     | token 级早停、动态重算、近似 cache/服务优化       |                     是 |                降低长回答延迟 | 工程复杂度和 cache 失配      |

最值得先做的组合是：**PDM/attention 诊断 → 保留空间结构的视觉 token 路由 → early-global/late-local 的 ROI 刷新 → 只在高不确定性步骤做视觉验证**。这条路线不改权重，且能分别报告每个机制的收益。

## 2. LaViDa 基线和推理约束

### 2.1 必读基线

1. [LaViDa: A Large Diffusion Language Model for Multimodal Understanding](https://arxiv.org/abs/2505.16839)（NeurIPS 2025 Spotlight）：将 SigLIP 视觉编码器接入 masked discrete diffusion LM；关键推理设计包括 complementary masking、prefix KV cache、timestep shifting。论文报告了可调的速度-质量折中，因此 `NFE/去噪步数` 是第一类推理变量。
2. [LaViDa 官方代码](https://github.com/jacklishufan/LaViDa)：当前仓库的 `predict.py`、`llada/generate.py`、prefix 构造和 KV cache 语义应以此实现为准。
3. [LaViDa-O](https://arxiv.org/abs/2509.19244)：LaViDa 的统一理解与生成扩展，用于了解统一 multimodal dLLM 的视觉条件接口；如果目标 checkpoint 是 LaViDa-O，需要重新核对视觉 token 与生成 token 的布局。
4. [LaViDa-R1](https://arxiv.org/abs/2602.14147)：多模态推理后训练工作。它证明了任务级 reasoning、grounding 和 generation 的潜力，但核心贡献含 SFT/RL，因此不属于本文的直接推理方案。

### 2.2 重要约束

- LaViDa 的生成是 masked diffusion/迭代去噪，不是严格左到右 AR；“第几个输出词”不能直接等价为“第几个视觉阶段”。应使用 timestep、mask ratio、transfer token 数、remasking 事件和 entropy/confidence 定义阶段。
- 视觉表示通常在 prefix 阶段编码一次并跨多步复用。任何“后期引入局部视觉证据”的方案都必须明确：是替换视觉 prefix、追加 ROI token、修改视觉 token 权重，还是另建一条验证分支。
- 删除视觉 token 是不可逆操作；对 OCR、计数、空间关系和小目标任务，应保留二维网格结构、局部邻域或可恢复候选池。
- 不能只看 MMMU/MME 平均分。需要同时测视觉依赖性、幻觉、OCR/细粒度感知、wall-clock latency、forward 次数、显存峰值和有效视觉 token 数。

## 3. 细粒度方向一：视觉输入、分辨率与 ROI

### 3.1 低分辨率全局图 + 高分辨率局部图

| 论文                                                              | 中文简介                                                                                                                        | 对 LaViDa 的推理期可用性                                                                                          |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| [LLaVA-UHD](https://arxiv.org/abs/2403.11703)                      | 将任意宽高比高分辨率图像拆成可变大小的局部块，并保留压缩后的全局上下文，说明“全局布局 + 局部细节”比简单缩放更适合细粒度 VQA。 | 主要是架构启发；LaViDa 可在推理时构造 global/ROI 两组视觉 prefix，但不要直接假设 checkpoint 支持任意分辨率。      |
| [Dragonfly](https://arxiv.org/abs/2406.00977)                      | 用多分辨率、全局图和局部 crop 组合增强小目标、文字和细节理解。                                                                  | 可作为多 crop 输入的基线；应把额外视觉编码成本纳入端到端延迟。                                                    |
| [Zoom-Refine](https://arxiv.org/abs/2506.01663)                    | 明确提出 training-free 的 localized zoom + self-refinement：先回答/定位，再放大相关区域并重新回答。                             | 适合直接改造成 LaViDa 的“初稿→ROI→局部验证/重采样”流程，但要限制最多 1 次 ROI 重编码。                        |
| [AwaRes / Look Where It Matters](https://arxiv.org/abs/2603.16932) | 以低分辨率全图为起点，按问题检索高分辨率局部区域，目标是在固定成本下补回关键细节。                                              | 可借鉴为 query-conditioned ROI 检索；LaViDa 中可由问题 token 与视觉 token 相似度、attention 或外部 OCR 产生候选。 |
| [Visual Funnel](https://arxiv.org/abs/2512.10362)                  | 指出只保留 salient crop 会产生“上下文失明”，提出让高保真局部细节和全局语境保持信息通路。                                      | 对 LaViDa 很重要：不要用 ROI 完全替换全图，优先采用 global tokens + ROI tokens 或带门控的融合。                   |
| [Zooming without Zooming](https://arxiv.org/abs/2602.11858)        | 将反复 zoom 的局部视觉信息压回图像级表示，减少多次工具调用和视觉重编码。                                                        | 论文方法可能包含训练/蒸馏，不直接作为零训练方案；可作为“如何减少 ROI 重编码”的设计上界。                        |

### 3.2 推理期建议

建议先做四组可证伪对照：`full image`、`blurred full image`、`global + random crop`、`global + attention/OCR ROI`。切换点用去噪步比例或 mask ratio 表示，例如前 1/3 步只用全局，后 2/3 步追加 ROI；同时测 early-local、late-local 和全程 ROI，避免把“多看了一张图”误判为调度收益。

ROI 选择优先级：问题中的实体/区域词 → OCR/检测候选 → 当前去噪步骤对视觉 token 的聚合 attention → 不确定性最高的视觉区域。若使用模型自身 attention 选 ROI，必须加入随机/中心/oracle ROI 对照，防止注意力自我强化。

## 4. 细粒度方向二：视觉 token 剪枝、合并与路由

### 4.1 静态或一次性压缩

| 论文                                         | 中文简介                                                                                                                              | 推理期判断                                                                                                                             |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| [DyVTE](https://arxiv.org/abs/2411.19628)     | 通过经验分析区分视觉 token 在早期融合、模态内建模和后期多模态推理阶段的作用，并采用动态 visual-token exit。                           | 很适合指导 LaViDa 的阶段化预算；需要把“层”映射为 diffusion timestep/denoising stage。                                                |
| [ST$^3$](https://arxiv.org/abs/2412.20105)  | 利用空间和时间上的视觉 token 冗余做 trimming，强调 token 重要性随层和阶段变化。                                                       | 图片任务可取空间部分；视频/多图输入可扩展到跨帧冗余。                                                                                  |
| [VASparse](https://arxiv.org/abs/2501.06553)  | 将视觉感知信号用于 token sparsification，以减少视觉幻觉，同时避免需要二次完整解码的代价。                                             | 可作为“视觉相关 token 优先保留”的训练免调基线；需重新定义 LaViDa 的视觉相关性分数。                                                  |
| [ERASE](https://arxiv.org/abs/2605.09982)     | 自适应两阶段 token pruning，针对高分辨率图像中的视觉冗余，动态决定压缩强度。                                                          | 适合做输入级预算控制；建议按问题难度和视觉 token 熵动态分配预算。                                                                      |
| [DCP-Prune](https://arxiv.org/abs/2606.16633) | 发现极低 token budget 下，保留 token 与完整特征的分布偏移会显著增大；用分布一致性、anchor-context recovery 和文本感知选择提高稳定性。 | 强烈建议加入 LaViDa 的 ultra-low budget 对照；不要只按 attention top-k 截断。                                                          |
| [F$^3$A](https://arxiv.org/abs/2605.16359)  | 将视觉 token 剪枝视为 task-conditioned evidence search，在固定预算下按任务和模型规模分配 token。                                      | 可把问题类型、视觉 token 熵和候选区域覆盖率用于 adaptive budget。                                                                      |
| [AsymVLM](https://arxiv.org/abs/2605.29535)   | 利用视觉 token 空间冗余高、文本 token 依赖强的模态不对称性，在 prefill 前对视觉 token 激进剪枝，并采用样本自适应预算。                | 论文含 learned importance scorer；严格零训练时只借鉴模态不对称预算，用 attention/PDM 等现有信号替代 scorer，并保留位置编码和局部覆盖。 |

### 4.2 动态、可恢复和后层路由

| 论文                                                             | 中文简介                                                                                                                                  | 对 LaViDa 的启发                                                                                              |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| [Reroute, Don&#39;t Remove](https://arxiv.org/abs/2606.12412)     | 认为一次性 rank-and-remove 不可靠，因为 token 重要性会随 decoder depth 改变；training-free 地让 deferred token 在后续阶段重新进入候选池。 | 这是比永久剪枝更适合 masked diffusion 的方向：把候选池按 denoising stage 分批注入，尤其保护 grounding token。 |
| [Focus-then-Context / SPpruner](https://arxiv.org/abs/2605.20950) | 先聚焦问题相关主体，再逐步恢复其上下文关系，避免只留下孤立目标。                                                                          | 可设计`focus -> local context -> global context` 的视觉 prefix schedule；适合空间关系和多目标比较。         |
| [Late-Layer Fusion is Enough](https://arxiv.org/abs/2606.09131)   | 针对视觉信息在深层仍被重复处理的问题，采用双路径视觉 token routing，让部分视觉 token 只在需要的后层融合。                                 | 与 LaViDa 的“视觉 prefix 被每个去噪步骤重复使用”高度相关；可先实现 attention mask/视觉权重门控的近似版。    |
| [ST-Merge](https://arxiv.org/abs/2606.29350)                      | training-free 地在视觉编码阶段合并时空冗余 token，并做位置修正；报告在视频 VLM 上明显降低延迟。                                           | 单图可退化为空间 token merge；必须验证 SigLIP patch 的位置编码和 OCR 是否被破坏。                             |

### 4.3 对 LaViDa 的落地规则

1. **优先做 recoverable routing，再做 irreversible pruning**：被延期的 token 不参与当前阶段，但保留原始 embedding 和坐标，在不确定性升高或 grounding 失败时恢复。
2. **保形而非只保分数**：top-k 中至少覆盖全局网格、问题相关区域和其邻域；记录保留 token 的二维覆盖率。
3. **区分 prefill 与 denoising savings**：视觉 prefix 变短可减少首轮 prefill 和 KV 显存，但若每步的视觉 KV 已被缓存，端到端收益可能小于预期；必须报告两者。
4. **预算按任务分配**：普通 caption/VQA 可尝试 25%-50% 视觉 token；OCR、计数、细粒度区域和 grounding 应使用更高预算或触发 ROI 回补。

## 5. 细粒度方向三：扩散解码、remasking 与阶段调度

| 论文                                                                                                | 中文简介                                                                                                             | 适配策略                                                                                                                   |
| --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| [ReMDM](https://arxiv.org/abs/2503.00307)                                                            | 提出 inference-time remasking，让已经生成的 token 在后续步骤重新变为 mask，从而通过迭代 refinement 修正早期错误。    | 直接测试`low_confidence/random/entropy/margin` remasking；对 LaViDa 应同时记录重新 mask 的视觉相关答案 token。           |
| [SAID](https://arxiv.org/abs/2606.04974)                                                             | Scaffold-Aware Iterative Decoding 把更多去噪计算先投入 scaffold token，再逐步扩展到其他 token，以减少无效迭代。      | 可将 scaffold 定义为答案格式、实体名或 grounding 槽位；与 LaViDa 的固定 block canvas 结合时需保持长度不变。                |
| [When to Plan, When to Polish](https://arxiv.org/abs/2606.21802)                                     | 将 noise level 视为生成粒度控制信号：高噪声阶段形成粗粒度结构，低噪声阶段精修词面。                                  | 直接支持“early global / late local”假设，但需要用 LaViDa 的 step-level 日志验证，而不能直接照搬文本模型结论。            |
| [LESS Is More](https://arxiv.org/abs/2606.16908)                                                     | 以 mutual-stability 作为采样依据，减少反复处理但不稳定的 token，探索 dLLM 的 inference-time scaling。                | 可把视觉证据稳定性纳入 transfer priority：稳定且有视觉支持的 token 先提交，不稳定 token 延后。                             |
| [PerceptionDLM](https://arxiv.org/abs/2606.19534)                                                    | 利用扩散模型的并行解码，在多个视觉区域上同时生成 caption，并设计结构化 attention mask；面向多区域感知效率。          | 其模型/训练并非 LaViDa 的零改动插件，但“多 ROI 并行 caption”非常适合用于 LaViDa 的区域验证分支。                         |
| [Masked Diffusion Models are Secretly Time-Agnostic Masked Models](https://arxiv.org/abs/2409.02908) | 分析 masked diffusion 与 time-agnostic masked model 的关系，提醒不能想当然地认为每个 timestep 都带来独立的时间语义。 | 做 timestep-aware 视觉刷新前，先测不同 timestep 的 logits/attention 可分性；若分布相近，优先优化 remasking 和 token 路由。 |

### 5.1 建议的零训练采样控制器

对第 `t` 个去噪步骤记录：

```text
u_t       = masked-token 平均 entropy
c_t       = 被 transfer token 的平均 confidence
v_t       = answer -> visual token 的 attention/梯度自由相关性
r_t       = 最近两步 token 预测的一致性
```

可用规则：

```text
若 u_t 高 且 v_t 低：增加视觉验证或回补视觉 token
若 u_t 低 且 r_t 高：跳过/合并一部分 denoising 计算
若 c_t 高 但 v_t 低：延迟提交，避免语言先验造成视觉幻觉
若 grounding/格式检查失败：只 remask 相关 answer span，不重采样整段答案
```

这类控制器只改变推理路径，不改变模型权重；阈值应在 validation split 上固定后再测试，避免对测试集调参。

## 6. 细粒度方向四：视觉 grounding、幻觉抑制与验证式解码

### 6.1 可直接采用的推理期方法

| 论文                                                                  | 中文简介                                                                                  | LaViDa 适配                                                                                                                       |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| [Visual Contrastive Decoding (VCD)](https://arxiv.org/abs/2311.07525)  | 同时使用正常视觉输入和弱视觉/扰动视觉输入，用两者 logits 差异抑制语言先验导致的对象幻觉。 | 在 LaViDa 中可做 weak-visual branch（null/blur/noise image）并只对高 entropy 步骤做 guidance；需验证 masked logits 差异是否稳定。 |
| [OPERA](https://arxiv.org/abs/2308.03764)                              | 通过识别视觉 attention sink、惩罚过度依赖少数视觉 token，并回溯重分配注意力来降低幻觉。   | 适合做视觉 attention sink 诊断和视觉 token reweight；不能直接假设 AR 的 retrospection 顺序适用于 diffusion。                      |
| [VASparse](https://arxiv.org/abs/2501.06553)                           | 将视觉感知信号用于稀疏化，减少幻觉并避免完整二次解码。                                    | 可与视觉 token routing 合并，优先保留支持当前问题的区域。                                                                         |
| [Preemptive Hallucination Reduction](https://arxiv.org/abs/2505.24007) | 在输入阶段预先增强/校正视觉条件，主张从 hallucination 产生前处理视觉信息。                | 可实现为问题驱动的 crop、OCR 提示或视觉 prefix 重加权，不需要训练；需把预处理耗时单独计入。                                       |
| [Generate, but Verify](https://arxiv.org/abs/2504.13169)               | 先生成候选，再用视觉一致性验证决定是否回退和重采样。                                      | 与 LaViDa 的 draft/remask 天然兼容；只重采样低置信 span，而不是整段回答。                                                         |
| [See What You Are Told](https://arxiv.org/abs/2503.03321)              | 研究视觉 attention sink，发现高 attention 不一定意味着真正的视觉证据。                    | 说明不能单独用 attention top-k 做 ROI；应结合遮挡/替换 probe 或 answer perturbation。                                             |

### 6.2 2026 年的新方向和边界

- [Look Carefully](https://arxiv.org/abs/2602.24041)：在解码中自适应增强与当前回答相关的视觉 token，避免无差别强化全部视觉 token。可作为 LaViDa 的 token-level visual reinforcement 参考；若其实现需要额外模型，部署时应单独计费。
- [Locate-then-Sparsify](https://arxiv.org/abs/2603.16284)：先用归因定位 hallucination 相关层，再做稀疏 feature steering。其归因数据/校准过程带有离线准备，适合作为“部署前校准、部署时无训练”的条件方案，不应称为完全免训练。
- [Mitigating Multimodal Hallucination via Phase-wise Self-reward](https://arxiv.org/abs/2604.17982)：提出按语义阶段做 self-reward decoding，但需要蒸馏轻量 reward model；属于训练相关参考，不列入严格零训练主线。
- [Bridging Modality Disconnect in Self-Reflection](https://arxiv.org/abs/2602.18746)：通过 draft、视觉区域 critique、verification、revision 闭环减少幻觉，但 ReflectV 数据和训练流程意味着它是可迁移的流程设计，不是直接插件。
- [Listening makes Vision Clear](https://arxiv.org/abs/2606.23763)：指出 answer-side attention 会受已生成文本和边界 token 污染，提出 prompt-side PV-TAM。对 LaViDa 的直接价值是改进视觉证据诊断：优先用 prompt/query-side attribution，而不是只看答案 token 的 attention。

### 6.3 推荐的 LaViDa 验证闭环

```text
draft diffusion
    -> 找到低 confidence / 高 entropy 的 answer span
    -> 用 query-side visual score 找 ROI
    -> 对 ROI 做 crop/blur/occlusion 或 weak-visual 对照
    -> 只 remask 受影响 span
    -> 再做少量 denoising/refill
```

应至少包含 `full visual`、`weak visual`、`null visual` 三种条件，避免把 guidance 的收益误认为单纯增加了计算量。

## 7. 细粒度方向五：cache、服务和部署效率

| 论文/资源                                                 | 中文简介                                                                                        | LaViDa 价值                                                                                                                       |
| --------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| [LaViDa prefix KV cache](https://arxiv.org/abs/2505.16839) | LaViDa 原始工作通过缓存 prefix 加速多步采样，是当前实现的第一基线。                             | 先确认任何视觉 token 剪枝/ROI 刷新是否破坏 cache；视觉 prefix 改变后必须重新建立对应 KV。                                         |
| [Sangam](https://arxiv.org/abs/2607.04206)                 | 讨论 dLLM 不能直接复用 AR 的精确 KV cache，并将 block 级 dLLM 服务映射到成熟 AR serving stack。 | 可指导批处理、prefill/decode 分离和近似 cache；它是系统优化，不会改善视觉质量。                                                   |
| [DiLaServe](https://arxiv.org/abs/2606.29094)              | 面向 diffusion LM 的高 SLO 服务，围绕多步去噪和服务调度优化吞吐/尾延迟。                        | 多请求 LaViDa 部署可参考；单样本研究阶段优先测真实 GPU wall-clock。                                                               |
| [Sparse-LaViDa](https://arxiv.org/abs/2512.14008)          | 通过动态截断无用 masked token 和 register token，报告最多约 2x 加速。                           | 这是 LaViDa 系列最直接的速度参考，但其专用 register token 和 attention mask 需要训练/架构配合；在本文约束下只作上界，不直接采用。 |

可选的部署级手段包括视觉 tower 的 FP16/BF16/INT8 对照、CUDA kernel/FlashAttention、batch padding 整理和异步 ROI 编码。它们必须分别报告数值误差、视觉 tower 时间、语言 backbone 时间和峰值显存。

## 8. 严格排除项：为什么不能直接当作“推理优化”

1. [LaViDa-R1](https://arxiv.org/abs/2602.14147)、[MMaDA](https://arxiv.org/abs/2505.15809)：核心收益来自 SFT/RL/统一后训练，不能在本项目中作为纯推理改进。
2. [Sparse-LaViDa](https://arxiv.org/abs/2512.14008)：需要专用 register token 与训练时 attention mask 来匹配稀疏采样；可作为性能上界和未来训练路线，不能把其结果宣称为 zero-shot inference gain。
3. [Zooming without Zooming](https://arxiv.org/abs/2602.11858)：其 region-to-image distillation 思路可能需要模型训练；本项目只采用“减少重复 ROI 编码”的思想，不复用训练结果。
4. [PerceptionDLM](https://arxiv.org/abs/2606.19534)：面向并行区域 caption 的专用模型，结构化 mask 和并行能力由模型/训练共同支持；LaViDa 只能借鉴其区域并行验证流程。
5. [PSRD](https://arxiv.org/abs/2604.17982)、[MIRROR](https://arxiv.org/abs/2602.18746)：需要 reward/反思数据或训练，属于未来扩展，不放入严格主实验。

## 9. 面向当前代码的实施顺序

### 阶段 A：建立可复现基线和诊断

- 固定 checkpoint、视觉 tower、prompt、`max_new_tokens`、batch、dtype、GPU；分别扫 `step_ratio`、`schedule` 和 `remasking`。
- 每个 denoising step 记录：mask 数、transfer 数、平均/分位 confidence、entropy、remask 数、视觉/prompt/已生成文本 attention 占比、wall-clock。
- 复用 `Lavida/Attention/` 的逐步 attention 分析、`Lavida/PDM/` 的 token 级视觉重要性和 `Lavida/Scale_Attention/` 的视觉权重/重加权脚本。

### 阶段 B：低风险视觉压缩

- 先做输入级视觉 token budget sweep：100%、75%、50%、25%；比较 attention top-k、PDM、随机、空间均匀、query-conditioned 五种选择。
- 加入 2D coverage、邻域保留和 deferred pool；优先复现 Reroute 的 recoverable routing，而不是直接永久删除 token。
- 只在视觉 prefix 不变时使用原 KV cache；发生 ROI 刷新时重新计时 prefill 和 cache 建立成本。

### 阶段 C：视觉阶段调度和验证

- 做 `global-only`、`global->ROI`、`ROI-from-start`、`global+ROI` 四组消融，并以 denoising step 比例作为切换变量。
- 只对高 entropy/低视觉相关性样本触发一次 crop 或 VCD weak-visual forward。
- 只 remask 受视觉验证影响的 answer span；记录修正率、退化率和额外 forward 次数。

### 阶段 D：动态控制器与部署

- 用 mutual stability、confidence 和视觉支持分数决定 token transfer、remask 或跳步。
- 再加入近似 cache、batch serving、视觉 tower 量化；任何系统优化都不能与模型质量变化混在一个实验中。

## 10. 评测协议

### 10.1 按能力拆分

| 能力           | 建议任务/指标                                             | 重点检查                     |
| -------------- | --------------------------------------------------------- | ---------------------------- |
| 通用视觉问答   | MMMU、MME、MMBench                                        | 平均能力不能因剪枝退化       |
| OCR/细节       | TextVQA、DocVQA、OCRBench                                 | 小字、数字、颜色和计数       |
| 空间/grounding | RefCOCO/RefCOCO+、区域 caption、Screenspot                | ROI 和二维结构是否保留       |
| 幻觉/视觉依赖  | POPE、CHAIR、Hallucination benchmark、视觉遮挡/替换 probe | 是否真的看图而非依赖语言先验 |
| 生成/长回答    | COCO caption、长答案自一致性                              | remasking 与验证的收益/代价  |

[Seeing without Looking](https://arxiv.org/abs/2605.22903) 提醒：一些 VLM benchmark 分数并不能证明模型使用了图像，因此必须加入 image shuffle、null image、blur、region occlusion 和 counterfactual crop 对照。

### 10.2 必报效率指标

- 端到端 p50/p95 latency、吞吐、峰值显存；不要只报理论 FLOPs。
- NFE/denoising steps、实际 forward 次数、额外 ROI/weak-visual forward 次数。
- 原始视觉 token 数、保留/合并/回补 token 数、每一步有效 token 数。
- 质量-延迟曲线，而不是单点 speedup；至少给出相对 baseline 的 `accuracy drop <= 1 point` 和 `latency reduction` 区间。

## 11. 最小可行实验矩阵

```text
模型：LaViDa-LLaDa、LaViDa-Dream（若资源允许）
任务：MMMU 子集 + TextVQA + POPE + 细粒度/grounding 子集
采样：step_ratio {0.25, 0.5, 0.75, 1.0}
调度：schedule {none, shift, cosine, logit_normal}
remasking：{low_confidence, entropy, margin, random}
视觉预算：{100%, 75%, 50%, 25%}
视觉路由：{attention top-k, PDM, spatial-uniform, recoverable routing}
视觉刷新：{none, early-global/late-ROI, global+ROI}
验证：{none, VCD, span-level remask/refill}
```

第一阶段的停止标准：如果 oracle ROI 在细粒度任务上都没有收益，就停止 attention-driven ROI；如果 50% 视觉 token 在通用任务下降超过 1 point，则优先研究 recoverable routing/局部回补，而不是继续压低 budget；如果验证分支的额外 forward 抵消延迟收益，则只保留质量分析，不把它作为加速方案。

## 12. 参考资料索引

- LaViDa 系列：[LaViDa](https://arxiv.org/abs/2505.16839)、[LaViDa-O](https://arxiv.org/abs/2509.19244)、[Sparse-LaViDa](https://arxiv.org/abs/2512.14008)、[LaViDa-R1](https://arxiv.org/abs/2602.14147)。
- 视觉输入与高分辨率：[LLaVA-UHD](https://arxiv.org/abs/2403.11703)、[Dragonfly](https://arxiv.org/abs/2406.00977)、[Zoom-Refine](https://arxiv.org/abs/2506.01663)、[Visual Funnel](https://arxiv.org/abs/2512.10362)、[Zooming without Zooming](https://arxiv.org/abs/2602.11858)、[AwaRes](https://arxiv.org/abs/2603.16932)。
- 视觉 token：[DyVTE](https://arxiv.org/abs/2411.19628)、[ST3](https://arxiv.org/abs/2412.20105)、[VASparse](https://arxiv.org/abs/2501.06553)、[ERASE](https://arxiv.org/abs/2605.09982)、[F3A](https://arxiv.org/abs/2605.16359)、[AsymVLM](https://arxiv.org/abs/2605.29535)、[SPpruner](https://arxiv.org/abs/2605.20950)、[Reroute](https://arxiv.org/abs/2606.12412)、[DCP-Prune](https://arxiv.org/abs/2606.16633)、[ST-Merge](https://arxiv.org/abs/2606.29350)。
- dLLM 解码与服务：[ReMDM](https://arxiv.org/abs/2503.00307)、[SAID](https://arxiv.org/abs/2606.04974)、[LESS Is More](https://arxiv.org/abs/2606.16908)、[PerceptionDLM](https://arxiv.org/abs/2606.19534)、[Sangam](https://arxiv.org/abs/2607.04206)、[DiLaServe](https://arxiv.org/abs/2606.29094)。
- 视觉 grounding/幻觉：[VCD](https://arxiv.org/abs/2311.07525)、[OPERA](https://arxiv.org/abs/2308.03764)、[Preemptive Hallucination Reduction](https://arxiv.org/abs/2505.24007)、[Generate but Verify](https://arxiv.org/abs/2504.13169)、[Look Carefully](https://arxiv.org/abs/2602.24041)、[Locate-then-Sparsify](https://arxiv.org/abs/2603.16284)、[PV-TAM](https://arxiv.org/abs/2606.23763)。
