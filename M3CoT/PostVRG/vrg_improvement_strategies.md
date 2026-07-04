# Refill 阶段 VRG 的改进策略 (位置已固定为 Proposal 选择)

> 前提:位置选择已确定用 **proposal_confidence**(refill 集合固定,`refill_overlap=1.0`)。
> 本文只讨论:**在这个固定 refill 集合上,VRG 引导本身怎么改进**,每条都有现有数据支撑。

---

## 0. 一句话结论

现有数据指出一个明确方向:**当前 VRG 失败的核心不是"引导太弱",而是"无差别接受所有改写"——改对(28)和改错(24)几乎各半**。
基于已有 trace,**用 confidence-delta 做"接受门控"**,把 net 从 +3 提升到 +9~+10(估算 acc 0.4072→**~0.410**),是**唯一有现成数据支撑、且代价为零(纯后处理,不需重跑生成)**的改进方向。

---

## 1. 先看已经试过什么:所有"更聪明的引导"都更差(反面证据)

在 proposal 选位上,我们已经跑了 12 个 VRG 变体(n=2318):

| final acc | 变体 | 改了什么 |
|---:|---|---|
| **0.4072** | **VRG α=0.5 noise** | 朴素全量引导(当前最优) |
| 0.4068 | VRG k=4 α=0.5 | 限制 top-k 范围 |
| 0.4063 | VRG k=8 α=0.5 | 限制 top-k 范围 |
| 0.4059 | proposal base | 不加 VRG |
| 0.4059 | VRG α=0.5 **null** | weak-visual 用全零图 |
| 0.4055 | softconf 校准 | 按置信度软加权 |
| 0.4055 | promptcontrast | 用"混淆图"做对比 |
| 0.4050 | hardconf 校准 | 按置信度硬门控 |
| 0.3986 | VRG α=1.0 null | 加大 α |
| 0.3964 | **rulecontent** | 只引导 rule/推理 span |
| 0.3925 | **highgain_lowconf** | 只引导"低置信+高视觉增益"token |
| 0.3886 | **highvisual** | 只引导"高视觉敏感"token |

**两条关键结论(都有数据)**:
1. **越是"在生成阶段挑 token 施加引导",越差**:highvisual(0.389)、highgain_lowconf(0.393)、rulecontent(0.396)全部低于朴素全量(0.407),甚至低于 proposal base(0.406)。
   → **生成阶段的事前 token 选择不可靠**(和 Visual-Select 选位掉点同源:"高视觉敏感"挑的是模板/格式 token,不是答案词)。
2. **加大 α、换 weak-visual 来源、置信度软/硬校准,都没有正收益**。α=1.0 反而掉到 0.399;null≈noise 说明增益非真实视觉。
   → **不要再在"引导强度/来源/事前加权"上调参,这条路已经被数据证否。**

---

## 2. 真正的问题在哪:接受准则,不是引导强度

`proposal_vrg_gain_case_analysis.json` 把 VRG **改变了最终预测**的 79 个样本拆开看:

| VRG 改变预测后的结果 | 数量 |
|---|---:|
| improved_by_vrg (改对了) | 28 |
| worsened_by_vrg (改错了) | 24 |
| both_wrong (错→错,白改) | 27 |

> **VRG 改预测时,改对和改错几乎是抛硬币(28 vs 24),另有 27 次是无效改动。** net 收益 +3 就是这么被吃掉的。
> 这说明:**VRG 有信号(它能改对 28 个),但没有"何时该信"的判据,于是连带改错 24 个。如果能把改错的挡回去,收益立刻翻倍。**

---

## 3. 改进策略 A(首选,有现成数据):confidence-delta 接受门控

**思路**:VRG 想改写一个 token 时,看它带来的 **selected confidence delta**(改写后该位置 softmax 置信度的变化)。只有当 VRG 让置信度**上升**时才接受改写,否则保留 cond 的原结果。

**判别性证据**(improved vs worsened 的均值差):

| 特征 | improved(28) | worsened(24) | 方向 |
|---|---:|---:|---|
| mean_selected_confidence_delta | **+0.0067** | +0.0028 | improved 更高 ✅ |
| mean_selected_rank_delta | **+0.282** | +0.182 | improved 更高 ✅ |
| guided_only_selected | **4.21** | 3.40 | improved 更高 ✅ |
| active_change_rate | **0.080** | 0.067 | improved 更高 ✅ |

**用门控重放现有 trace(零成本,纯后处理)**:

| 策略 | 接受的改写 | net correct (vs base) | 估算 acc |
|---|---|---:|---:|
| 当前(接受全部改写) | improved 28 / worsened 24 | **+3** | 0.4072 |
| **Gate A: confidence_delta > 0** | improved 22 / worsened 13 | **+9** | **~0.4098** |
| **Gate B: confidence_delta>0 且 rank_delta>0** | improved 22 / worsened 12 | **+10** | **~0.4103** |

> Gate A 的机制:在 79 个改写里,confidence_delta>0 的有 54 个(22 对 / 13 错),confidence_delta≤0 的有 25 个(只 6 对 / 11 错)。**把后者挡回去,保住 11 个改错、只损失 6 个改对,净赚 +5~+6。**

⚠️ **诚实标注**:这是在**同一批数据上**用 trace 重放估算的(in-sample),是改进**上界的指示**,不是已验证的 held-out 结果。要做实的话:在生成时实时计算 confidence_delta 做门控,或用一半数据定阈值、另一半验证。但**判别方向(improved 的 confidence_delta 系统性高于 worsened)是数据里实打实的**,值得实现并验证。

---

## 4. 改进策略 B:把"事后接受门控"提升为"事中校准接受"

策略 A 是二值门控。更细一点:既然 `improved` 的 `guided_only_selected`(4.21)和 `rank_delta`(0.28)都高于 `worsened`,可以做一个**轻量打分接受**:

```
accept_score = w1·Δconfidence + w2·Δrank + w3·guided_only_flag
只有 accept_score 超阈值才接受 VRG 改写,否则回退 cond
```

**数据支撑**:第 3 节四个特征在 improved/worsened 上全部单调可分;权重可在 79 个 changed 样本(或扩展到 2318 全量 trace)上用 logistic 回归拟合。这比单特征门控更稳,且**不需要重新跑生成**——所有特征都已在 `event_summary` 里存好。

---

## 5. 改进策略 C(需重跑、但低成本):只在"answer-slot"做接受门控

第 1 节证明"生成阶段挑 token 施加引导"会掉点,但**接受门控可以反过来按 token 类型差异化**:
- 对 **content / number / option** token:用宽松门控(让 VRG 有机会改对视觉读数);
- 对 **template / punct / white** token:用严格门控甚至禁止改写(它们改了多半是噪声)。

**数据支撑**:`visual_proposal_vrg_comparison.json` 的 category_counts 显示,收益与 selected token 的 semantic_ratio 正相关(proposal 0.46→41.75% vs visual 0.27→39.25%)。把 VRG 的"接受权"集中到内容词上,等价于提高有效 semantic_ratio。这条需要在 refill 时记录被改 token 的类别(实现简单),再差异化门控。

---

## 6. 不建议继续投入的方向(已被数据证否)

| 方向 | 证据 | 结论 |
|---|---|---|
| 加大 α | α=1.0 → 0.3986 | ❌ 越大越差 |
| 换 weak-visual 来源 | null≈noise(0.4059 vs 0.4072) | ❌ 非真实视觉,换源无意义 |
| 生成阶段事前挑 token 引导 | highvisual 0.389 / highgain_lowconf 0.393 | ❌ 事前选择不可靠 |
| 置信度软/硬校准(现有实现) | softconf 0.4055 / hardconf 0.4050 | ❌ 当前校准方式无增益 |
| prompt-contrast / confused-map | 0.4055 | ❌ 不超朴素 |

---

## 7. 落地优先级

1. **(本周可做,零成本)** 用现有 `event_summary` 的 confidence_delta / rank_delta 重放,严格验证 Gate A/B 的 net 提升,确认 ~0.410 的估算 → 这是写进汇报最硬的一条。
2. **(需小改生成代码)** 把 Gate A 做成 refill 时的实时接受门控,在 held-out(或换 seed)上验证,排除 in-sample 乐观偏差。
3. **(进阶)** 策略 B 的打分接受 + 策略 C 的按 token 类别差异化门控。

---

## 附:证据文件
- 12 个 VRG 变体 acc: `M3CoT/PostVRG/outputs/{main_postvrg_*, postvrg_*, highvisual_postvrg_*, highgain_lowconf_postvrg_*, main_rulecontent_postvrg_*, *promptcontrast*}/summary.json`
- 改对/改错判别特征 + 门控重放来源: `M3CoT/PostVRG/outputs/proposal_vrg_gain_case_analysis.json`(`by_outcome` + `cases[].event_summary`)
- token 类别与 semantic_ratio: `M3CoT/PostMaSK/outputs/visual_proposal_vrg_comparison.json`
- null vs noise 控制: `main_postvrg_alpha0p5_nullvisual_*` vs `main_postvrg_alpha0p5_noise500_*`
