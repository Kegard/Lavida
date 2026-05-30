# LaViDa 模态因果分析：可行方案

## 背景与核心差异

LaViDa 是**理解模型**（VQA/captioning），不是生成模型。
这意味着因果效应的"结果变量"不是图像质量，而是**答案的正确性/质量**。

与 Consis-GCPO（图像生成）的对比：

| 维度 | Consis-GCPO | 本方案（LaViDa） |
|---|---|---|
| 任务 | R2I/R2V 生成 | VQA / captioning |
| 结果变量 Y | 图像质量（DINO/CLIP-I） | 答案质量（accuracy / CLIP-T） |
| 处理变量 | 文本 prompt P，参考图像 Ir | 文本 instruction C，视觉 token V |
| 干预手段 | SDE 中单步消融 | MDM 中单步消融（mask image/text token） |
| 结果读取 | 最终生成图像 | 最终 decode 出的文字答案 |

---

## 方案：Step-wise 模态因果干预

### Step 1：理解 LaViDa 的推理循环

LaViDa 的推理是一个离散掩码扩散过程，序列结构如下：

```
[IMG_TOKEN_1 ... IMG_TOKEN_N] [TEXT_INST_1 ... TEXT_INST_M] [MASK MASK MASK ...]
      ↑ 视觉 prefix（固定）          ↑ 文本 prefix（固定）         ↑ 待解码的响应 tokens
```

每一步 t，模型从全 MASK 出发，逐步 unmask 响应 token：

```python
# 伪代码：LaViDa 一步 decode
logits = model(input_ids, attention_mask)   # input_ids 含 image tokens + text tokens + [MASK]
# 根据 confidence 选择本步 unmask 哪些位置
scores = logits[mask_positions].max(-1).values
top_k = scores.topk(num_to_unmask)
x[top_k.indices] = logits[top_k.indices].argmax(-1)
```

**关键**：图像 token 和文本 token 都在 prefix 里，只有响应 token 被 mask/unmask。
干预的对象是"prefix 中的图像/文本部分"。

---

### Step 2：定义干预操作

参考 Consis-GCPO 的 Definition 1，在 LaViDa 中定义逐步干预：

**do(V=∅, t')**：在第 t' 步，将图像 token 替换为零向量或 [UNK] token：
```python
def intervene_vision(input_ids, image_token_mask, step):
    """在第 step 步，将 image tokens 替换为 pad token"""
    modified = input_ids.clone()
    modified[image_token_mask] = tokenizer.pad_token_id  # 或 zero embedding
    return modified
```

**do(C=∅, t')**：在第 t' 步，将文本 instruction token 替换为空/pad：
```python
def intervene_text(input_ids, text_token_mask, step):
    modified = input_ids.clone()
    modified[text_token_mask] = tokenizer.pad_token_id
    return modified
```

**注意**：只在第 t' 步做干预，该步 decode 结束后，后续步骤恢复正常条件继续跑完。
这与全程消融不同，捕获的是"第 t' 步的边际因果贡献"。

---

### Step 3：三条轨迹的生成

对每个样本 (image, question)，运行三条推理轨迹：

```python
import torch
import copy

def run_three_trajectories(model, tokenizer, image, question, num_steps=10):
    """
    返回三组最终答案 token 序列
    """
    results = {}

    # (a) 主轨迹：完整条件
    results['main'] = lavida_decode(model, image, question, steps=num_steps)

    # (b) 逐步文本干预：对每个 step t' 单独运行一次
    text_intervention_outputs = {}
    for t_prime in range(num_steps):
        out = lavida_decode_with_intervention(
            model, image, question,
            intervene_at=t_prime,
            intervene_modality='text',
            steps=num_steps
        )
        text_intervention_outputs[t_prime] = out
    results['text_interventions'] = text_intervention_outputs

    # (c) 逐步视觉干预
    vision_intervention_outputs = {}
    for t_prime in range(num_steps):
        out = lavida_decode_with_intervention(
            model, image, question,
            intervene_at=t_prime,
            intervene_modality='vision',
            steps=num_steps
        )
        vision_intervention_outputs[t_prime] = out
    results['vision_interventions'] = vision_intervention_outputs

    return results


def lavida_decode_with_intervention(model, image, question,
                                     intervene_at, intervene_modality, steps):
    """
    带单步干预的 decode loop
    """
    # 初始化：全 MASK 响应序列
    input_ids = prepare_input(image, question)  # [img_tokens | text_tokens | MASK*L]
    image_positions = get_image_positions(input_ids)
    text_positions = get_text_positions(input_ids)

    for step in range(steps, 0, -1):  # t 从高到低
        t_ratio = step / steps  # 当前 timestep 归一化到 [0,1]

        if step == intervene_at:
            # 干预：替换对应 modality 的 token
            intervened_ids = input_ids.clone()
            if intervene_modality == 'vision':
                intervened_ids[image_positions] = model.config.mask_token_id
                # 或直接用 zero embedding hook
            elif intervene_modality == 'text':
                intervened_ids[text_positions] = model.config.pad_token_id
            logits = model(intervened_ids).logits
        else:
            logits = model(input_ids).logits

        # 选择本步 unmask 的 token
        input_ids = unmask_step(input_ids, logits, t_ratio)

    return decode_response(input_ids)
```

---

### Step 4：量化因果效应

定义度量函数 ψ（对应答案质量）：

- **VQA 任务**：ψ = 答案是否包含正确关键词（0/1 或 soft match score）
- **Captioning 任务**：ψ = CLIP 文本相似度（生成文本 vs ground truth）
- **通用**：ψ = 困惑度的倒数（越自信答案越好）

```python
def compute_causal_effects(results, ground_truth, psi_fn):
    """
    计算每个 step 的瞬时因果效应 δ
    """
    R_main = psi_fn(results['main'], ground_truth)

    delta_text = {}
    delta_vision = {}

    for t_prime in results['text_interventions']:
        R_text_intervened = psi_fn(results['text_interventions'][t_prime], ground_truth)
        delta_text[t_prime] = R_main - R_text_intervened  # 越大 = 文本在该步越重要

    for t_prime in results['vision_interventions']:
        R_vision_intervened = psi_fn(results['vision_interventions'][t_prime], ground_truth)
        delta_vision[t_prime] = R_main - R_vision_intervened

    return delta_text, delta_vision


def softmax_normalize(delta_dict, tau=1.0):
    """将 δ 转化为归一化重要性权重 ω（对应论文公式 11）"""
    import numpy as np
    steps = sorted(delta_dict.keys())
    values = np.array([delta_dict[s] for s in steps])
    exp_vals = np.exp(values / tau)
    weights = exp_vals / exp_vals.sum()
    return {s: w for s, w in zip(steps, weights)}
```

---

### Step 5：汇总与可视化

```python
import numpy as np
import matplotlib.pyplot as plt

def aggregate_and_plot(dataset_results, tau=1.0):
    """
    在多个样本上取平均，绘制模态重要性随 timestep 的变化曲线
    """
    all_delta_text = []
    all_delta_vision = []

    for delta_text, delta_vision in dataset_results:
        omega_text = softmax_normalize(delta_text, tau)
        omega_vision = softmax_normalize(delta_vision, tau)

        steps = sorted(omega_text.keys())
        all_delta_text.append([omega_text[s] for s in steps])
        all_delta_vision.append([omega_vision[s] for s in steps])

    # 取均值
    mean_text = np.mean(all_delta_text, axis=0)
    mean_vision = np.mean(all_delta_vision, axis=0)
    std_text = np.std(all_delta_text, axis=0)
    std_vision = np.std(all_delta_vision, axis=0)

    # 注意：LaViDa 的 step 是从 t=1（高噪声）到 t=0（低噪声）
    # x 轴从右到左代表 denoising 方向
    x = np.array(steps) / max(steps)  # 归一化为 [0, 1]

    plt.figure(figsize=(8, 4))
    plt.plot(x, mean_text, label='Text ω(t)', color='#5B8DB8', linewidth=2)
    plt.fill_between(x, mean_text - std_text, mean_text + std_text, alpha=0.2, color='#5B8DB8')
    plt.plot(x, mean_vision, label='Vision ω(t)', color='#E07B54', linewidth=2)
    plt.fill_between(x, mean_vision - std_vision, mean_vision + std_vision, alpha=0.2, color='#E07B54')
    plt.xlabel('Timestep t (1=high noise → 0=clean)')
    plt.gca().invert_xaxis()
    plt.ylabel('Importance weight ω(t)')
    plt.title('Modality Causal Contribution over Denoising Steps (LaViDa)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lavida_causal_weights.png', dpi=150)
    plt.show()
```

---

## 关键工程细节（LaViDa 特有）

### 1. 图像 token 的干预方式

LaViDa 用 vision encoder 把图像变成连续向量，再拼接进 input_ids。
干预不能简单地替换 token id（因为图像 token 在 embedding 层），要在 embedding 之后做：

```python
# 方法A：hook embedding层，将图像位置置零
def zero_vision_hook(module, input, output):
    output[:, image_positions, :] = 0.0
    return output

handle = model.language_model.embed_tokens.register_forward_hook(zero_vision_hook)
logits = model(input_ids).logits
handle.remove()

# 方法B（更简洁）：直接传 inputs_embeds
embeds = model.get_input_embeddings()(input_ids)
embeds[:, image_positions, :] = 0.0  # 或替换为 [MASK] embedding
logits = model(inputs_embeds=embeds).logits
```

### 2. 计算量控制

T 个 step，每个 step 需要多跑 2 条轨迹（文本干预 + 视觉干预），总共是 2T+1 次 forward。
对于 T=10，一个样本需要 21 次 forward。建议：

- 分析阶段用小样本（50-100 个 VQA 样本）
- 用 `torch.no_grad()` 包住所有干预轨迹
- 批量化同一 step 的干预（stack batch 维度）

### 3. ψ 函数的选择（LaViDa 是理解任务）

| 任务 | 推荐 ψ |
|---|---|
| VQA（Yes/No） | 0/1 准确率 |
| VQA（开放） | VQA-score（soft token match） |
| Captioning | CIDEr 或 CLIP-T |
| 通用 | 答案 token 的平均 log-probability（自信度） |

**最轻量的选择**：用模型自身输出的 log-probability 作为 ψ，无需外部评测器。
这等价于测量"干预后模型对主轨迹答案的困惑度增量"。

---

## 预期发现（假设）

根据 Consis-GCPO 的"Coarse-to-Fine"规律，LaViDa 中预期：

- **早期步骤（t → 1）**：文本权重 ω_text 主导 → 文本 instruction 决定回答的大方向（是"是/否"还是"描述颜色"）
- **晚期步骤（t → 0）**：视觉权重 ω_vision 升高 → 图像细节锚定最终词汇选择（"红色"还是"蓝色"）

但 LaViDa 是理解任务，也有可能出现**视觉从始至终都重要**（因为答案直接依赖图像内容）——这本身就是一个有价值的发现，区别于生成任务的规律。

---

## 完整实验流程总结

```
1. 准备 50-100 个 VQA 样本（image + question + ground truth）
2. 对每个样本：
   a. 主轨迹 forward（T steps）
   b. 对每个 t' ∈ {1..T}：文本干预轨迹（T steps，仅第t'步改变）
   c. 对每个 t' ∈ {1..T}：视觉干预轨迹（T steps，仅第t'步改变）
3. 用 ψ 函数计算 δ_text(t'), δ_vision(t')
4. Softmax 归一化得到 ω_text(t'), ω_vision(t')
5. 在所有样本上取均值，绘制重要性曲线
6. 分任务类型（视觉推理 vs 文本推理）分析差异
```

总计算量：N_samples × (2T + 1) × T_forward ≈ 100 × 21 × 1s ≈ 35 分钟（单 GPU）
