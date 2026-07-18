# PostVRG

## 1. 代码位置

从仓库根目录看，PostVRG 相关文件位于：

```text
Lavida/
├── M3CoT/
│   ├── PostVRG/
│   │   ├── postvrg.py
│   └── utils/
│       └── metric.py
├── VRG/
│   └── timestep_vrg.py
├── llava/
├── eval/
├── pyproject.toml
└── README.md
```

## 2. 核心文件说明

`postvrg.py` 是主实现文件，包含两阶段 PostVRG 解码流程：

- draft 阶段：先按置信度逐步填充答案 token，得到初始草稿答案。
- refine/postmask 阶段：根据 draft 阶段的 proposal confidence 选择低置信度位置，固定一组需要重生成的位置，再进行 remask-refill。
- VRG/VCD guidance：在 draft 或 refill 阶段可使用强视觉前缀与弱视觉前缀的 logits 差值做 guidance。
- 结果记录：输出逐样本 `records.jsonl` 和整体 `summary.json`。

## 3. 模型和数据准备

默认参数约定如下：

```text
--dataset-path  LightChen2333/M3CoT
--split         test
--pretrained    weight/lavida-reason
--vision-tower  weight/siglip
--model-name    llava_llada
--conv-template llada
```

模型位置如下：

```text
Lavida/
└── weight/
    ├── lavida-reason/
    └── siglip/
```

数据集默认通过 Hugging Face `datasets.load_dataset("LightChen2333/M3CoT", split="test")` 加载。

## 4. 运行 PostVRG

最小 test 建议先跑 1 条样本：

```bash
python M3CoT/PostVRG/postvrg.py \
  --limit 1 \
  --output-dir M3CoT/PostVRG/outputs/smoke_test
```

默认实验参数已在 `apply_postvrg_defaults()` 中设置，包括：

```text
prompt=cot
max_new_tokens=64
block_length=64
step_ratio=0.5
sample_mode=random
sample_seed=42
draft_steps=16
postmask_steps=16
fixed_set_size=32
fixed_refill_per_step=2
refill_guidance=vcd
refill_weak_visual_mode=diffusion_noise
vcd_refill_alpha=1.0
vcd_noise_step=500
vcd_noise_seed=42
```

常用完整运行命令：

```bash
python M3CoT/PostVRG/postvrg.py \
  --limit 400 \
  --sample-mode random \
  --sample-seed 42 \
  --vcd-refill-alpha 1.0 \
  --refill-vrg-calibration none \
  --vcd-noise-step 500 \
  --output-dir M3CoT/PostVRG/outputs/postvrg_seed42_n400
```

## 5. 关键参数

数据与采样：

- `--dataset-path`：数据集路径或 Hugging Face 数据集名，默认 `LightChen2333/M3CoT`。
- `--split`：`train`、`validation` 或 `test`。
- `--limit`：运行样本数。
- `--domain-filter`：只跑某个 domain。
- `--sample-mode`：`sequential` 或 `random`。
- `--sample-seed`：随机采样种子。

模型：

- `--pretrained`：LaViDa/LLaDA 推理 checkpoint。
- `--vision-tower`：视觉塔 checkpoint。
- `--torch-dtype`：`float16`、`bfloat16` 或 `float32`。
- `--device` / `--device-map`：默认使用 `cuda` 和 `cuda:0`。

生成与两阶段解码：

- `--max-new-tokens`：答案生成长度，默认 64。
- `--block-length`：当前 `postvrg.py` 要求 `block_length == max_new_tokens`。
- `--step-ratio` 或 `--step-per-block`：控制总 denoising steps，二者不要同时设置。
- `--draft-steps`：draft 阶段步数。
- `--postmask-steps`：refine 阶段步数。
- `--fixed-set-size`：draft 后固定 remask 的答案位置数量。
- `--fixed-refill-per-step`：每个 refine step 重新填充多少个 masked 位置。

VRG/VCD：

- `--draft-guidance`：draft 阶段是否启用 VCD guidance，取值 `none` 或 `vcd`。
- `--refill-guidance`：refill 阶段是否启用 VCD guidance，取值 `none` 或 `vcd`。
- `--draft-weak-visual-mode` / `--refill-weak-visual-mode`：弱视觉条件，支持 `diffusion_noise` 和 `null_visual`。
- `--vcd-draft-alpha` / `--vcd-refill-alpha`：guidance 强度。
- `--refill-vrg-calibration`：`none`、`soft_confidence` 或 `hard_confidence`。
- `--refill-vrg-confidence-threshold`：hard confidence gate 的阈值。
- `--refill-vrg-confidence-gate-tau`：按 guided confidence 与 conditional confidence 差值做 token gate。
- `--refill-guidance-steps`：只在前 k 个 refine steps 使用 VRG。
- `--vcd-noise-step`：diffusion noise timestep，必须在 `[0, 999]`。

## 6. 输出文件

运行后输出目录通常包含：

```text
output-dir/
├── records.jsonl
└── summary.json
```

`records.jsonl` 每行是一条样本，包含：

- `dataset_index`、`id`、`question`、`choices`、`answer`、`domain`、`topic`
- `draft_text`、`draft_answer_ids`、`draft_correct`
- `final_text`、`final_answer_ids`、`final_correct`
- `draft_records`：draft 每一步填充位置和中间文本
- `postmask_records`：refine 每一步 remask/refill 位置和中间文本
- `meta`：本样本使用的解码参数与 proposal confidence 统计

`summary.json` 包含：

- 样本数、总耗时、平均耗时
- draft accuracy、final accuracy
- postmask 后 improved/worsened 样本数
- 完整 generation 配置

## 7. 参考仓库

[Kegard/Lavida](https://github.com/Kegard/Lavida)
