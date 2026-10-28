# 评测结果

三个 8B 级 VLM 在冻结测试集上的逐题输出。汇总表见 [`summary.md`](summary.md)，完整分析见 [`../RESULTS.md`](../RESULTS.md)。

## 文件

```text
<model>/items.jsonl       84 题 × 3 变体 = 252 行/模型
<model>/conflicts.jsonl   252 条图文冲突/模型
summary.md                由 report_eval.py 生成的汇总，不要手改
```

`items.jsonl` 保留原始回答、`selected`、`score`、`parse_ok`、`finish_reason`、`truncated` 和 `latency_s`；`conflicts.jsonl` 保留 `predicted_label`、`correct` 和原始回答。失败的 API 调用标为 `excluded`，解析失败按协议计 0 分，不剔除。

## 汇总

| 模型 | `text_b` | `photo` | `text_a` | $\Delta\mathrm{acc}$ | 95% CI | 冲突 acc |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `qwen3-vl-8b-instruct` | 0.361 | 0.425 | 0.659 | +0.063 | [+0.008, +0.129] | 0.563 |
| `internvl3.5-8b` | 0.272 | 0.284 | 0.591 | +0.012 | [-0.058, +0.075] | 0.516 |
| `glm-4.1v-9b-thinking` | 0.327 | 0.361 | 0.595 | +0.034 | [-0.042, +0.103] | 0.516 |

$\Delta\mathrm{acc}$ 是逐题配对差值 `photo - text_b` 的均值；区间为 1000 次 percentile bootstrap。

## 运行设置

- 代码：`a6e4888`；数据：`bench/data` 的 tree `c3f2cfa4e83ffc47d4a75f9280495539c632deae`
- 推理：vLLM 0.11.0，torch 2.8.0+cu128，transformers 4.57.3，4 × A100 80GB（与其他任务共享）
- 解码：greedy，`temperature=0`，`--max-tokens 8192`，`--max-model-len 32768`
- 服务：`--gpu-memory-utilization 0.45 --max-num-seqs 1`，`limit_mm_per_prompt={"image": 4, "video": 0}`
- GLM 额外使用 `--max-image-pixels 2500000`

## 已知偏差

- GLM 图像被压到最多 250 万像素；Qwen/InternVL 使用各自原生预处理，跨模型比较需注意。
- GLM 的 `iiyi-0067` `photo` 在 8192 token 处截断，按协议计 0 分。
- Qwen 有 4 条模型输出空 `selected`，按协议计 0 分。
- `text_a` 有少数原始病例字段不完整，未剔除。
- 评测与无关训练共享 GPU，`latency_s` 不可用于横向性能比较；答案分布不受影响。
- GLM 原始 JSONL 的 `model` 字段曾为空，已按目录名补齐；其余字段未改，`summary.md` 重新生成后逐字节一致。

## 复现

```bash
python bench/report_eval.py --results-dir bench/results --out /tmp/summary.md
```

脚本还会读取 `bench/data/items.jsonl`、`bench/data/conflict_items.jsonl` 和 `bench/TEXT_AUDIT.md`，因此需要在完整仓库中运行。
