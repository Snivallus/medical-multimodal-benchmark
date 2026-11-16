# bench：中文真实病例的图文诊断评测集

`bench/data/` 是已冻结的数据集，随仓库分发，**clone 下来就能评测**：不需要爬虫、不需要上游语料、不需要出题脚本，也不需要任何构建步骤。

- **100 道不定项选择题**：每题 1–3 个正确诊断 + 3 个干扰项（选项 A–F），配 1–4 张影像/病理/内镜/临床照片；
- **300 条图文冲突三分类题**：每道诊断题对应三个图文判断（100 支持 / 100 反驳 / 100 信息不足）；
- 100 个源病例的原文（`cases.jsonl`），供 `text_a` 口径使用；
- 题图数量、总大小与文件哈希见 [`data/MANIFEST.json`](data/MANIFEST.json)。

本版保留或修订 63 道原题，另从上游病例库纳入 37 道新题。逐题取舍、视觉证据与新题来源见 [`CURATION_V2.md`](CURATION_V2.md)。[`REVIEW.md`](REVIEW.md) 和 [`TEXT_AUDIT.md`](TEXT_AUDIT.md) 记录的是旧版审查。

---

## 目录

```bash
bench/
├── requirements.txt
├── evaluate.py         # 变体评测（items 模式）与图文冲突（conflicts 模式）
├── freeze_data.py      # 重算 / 校验 MANIFEST.json
├── check_conflict_pairs.py # 核对跨病例配对、金标与来源
├── llm.py              # OpenAI 兼容客户端（重试、JSON 修复、按阶段指定模型）
├── medical.py          # 模态识别、诊断解析、合并症分离、评分口径
├── report_eval.py      # 汇总逐题结果：配对 bootstrap、符号检验、分层表
├── run_vlm_eval.py     # 本地 vLLM 起服务并批量跑三模型
├── data/
│   ├── items.jsonl             # 100 题
│   ├── conflict_items.jsonl    # 300 条冲突题
│   ├── cases.jsonl             # 100 个源病例的原始记录（text_a 用）
│   ├── MANIFEST.json           # 计数 + 每个文件的 sha256
│   └── images/<case>/          # 181 张图，约 59.3 MB
├── CURATION_V2.md      # 本版逐题取舍、来源与标注依据
├── RESULTS.md          # 旧版三模型评测报告
├── REVIEW.md           # 数据审查档案（历史，见文末说明）
├── TEXT_AUDIT.md       # 逐题文本充分性审查（哪些题不看图也能答，附修复记录）
└── results/            # 最终逐题结果与 summary.md（日志、smoke 本地）
```

---

## 快速开始

```bash
pip install -r bench/requirements.txt

export LLM_API_KEY=<your key>
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_MODEL=Qwen/Qwen3-VL-8B-Instruct

python bench/freeze_data.py --verify            # 先确认数据完整
python bench/evaluate.py --variants photo,text_b
```

任何 OpenAI 兼容端点都行（vLLM、Ollama、AI STORE、DeepSeek……），三个变量也可以写成 `bench/.env`（该文件被 git 忽略），或者用 `--base-url` / `--api-key` / `--model` 直接指定。

---

## 三种变体

同一道题、同一套提示词，**只换证据**：

| 变体 | 输入 | 用途 |
| --- | --- | --- |
| `photo` | 图片 + 去诊断化文本 | 主评测 |
| `text_b` | 只给去诊断化文本（遮图） | **主指标**的图像必要性对照 |
| `text_a` | 病例原文（保留影像结论），遮图 | proposal 字面口径的严格对照 |

`text_a` 的原文来自 `data/cases.jsonl`。**缺行时脚本直接报错退出**，不会退化成 `text_b` 再冒充一个"严格对照"。

---

## 指标口径

单题得分用部分给分 + 错选扣分：

$$\text{score}_i=\operatorname{clip}\!\left(\frac{TP_i-FP_i}{|G_i|},\,0,\,1\right)$$

$G_i$ 为正确诊断集合，$TP$ 为选中的正确项数，$FP$ 为选中的错误项数。全选得 0 分；单选题（$|G|=1$）退化为 0/1 准确率，因此

$$\Delta\mathrm{acc}_i=\mathrm{acc}_i(\text{图文})-\mathrm{acc}_i(\text{纯文本})$$

仍是准确率之差。`evaluate.py` 会打印：

- 各变体的平均分与**主要诊断命中率**（正确项里的第一个诊断是否被选中）；
- `delta_acc (photo - text_b)`，以及 `Δ>0` / `Δ=0` / `Δ<0` 的题目数。

两个纯文本条件都要报：**`text_b`（主指标）**用改写后的文本，图像成为唯一信息源，直接测"到底看没看图"；**`text_a`（严格对照）**照搬病例原文，对应 proposal 字面口径。

冲突题（`--mode conflicts`）覆盖全部 100 个诊断病例并报告三分类准确率与混淆矩阵。每个病例提供固定的临床文本和诊断或征象。支持项配同一病例的图，反驳项配能显示相反证据的近邻病例图，信息不足项配来自相关病例但缺少判别证据的图。模型只能看到文字与配图。每组三条共享相同可见文本。

---

## 常用命令

```bash
# 全量三变体（300 次调用）
python bench/evaluate.py --variants photo,text_b,text_a

# 冒烟测试：先跑 5 题
python bench/evaluate.py --limit 5 --variants photo,text_b

# 图文冲突三分类
python bench/evaluate.py --mode conflicts

# 冲突题遮图对照，默认写入独立输出文件
python bench/evaluate.py --mode conflicts --conflict-text-only

# 指定输出文件与端点
python bench/evaluate.py --variants photo,text_b \
    --output bench/results/eval_local.jsonl \
    --base-url http://localhost:8000/v1 --model Qwen/Qwen3-VL-8B-Instruct
```

`--image-root` 可以换一个图片目录（默认 `bench/data/images`），`--items` / `--conflicts` 同理。默认结果目录为 `bench/results/_v2/`。使用 `--resume` 时只接续本版输出；旧版结果位于 `bench/results/`，两版不能混算。用 `python bench/report_eval.py` 汇总本版的批量运行。复现历史汇总使用 `python bench/report_eval.py --version v1`。

---

## 数据维护

`data/` 是就地维护的：人工裁图、换图、删题都直接改文件，改完把清单重算一次。

```bash
python bench/freeze_data.py --dry-run   # 先看会删掉哪些图
python bench/freeze_data.py             # 重算 MANIFEST.json（哈希 + 计数）
python bench/freeze_data.py --prune     # 删掉不再被任何题引用的图
python bench/freeze_data.py --verify    # 校验：clone 下来的数据是否和清单一致
python bench/check_conflict_pairs.py     # 校验跨病例配对、标签均衡和来源
```

- `--verify` 只需要 `data/`，不需要 key；建议在服务器上每次开跑前先跑一次。
- 有题目引用的图不在 `data/images/` 时，重算清单会**直接报错退出、不写 `MANIFEST.json`**，否则清单会悄悄少一张图却照样校验通过。
- 改了数据就要把 `MANIFEST.json` 一起提交，否则别人 clone 下来校验不过。
- **`cases.jsonl` 不会自动重导**：增删题目时手工同步它；漏了的话 `freeze_data.py` 会提示有多少题缺原文，`text_a` 也会在开跑前报错。
- `results/` 下只入库最终逐题结果与 `summary.md`；`server.log`、`_smoke/` 等本地产物不入库，`MANIFEST.json` 的 `generated_at` 每次重算都会变，属于正常现象。

---

## 已知限制

- 本版逐题对照题干、源病例及图像，并在必要时裁掉图注、缩小待判断的诊断范围。部分病理拼图和单层影像仍限制可判断的细节。逐题理由见 [`CURATION_V2.md`](CURATION_V2.md)。
- 新题偏向有清楚结构性证据的影像，骨科题较多；题库分布不代表临床患病率。一些题即使只看文本也可能答对，图像增益须用新版 `photo` / `text_b` 重新测量。
- 源站个别病例没有独立现病史，`text_a` 使用已有字段。旧版的模型结果和置信区间只适用于旧题库；本版为 100 例试点，真实模型实测和独立人工复标待完成。

## 构建过程（历史）

出题管线已经完成使命，脚本从仓库移除（筛选、去诊断化改写、逐诊断审计、干扰项精排、图文冲突配对、原图重下、图片分诊、图注裁剪一共 8 个脚本 + `.env` 模板）。最后包含它们的提交是 `e52a8b6`，需要重跑或改题时从那里取：

```bash
git show e52a8b6:bench/build_items.py > /tmp/build_items.py
```

一句话版本：从 2395 例带图病例里筛出 700 个候选，出 100 题 / 299 条冲突题，人工清理图片后删去 15 个病例的全部图片，定稿为 **85 题 / 254 条 / 155 张图**；文本审查后又删掉 `iiyi-0088`、补齐 `0106` 缺的一条冲突题，形成上一版的 **84 题 / 252 条 / 154 张**（见 `TEXT_AUDIT.md`）。详细的检查项、盲测结果和取舍理由见 [`REVIEW.md`](REVIEW.md)，评测口径与结果见 [`RESULTS.md`](RESULTS.md)。
