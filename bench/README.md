# bench：中文真实病例的图文诊断评测集

`bench/data/` 是已冻结的数据集，随仓库分发，**clone 下来就能评测**：不需要爬虫、不需要上游语料、不需要出题脚本，也不需要任何构建步骤。

- **84 道不定项选择题**：每题 1–3 个正确诊断 + 3 个干扰项（选项 A–F），配 1–4 张影像/病理/内镜/临床照片；
- **252 条图文冲突三分类题**：同一批图配三种文本（84 支持 / 84 反驳 / 84 信息不足）；
- 84 个源病例的原文（`cases.jsonl`），供 `text_a` 口径使用；
- 154 张图，27.7 MB。

数据是怎么挑出来、怎么出题的，见 [`REVIEW.md`](REVIEW.md)（100 题时期的审查档案）和本文末的"构建过程"。

---

## 目录

```bash
bench/
├── requirements.txt
├── evaluate.py         # 变体评测（items 模式）与图文冲突（conflicts 模式）
├── freeze_data.py      # 重算 / 校验 MANIFEST.json
├── llm.py              # OpenAI 兼容客户端（重试、JSON 修复、按阶段指定模型）
├── medical.py          # 模态识别、诊断解析、合并症分离、评分口径
├── data/
│   ├── items.jsonl             # 84 题
│   ├── conflict_items.jsonl    # 252 条冲突题
│   ├── cases.jsonl             # 84 个源病例的原始记录（text_a 用）
│   ├── MANIFEST.json           # 计数 + 每个文件的 sha256
│   └── images/<case>/          # 154 张题图，27.7 MB
├── RESULTS.md          # 评测报告框架（待填的表与结论）
├── REVIEW.md           # 数据审查档案（历史，见文末说明）
├── TEXT_AUDIT.md       # 逐题文本充分性审查（哪些题不看图也能答，附修复记录）
└── results/            # 评测产物（不进版本库）
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

冲突题（`--mode conflicts`）报三分类准确率与混淆矩阵（行为真值、列为预测）。

---

## 常用命令

```bash
# 全量三变体（252 次调用，约 50 分钟）
python bench/evaluate.py --variants photo,text_b,text_a

# 冒烟测试：先跑 5 题
python bench/evaluate.py --limit 5 --variants photo,text_b

# 图文冲突三分类
python bench/evaluate.py --mode conflicts

# 指定输出文件与端点
python bench/evaluate.py --variants photo,text_b \
    --output bench/results/eval_local.jsonl \
    --base-url http://localhost:8000/v1 --model Qwen/Qwen3-VL-8B-Instruct
```

`--image-root` 可以换一个图片目录（默认 `bench/data/images`），`--items` / `--conflicts` 同理。

---

## 数据维护

`data/` 是就地维护的：人工裁图、换图、删题都直接改文件，改完把清单重算一次。

```bash
python bench/freeze_data.py --dry-run   # 先看会删掉哪些图
python bench/freeze_data.py             # 重算 MANIFEST.json（哈希 + 计数）
python bench/freeze_data.py --prune     # 删掉不再被任何题引用的图
python bench/freeze_data.py --verify    # 校验：clone 下来的数据是否和清单一致
```

- `--verify` 只需要 `data/`，不需要 key；建议在服务器上每次开跑前先跑一次。
- 有题目引用的图不在 `data/images/` 时，重算清单会**直接报错退出、不写 `MANIFEST.json`**，否则清单会悄悄少一张图却照样校验通过。
- 改了数据就要把 `MANIFEST.json` 一起提交，否则别人 clone 下来校验不过。
- **`cases.jsonl` 不会自动重导**：增删题目时手工同步它；漏了的话 `freeze_data.py` 会提示有多少题缺原文，`text_a` 也会在开跑前报错。
- `results/` 是评测产物，不进版本库；`MANIFEST.json` 的 `generated_at` 每次重算都会变，属于正常现象。

---

## 已知限制

- **图上原本印着图注**：图是从期刊病例页扒下来的**图版**，上面是影像、下面印着文章原配的中文图注。图注已逐张人工裁剪（`9cc1a78`，181 张改了 90 张；自动裁剪试过一轮，因边界不精确已回退），并确认剩下的文字不含诊断答案。题库只有这一套图，不区分"裁前 / 裁后"两种条件。
- 22 张图长边 < 260 px，且源站没有更大的原图；`case_0136` 裁完只有 195×121（`iiyi-0065`）。这类图在 `photo` 条件里基本读不出征象，是数据本身的短板。
- **"图是否足以支持该诊断"没有逐张判定**：此前的自动分诊只回答"图上有没有字、是不是影像类"（模型判定，清单已删），人工确认的是图注不含答案，不是"病灶一定可见"。
- 筛选的规则层只按文本判断模态，**没有读图**；「诊断结果」里诊断数 ≥4 的病例（约 24%）不参与出题。
- 合并症过滤用的是关键词表，少数以内分泌/心血管为主诊断的病例可能被误删。
- 改写质量靠"确定性检查 + 逐诊断审计"兜底，审计只判断"能否从文本确立诊断"，不保证改写没有改动临床事实。
- 图文冲突的三类标签由配对方式决定，同一张图配不同文本是否真的构成"反驳/信息不足"，需要人工确认。
- **"只看文本能不能答对"没有逐题解决**：`TEXT_AUDIT.md` 的结论是 14/84 题文本即充分——12 题靠化验、暴露史或典型体征，1 题（`iiyi-0044`）靠题干里的 Tullio 征，1 题（`iiyi-0024`）靠免疫球蛋白定量。靠"干扰项露馅"的 27 题已改掉选项，剩下 14 题要么改题干，要么按"文本可答"子集单独报。T1 只说明信息足够、不说明容易，真实影响要跑 `text_b` 才知道。

---

## 构建过程（历史）

出题管线已经完成使命，脚本从仓库移除（筛选、去诊断化改写、逐诊断审计、干扰项精排、图文冲突配对、原图重下、图片分诊、图注裁剪一共 8 个脚本 + `.env` 模板）。最后包含它们的提交是 `e52a8b6`，需要重跑或改题时从那里取：

```bash
git show e52a8b6:bench/build_items.py > /tmp/build_items.py
```

一句话版本：从 2395 例带图病例里筛出 700 个候选，出 100 题 / 299 条冲突题，人工清理图片后删去 15 个病例的全部图片，定稿为 **85 题 / 254 条 / 155 张图**；文本审查后又删掉 `iiyi-0088`、补齐 `0106` 缺的一条冲突题，现在是 **84 题 / 252 条 / 154 张**（见 `TEXT_AUDIT.md`）。详细的检查项、盲测结果和取舍理由见 [`REVIEW.md`](REVIEW.md)，评测口径与待填的表见 [`RESULTS.md`](RESULTS.md)。
