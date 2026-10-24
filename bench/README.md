# bench: 图文病例诊断评测集构建管线

本目录把 `iiyi_dataset/` 清洗出的病例转成**不定项选择题**，用于 A 组的"中文真实病例多模态诊断推理对抗评测"。

`iiyi_dataset/` 被视为只读数据源：本管线只读 `results/case_with_images.json` 与 `results/images/`，不回写任何文件。

`data/` 是**已冻结的数据集**，随仓库一起分发：`items.jsonl`、`conflict_items.jsonl`、`cases.jsonl`（原始病例）、`images/`（题目用到的图片）与 `MANIFEST.json`（计数 + 每个文件的 sha256）。服务器上 clone 下来即可直接评测，不必重跑爬虫或出题；`results/` 放评测产物，不进版本库。

---

## 管线

```
results/case_with_images.json   （2395 例带图病例，只读）
        │
        ├─ 1. screen_cases.py   规则分桶 + LLM 0-3 打分 ──▶ data/candidates.jsonl
        │
        ├─ 2. build_items.py    去诊断化 → 逐诊断审计 → 干扰项精排 → 组卷
        │                                                   └─▶ data/items.jsonl
        │
        ├─ 3. build_conflicts.py 同一批图，三种图文配对 ──▶ data/conflict_items.jsonl
        │
        ├─ 4. freeze_data.py    题目引用的图片 + 原始病例 + 清单 ──▶ data/images/、data/cases.jsonl、data/MANIFEST.json
        │
        └─ 5. evaluate.py       图文 / 纯文本B / 纯文本A 三变体 ──▶ results/eval_results.jsonl
```

### 1. 筛选（`screen_cases.py`）

- **规则层**：按 CT / MRI / X 线 / 超声 / 内镜 / 病理 / 皮肤镜 / 临床照片 / 眼底 / 心电图 / 核医学 给病例分桶。高召回即可，不追求精度。
- **LLM 层**（`--use-llm`）：读病例文本，判断"最终诊断对可视化检查的依赖程度"，输出 0–3 分。
- 输出 JSONL，逐条落盘，可断点续跑。

### 2. 出题（`build_items.py`）

每个病例四步：

1. **改写**：LLM 删除一切点明诊断的表述——标题不进题干，`门诊以"…"收入院`、影像/病理/术中结论、治疗方案全部去掉；病史、查体、化验数值保留。
2. **审计**：先做确定性检查（题干里是否还残留影像报告），再按**每一个正确诊断**逐条问模型"仅凭这段文本能否确立该诊断"（A 能 / B 需影像 / C 不能）。出现 A 即判定泄漏，第二次改写升级到强模型。
3. **干扰项**：字符 n-gram 检索 20 个相近诊断 → LLM 挑 3 个"粗心读者会误选"且确实不成立的；同义、亚型、分期变体被显式排除。
4. **组卷**：正确项（1–3 个）+ 干扰项（3 个）按病例 uid 派生的种子打乱，选项标 A–F，记录正确项 ID。

入集硬性条件：诊断数 1–3、图片数 1–4、命中可视化模态、有 LLM 打分时 ≥2，
且**病史+查体至少 150 字**——否则病例内容几乎全在影像报告里，改写后题干只剩一句话，题目会退化成"看图猜病"。

出题结果分两个文件：`data/items.jsonl` 只放通过校验的题；被标记的题（文本仍泄漏诊断、题干过短、干扰项异常）进
`data/items_rejected.jsonl` 并附原因。**被拒的病例不会在下次运行里重做**，避免反复烧同样的 API 调用；
`--limit` 统计的是 `items.jsonl` 里的题数，因此可以反复运行直到补满。
`诊断结果` 里的慢性合并症（高血压、糖尿病等）不计入正确项，只记在 `comorbidities` 里——否则"选全"会把合并症也算进答案，扭曲分数。

### 3. 图文冲突版（`build_conflicts.py`）

同一批图片，三种文本配对：

| 标签 | 构造 | 检查的能力 |
| --- | --- | --- |
| 支持 | 本病例自己的文本 | 图文一致时能否确认 |
| 反驳 | 另一个病例的文本，其诊断与本病例最接近但不同（优先取检索池里的近邻） | 会不会无视影像、跟着文本走 |
| 信息不足 | 另一个系统病例的文本（相似度最低者） | 会不会对影像无关的诊断强答 |

标签由构造方式决定，不靠模型猜。**配对是组合式的，必须人工过一遍。**

评测用 `python bench/evaluate.py --mode conflicts`，报告三分类准确率与混淆矩阵。

### 4. 打包（`freeze_data.py`）

出题和配对写完后跑一次，把 `data/` 变成可独立分发、可校验的数据集：

- 把 `items.jsonl` / `conflict_items.jsonl` 引用到的图片从 `iiyi_dataset/results/images/` 复制到 `data/images/<case>/`（100 题 + 299 条冲突题共用 181 张，共 15.2 MB；全量图 4334 张 790 MB 不进仓库）；
- 导出这 100 个病例的原始记录到 `data/cases.jsonl`，审核与报告举例时不必再翻语料；
- 写 `data/MANIFEST.json`：各文件条数、字节数、sha256，以及语料的 sha256。

`--prune` 会删掉不再被引用的图片；**人工审核改过题目后必须重跑一次**，否则 `images/` 与清单会和题目对不上。

### 4b. 去图注（`strip_captions.py`）

这批图是从期刊病例页扒下来的**图版**：上面是影像，下面印着文章原配的中文图注，图注里往往直接写着诊断（"图1 患者，男，44岁，C6椎板软骨瘤…"）。不裁掉的话，`photo` 变体测的是"能不能读图上的字"，不是读片。

做法：先让视觉模型指出图注在哪条边、约占多高，再按这个比例裁掉，然后用与分诊相同的提示词重新描述裁剪后的图——**图上不再出现结论、不含正确答案、仍能看到病灶**才接受，否则往下多裁一档（最多裁到 60%）。文字直接烧在片子里的（阅片器叠加、手机翻拍）裁不掉，整张就是报告截图的也裁不了，这些留给人工。

```bash
python bench/strip_captions.py --dry-run --workers 5   # 只问图注在哪
python bench/strip_captions.py --workers 4             # 裁 + 验证 + 覆盖原图
python bench/freeze_data.py --manifest-only            # 重新计 sha256
```

原图在覆盖前备份到 `.tmp/originals/`，每次裁剪的比例与验证结果记在 `data/caption_crops.json`；要恢复可以重跑 `refresh_images.py`。

### 5. 评测（`evaluate.py`）

同一道题三种条件，只有证据变化：

| 变体 | 输入 | 用途 |
| --- | --- | --- |
| `photo` | 图片 + 去诊断化文本 | 主评测 |
| `text_b` | 只给去诊断化文本（遮图） | **主指标**的图像必要性对照 |
| `text_a` | 病例原文（保留影像结论），遮图 | proposal 字面口径的严格对照 |

输出逐题结果 JSONL（默认 `results/eval_results.jsonl`），并汇总 $\Delta\mathrm{acc}$、主要诊断命中率、$\Delta>0$ 的题目占比。

---

## 指标口径

单题得分用部分给分 + 错选扣分：

$$\text{score}_i=\operatorname{clip}\!\left(\frac{TP_i-FP_i}{|G_i|},\,0,\,1\right)$$

$G_i$ 为正确诊断集合，$TP$ 为选中的正确项数，$FP$ 为选中的错误项数。全选得 0 分；单选题（$|G|=1$）退化为 0/1 准确率，因此

$$\Delta\mathrm{acc}_i=\mathrm{acc}_i(\text{图文})-\mathrm{acc}_i(\text{纯文本})$$

仍是准确率之差。另报**主要诊断命中率**（正确项中的第一个诊断是否被选中）。

纯文本版本有两个口径，都要报：

| 口径 | 做法 | 用途 |
| --- | --- | --- |
| **B（主指标）** | 改写后的文本（已去掉影像结论与诊断名） | 图像成为唯一信息源，直接测"到底看没看图" |
| **A（严格对照）** | 病例原文照搬，只遮图 | 对应 proposal 字面口径，作为对照 |

---

## 运行

```bash
# 0. 配置密钥
cp bench/.env.example bench/.env      # 填入 AI STORE 的 key 与模型名
python bench/check_api.py          # 列出可用模型并发一条测试请求

# 1. 筛选（规则层不需要 key，LLM 层需要）
python bench/screen_cases.py                                 # 只跑规则分桶
python bench/screen_cases.py --use-llm --workers 8 --model Qwen-Flash-A

# 2. 出题：改写用强模型，审计用快模型，干扰项用强模型
#    --limit 是"文件里最终有多少题"，已有条目会被跳过，可反复运行直到补满
python bench/build_items.py --limit 100 --workers 8 \
    --model-rewrite Qwen3.8-Max --model-fast Qwen-Flash-A --model-judge Qwen3.8-Max

# 3. 图文冲突版
python bench/build_conflicts.py

# 4. 打包成自包含数据集（改了题目就要重跑）
python bench/freeze_data.py --prune

# 5. 评测（本机跑托管 VLM，或指向自己的 GPU 服务）
python bench/evaluate.py --model Qwen3.8-Max --variants photo,text_b,text_a
python bench/evaluate.py --mode conflicts --model Qwen3.8-Max     # 三分类冲突题
python bench/evaluate.py --base-url http://localhost:8000/v1 --model Qwen/Qwen3-VL-8B-Instruct
```

实测速率（AI STORE，8 并发）：筛选约 1.5 s/例，出题约 200 s/例（含两次改写与审计）。

### 在另一台机器（GPU 服务器）上直接评测

仓库已带冻结数据集，clone 后不需要爬虫、语料或出题，只要装依赖和 key：

```bash
git clone <repo> && cd <repo>
pip install -r bench/requirements.txt
cp bench/.env.example bench/.env     # 填 LLM_BASE_URL / LLM_MODEL / LLM_API_KEY（vLLM 就填本机地址）

python bench/evaluate.py --mode items --model Qwen/Qwen3-VL-8B-Instruct \
    --base-url http://localhost:8000/v1 --variants photo,text_b
```

图片默认从 `bench/data/images/` 读；想用全量图（或换了目录）就加 `--image-root`。
用 `MANIFEST.json` 核对数据与实验时是否一致（不需要语料，也不需要 key）：

```bash
python bench/freeze_data.py --verify      # 逐个文件比对 sha256
```

---

## 目录

```bash
bench/
├── .env.example        # 端点与 key 模板（.env 本身被 git 忽略）
├── requirements.txt    # 评测/出题所需的最小依赖
├── check_api.py        # 连通性自检 + 列出模型
├── llm.py              # OpenAI 兼容客户端（.env 读取、重试、JSON 修复、按阶段指定模型）
├── medical.py          # 模态识别、诊断解析、合并症分离、影像擦除、评分口径
├── retrieval.py        # 诊断近邻检索与同义归一
├── screen_cases.py     # 步骤 1：筛选
├── build_items.py      # 步骤 2：出题
├── build_conflicts.py  # 步骤 3：图文冲突配对
├── freeze_data.py      # 步骤 4：打包（图片 + 原始病例 + 清单）
├── refresh_images.py   # 从病例页重下原图（旧爬虫存的是 150 px 缩略图）
├── triage_images.py    # 逐图分诊：图里到底是什么、有没有把结论印在图上
├── strip_captions.py   # 步骤 4b：裁掉印在图版下方的期刊图注
├── evaluate.py         # 步骤 5：三变体评测
├── IMAGE_TRIAGE.md     # 分诊产出的人工复核清单（A/B/C 三档）
├── data/               # 冻结数据集（随仓库分发）
│   ├── items.jsonl             # 100 题
│   ├── conflict_items.jsonl    # 299 条三分类冲突题
│   ├── cases.jsonl             # 100 个源病例的原始记录
│   ├── candidates.jsonl        # 筛选结果（700 例，留档）
│   ├── items_rejected.jsonl    # 被拒题目及原因（留档）
│   ├── MANIFEST.json           # 计数与 sha256
│   ├── caption_crops.json      # 每张图裁掉图注的比例与验证结果
│   └── images/<case>/          # 181 张题图（全尺寸、已去图注，28.9 MB）
└── results/            # 评测产物（不进版本库）
```

---

## 已知限制

- 规则层只按文本判断模态，**没有读图**；图片里是否真有决定性病灶，最终要靠人工审核。
- **一半的图上曾经写着答案**：分诊（`IMAGE_TRIAGE.md`）发现 83 张图带期刊图注、71 张的文字就是检查结论，50 道题受影响。`strip_captions.py` 已裁掉 61 张图的图注并逐张复验，**题 74 A / 14 B / 12 C**。剩下 12 道：6 道是文字直接烧在片子上或整张就是报告，裁不掉；6 道因 API 配额耗尽没裁完（`caption_crops.json` 里状态是 `error`）。
- 裁图注是"模型定位 + 模型复验"的启发式，不是人眼逐张确认；61 张里我抽查了 `case_0295`、`case_0314` 两张，影像面板完整、图注确实没了。原图在 `.tmp/originals/`，可用 `refresh_images.py` 恢复。
- `诊断结果` 里诊断数 ≥4 的病例（约 24%）不参与出题：每个诊断都得是正确项，题面会变得不可答。
- 合并症过滤用的是关键词表，少数以内分泌/心血管为主诊断的病例可能被误删。
- 改写质量靠"确定性检查 + 逐诊断审计"兜底，审计只判断"能否从文本确立诊断"，不保证改写没有改动临床事实，正式入集前仍需人工过一遍。
- 图文冲突的三类标签由配对方式决定，同一张图配不同文本是否真的构成"反驳/信息不足"，需要人工确认。
