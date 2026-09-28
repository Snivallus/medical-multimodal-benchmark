# Landscape Survey: Is a "Multimodal Chinese Medical MCQ Dataset from Online Case Reports" Novel and Defensible?

**Survey date:** ~28 September 2026 · **Method:** web search + direct fetch of arXiv abs pages, ACL Anthology, MICCAI open access, NeurIPS proceedings, Nature/JoVE records, GitHub/HF raw files.

**Evidence policy:** every number below was read off a fetched primary source. Anything I could not verify is marked **[UNVERIFIED]**. Several 2026 arXiv IDs look implausible (2604.*, 2605.*, 2606.*) but I fetched each abs page and they resolve — they are real in this timeline.

---

## 0. Executive verdict (short)

| Question | Answer |
|---|---|
| Is a Chinese multimodal medical MCQ dataset from online case reports **already done**? | **Not as a peer-reviewed benchmark.** Chinese multimodal medical MCQ benchmarks exist (TCM-Ladder, MedBench v4, OphthalWeChat) but are built from **textbooks, exams, WeChat public accounts, and video** — not from community case-sharing forums. |
| Is the field **saturated** for *text-only* Chinese medical MCQ? | **Yes, essentially.** Frontier models are at 78–91% on CMExam / CMB-Exam / MedQA-CN / CNMLEQA. |
| Is it saturated for *multimodal* Chinese medical? | **No — but the MCQ framing is the weak part.** The live frontier is adversarial / evidence-grounded / multi-image / unanswerable, and almost all of it is **English (MIMIC-CXR, CheXpert, SLAKE-En)**. |
| **Opening?** | **Yes, narrow but real.** See §6. The defensible gap is *not* "more MCQs" — it is **Chinese, real-world heterogeneous patient-photo case material + verified image-dependence + adversarial/conflict controls.** |

---

## 1. Chinese medical **text-only** exam MCQ datasets

### 1.1 Foundations (2021–2024)

- **CMExam** — Liu Junling et al. (Shanghai Jiao Tong / HKUST / MIT). **NeurIPS 2023 Datasets & Benchmarks.** Sourced from the **Chinese National Medical Licensing Examination (NMLE)**. **60K+ MCQs** plus solution explanations. Five expert annotations: disease group, clinical department, medical discipline, competency area, difficulty. GPT-4 = **61.6%** accuracy, weighted F1 0.617; **human = 71.6%**. → https://arxiv.org/abs/2306.03030 · https://papers.nips.cc/paper/2023/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html
- **CMB / CMB-Exam / CMB-Clin** — Wang Xidong, Chen Guiming Hardy, Song Dingjie, … Benyou Wang, Haizhou Li (**CUHK-Shenzhen, FreedomIntelligence**). **NAACL 2024.** Chinese-native, TCM-integrated. **CMB-Exam = 280,839 questions** across **6 categories / 28 subcategories / 176 subjects** (Physician 124,926; Nurse 16,919; Technicians 27,004; Pharmacist 33,354; Undergraduate 62,271; Graduate entrance 16,365). Split **269,359 train / 280 val / 11,200 test** (400 per subcategory). Data source: publicly available mock exams, course practice, "common misconception" questions from the **Chinese Medical Question Database**, used with official authorization. **CMB-Clin = 74 textbook cases** with free-text multi-turn QA (diagnosis, differential, treatment principles). → https://arxiv.org/abs/2308.08833 · dataset card: https://raw.githubusercontent.com/openmedlab/Awesome-Medical-Dataset/refs/heads/main/resources/CMB.md · https://github.com/FreedomIntelligence/CMB
- **MedBench** — Cai Yan et al. (Shanghai AI Lab / SJTU). **AAAI-24.** **40,041 questions** from four components: Chinese Medical Licensing Examination, Resident Standardization Training Examination, Doctor In-Charge Qualification Examination, and real-world clinic cases (exams, diagnoses, treatments). → https://arxiv.org/abs/2312.12806
- **Zhongjing** — Yang Songhua et al. **AAAI 2024.** Chinese medical LLM trained on **real-world multi-turn online consultation dialogues** with expert feedback. This is the key prior art for *"Chinese online medical dialogue"* as a source — but it is **text-only, dialogue, not case-report images, and not MCQ**. → https://arxiv.org/abs/2308.03549 · https://ojs.aaai.org/index.php/AAAI/article/download/29907/31586
- **MLEC-QA** — Li Jing et al. **EMNLP 2021.** Chinese **multi-choice biomedical QA** from the National Medical Licensing Examination. Earlier and smaller than CMExam; establishes the "Chinese NMLE as MCQ source" line. → https://aclanthology.org/2021.emnlp-main.698.pdf
- **MedQA / MedQA-CN** — Jin Di et al. USMLE-derived; the **CN** variant is the standard Chinese text-only comparison. Human-expert-level USMLE; used as a 4-way MCQ benchmark.
- **MedOdyssey** (Findings of NAACL 2025) — medical **long-context** benchmark, text-only. → https://aclanthology.org/2025.findings-naacl.3.pdf
- **Huatuo-26M** — Li Jianquan et al. **26M+ Chinese medical QA pairs**, the largest Chinese medical QA corpus. Composition: online medical encyclopedia **364,420**; medical knowledge graph **798,444**; **public online medical Q&A forums 32,708,346**; test set 6,000. Note the **forum-derived component is exactly "Chinese online medical Q&A"** — but it is **text-only, no images, no MCQ**. → https://arxiv.org/abs/2305.01526 · https://huggingface.co/datasets/FreedomIntelligence/huatuo_consultation_qa

### 1.2 The 2025–2026 successors — and the saturation signal

- **CNMLEQA** — Zong Hui, Cha Jiaxue, Wang Jiao (**West China Hospital, Sichuan University**). **Scientific Data, 17 April 2026.** Chinese NMLE benchmark integrating QA pairs from **three sources: PubMed, GitHub, and MedExamLLM**. Two subsets: **CNMLEQA-10k (9,890 questions)** and **CNMLEQA-3k (2,949 questions)**; **five options, one correct answer**. Annotated by clinical experts on (1) question type (knowledge-based vs **case-based**), (2) exam year, (3) five clinical scenario dimensions (disease/diagnosis, surgery, medication, laboratory examination, symptom/sign). Results: **Qwen2.5-32B = 90.88% on CNMLEQA-10k; DeepSeek-R1 = 91.59% on CNMLEQA-3k.** → https://www.nature.com/articles/s41597-026-07261-9 · https://pubmed.ncbi.nlm.nih.gov/41998016/
- **CAMEC** — Wu Yukang et al. **ACL 2026 (Long Papers).** Not a dataset but a *method paper*, and it is the clearest saturation evidence: **78.86% (CMExam), 84.15% (MedQA-CN), 78.51% (CMMLU-Med), 74.40% (CMB-exam)**, with "consistent absolute improvements over the previous state-of-the-art HuatuoGPT-o1-7B across all benchmarks." When four independent Chinese medical MCQ sets are all above ~74%, headroom for *new text MCQ sets* is thin. → https://aclanthology.org/2026.acl-long.1781/
- **TCM national licensing exam** — three LLMs evaluated on TCM NMLE (Scientific Reports 2026). → https://www.nature.com/articles/s41598-026-49200-z **[paywalled/blocked; title+venue verified via search snippet only, numbers UNVERIFIED]**

**Takeaway:** the Chinese text-only medical MCQ axis is mature, crowded, and approaching ceiling. A new text MCQ set is not defensible.

---

## 2. Medical **multimodal** benchmarks and VQA datasets

### 2.1 Classic English Med-VQA (small, single-image, mostly open-ended)

| Dataset | Images | QA pairs | Language | Task | Source |
|---|---|---|---|---|---|
| **VQA-RAD** | **315** | **~3.5K** | EN | vision-only VQA (open + closed) | Lau et al. 2018, clinician-authored on radiology images |
| **SLAKE** | **642** | **14,028** (train 9,834 / val 2,098 / test 2,094) | **Bilingual EN + ZH** | knowledge-based **and** vision-only VQA; has KG, detection boxes, masks | Liu Bo, Zhan Li-Ming, … Wu Xiao-Ming, **PolyU ISBI 2021**; images from MSD, CHAOS, ChestX-ray8; questions by clinical physicians |
| **PathVQA** | ~5K | ~32.8K | EN | open VQA on pathology | He et al. 2020, from pathology textbooks |

SLAKE is the **single most important precedent** for "Chinese medical multimodal": it is bilingual, but it is **only 642 images**, images come from **existing public imaging datasets** (not from Chinese case reports), and it is **not MCQ** (answers are open/closed words). → https://raw.githubusercontent.com/openmedlab/Awesome-Medical-Dataset/refs/heads/main/resources/SLAKE.md · https://www.med-vqa.com/slake/

### 2.2 Large-scale general medical multimodal benchmarks

- **GMAI-MMBench** — Chen Pengcheng et al. (Shanghai AI Lab / OpenGVLab + 20+ institutions). **NeurIPS 2024 D&B.** Constructed from **284 datasets**, **38 medical image modalities**, **18 clinical tasks**, **18 departments**, **4 perceptual granularities**, in **VQA format** with a customizable lexical tree. **50 LVLMs evaluated; GPT-4o only 53.96%.** Identifies **five key insufficiencies** in frontier LVLMs. **Primarily English** (assembled from existing English/public datasets). → https://arxiv.org/abs/2408.03361 · https://github.com/uni-medical/GMAI-MMBench
- **MedXpertQA** — Zuo Yuxin et al. (Tsinghua C3I). **ICML 2025.** **4,460 questions**, 17 specialties, 11 body systems. Subsets **Text** and **MM**. The MM subset deliberately introduces **expert-level exam questions with diverse images plus rich clinical information (patient records, examination results)**, explicitly positioned *against* "traditional medical multimodal benchmarks with simple QA pairs generated from image captions." Uses **data synthesis to mitigate leakage** plus **multiple rounds of expert review**. → https://arxiv.org/abs/2501.18362 · https://github.com/TsinghuaC3I/MedXpertQA
- **OmniMedVQA** — large aggregated medical VQA set; also used as a mirror in contamination audits. → https://huggingface.co/datasets/mtybilly/OmniMedVQA

### 2.3 **Chinese** multimodal medical benchmarks (the most load-bearing section for the decision)

- **TCM-Ladder** — Xie Jiacheng, Yu Yang, … Dong Xu. **NeurIPS 2025 Datasets & Benchmarks.** "**The first comprehensive multimodal QA dataset specifically designed for evaluating large TCM language models.**" Covers fundamental theory, diagnostics, herbal formulas, internal medicine, surgery, pharmacognosy, pediatrics. **Modalities: text + images + video** (video recorded by faculty from the **Department of Acupuncture and Tuina, Shanghai University of Traditional Chinese Medicine**). **>52,000 questions**, mixing **single-choice, multiple-choice, fill-in-the-blank, diagnostic dialogue, and visual comprehension**. Built with **automated + manual filtering**. Proposes **Ladder-Score** (terminology usage + semantic expression). Benchmarks 9 general + 5 TCM-specific LLMs. **This is the closest existing thing to "Chinese multimodal medical MCQ" — and it is TCM/textbook/video, not case-report-derived.** → https://arxiv.org/abs/2505.24063 · https://proceedings.neurips.cc/paper_files/paper/2025/hash/cbdd26a4490f5a8dad0d732508714029-Abstract-Datasets_and_Benchmarks_Track.html · https://tcmladder.com
- **MedBench v4** — Ding Jinru et al. (Shanghai AI Lab + **500+ institutions**). **Nov 2025.** **>700,000 expert-curated tasks**, 24 primary / 91 secondary specialties, with **dedicated tracks for LLMs, multimodal models, and agents**. Multi-stage refinement and multi-round clinician review. **15 frontier models evaluated.** Base LLMs mean **54.1/100** (best Claude Sonnet 4.5, 62.5/100); **safety & ethics only 18.4/100**. **Multimodal models worse: mean 47.5/100 (best GPT-5, 54.9/100), "solid perception yet weaker cross-modal reasoning."** Agents on same backbones reach mean **79.8/100**. This **is** a Chinese multimodal medical benchmark — but it is a **closed, hosted, expert-curated evaluation platform**, not an open crawled dataset, and it is not case-report-derived MCQ. → https://arxiv.org/abs/2511.14439 · https://ieeexplore.ieee.org/document/10778138
- **OphthalWeChat** — Xu Pusheng, … Shi Danli (**PolyU / HKU**). 2025. **Bilingual (CN + EN) ophthalmic VQA.** Images **and captions crawled from WeChat Official Accounts, 1 Jan 2016 – 31 Dec 2024**; **QA pairs generated from those captions using GPT-4o-mini**. Final: **3,469 images, 30,120 QA pairs**, 9 ophthalmic subspecialties, **548 conditions, 29 imaging modalities, 68 modality combinations**. Six subsets: **Binary_CN/EN, Single-choice_CN/EN, Open-ended_CN/EN**. Results (max overall 0.548): Gemini 2.0 Flash **0.548** overall (CN 0.546 / EN 0.550), Single-choice_CN **0.666**, Single-choice_EN 0.646; GPT-4o 0.522; Qwen2.5-VL-72B 0.514. **This is the single most relevant precedent for the proposed project:** it proves (a) Chinese social-media medical images are a viable source, (b) LLM-generated QA from captions is an accepted pipeline, (c) **single-choice MCQ** on Chinese medical images is already published. Its limitations — **one specialty only (ophthalmology), single-modality posts, WeChat Official Accounts (professional publishers, not peer case discussion), caption-derived rather than case-reasoning-derived** — are exactly where a new project could differentiate. → https://arxiv.org/abs/2505.19624 · https://www.sciencedirect.com/science/article/pii/S2667376225000538
- **RJUA-QADatasets** (Ant Group + **Renji Hospital Urology**) — Chinese urology QA, **2,129 items** (train 1705 / valid 211 / test 213), constructed from **virtual patients rewritten by doctors from real clinical experience**, contexts from Chinese urology guidelines, answers generated by LLM then annotated by urology experts. Text-centric with structured clinical context. → http://data.openkg.cn/dataset/dd526b6e-fa9f-419c-9503-4fe89d0f0006
- **RJUA-MedDQA** — Ant Group: "**A Multimodal Benchmark for Medical Document Question Answering and Clinical Reasoning**." **2,000 images: 402 screenshots, 619 scanned PDFs, remainder other document types**; tasks include entity/relation extraction with question and answer templates. **This is Chinese *medical document* (lab reports, discharge summaries) multimodal QA — not patient clinical photos, not case reports.** → https://dl.acm.org/doi/pdf/10.1145/3637528.3671644
- **WoundcareVQA** — Yim Wen-wai et al. **Journal of Biomedical Informatics, Aug 2025.** Bilingual/multilingual wound-care VQA: **477 patients, 768 images**, expert-annotated, collected from an **online telemedicine platform** (real patient queries to licensed physicians). Multimodal multi-lingual. GPT-4o metadata classification accuracy 0.776 (wound type) but tissue color only **0.296**; open-source: https://osf.io/xsj5u/ — **evidence that online/telemedicine patient images + expert annotation is a live, publishable pipeline.** → https://www.ebiotech.com/NEWSF/2025-8/20250831002044655.htm
- **OMNI (Oral and Maxillofacial Natural Images)** — Xue Pujun et al. **2025.** ⚠️ **Name collision warning:** "OMNI" in the medical literature is **this dental malocclusion dataset** — **4,166 multi-view images, 384 participants**, annotated by professional dentists, used for **classification/diagnosis**, not MCQ. It is **not** a general Chinese medical multimodal benchmark. → https://arxiv.org/abs/2505.15637 · https://github.com/RoundFaceJ/OMNI

### 2.4 The 2026 "hard / adversarial" frontier — all English, all chest-X-ray-centric

- **Med-R2 Bench** — Ma Wen, Niu Fucheng, … Zuozhu Liu (Zhejiang). **May 2026.** "**An Adversarial Benchmark for Evidence-Grounded Reasoning in Medical VLMs.**" Hierarchical, aligned to the clinical workflow: stepwise QA across **four clinical stages**, plus **adversarial perturbations to test robustness against misleading cues**. **42,432 images, 31 task categories, 110,406 QA pairs.** 14 VLMs: sequential performance degradation along the four stages; **"models rely heavily on correct prompts to guess answers"**; even with explicit visual cues, models **fail to align textual descriptions**. Stepwise fine-tuning improves robustness. → https://arxiv.org/abs/2605.24492
- **CXR-ContraBench** — Fang Zhengru et al. **May 2026.** Benchmarks **negated-option attraction** — a model answering "No consolidation" when consolidation is visible. Protocols over **ReXVQA (internal), OpenI and CheXpert (external)**. On a strict direct presence probe: **MedGemma 31.49%, Qwen2.5-VL 30.21%**; on a **135,754-record CheXpert training-split protocol both models select negated options on over 62% of presence questions.** CoT reduces some presence-side reversals but **amplifies absence-side contradictions**. Their QCCV-Neg verifier raises accuracy to **96.60% / 95.32%** without retraining. **This is the clearest published statement that "standard accuracy can hide a clinically meaningful inference-time polarity failure."** → https://arxiv.org/abs/2605.05810 · https://github.com/fangzr/cxr-contrabench-code
- **MMRareBench** — Ning Junzhi et al. (Shanghai AI Lab). **Apr 2026.** **1,756 QA pairs with 7,958 associated medical images curated from PMC case reports**, with **Orphanet-anchored ontology alignment, track-specific leakage control, evidence-grounded annotations, two-level evaluation.** Four workflow-aligned tracks: diagnosis, treatment planning, **cross-image evidence alignment**, examination suggestion. 23 MLLMs: **universally low treatment-planning performance**; medical-domain models **trail general-purpose MLLMs substantially on multi-image tracks** despite competitive diagnostic scores ("**capacity dilution effect**"). **Note: this is PMC case reports — the English analogue of the proposed idea, and it is already done.** → https://arxiv.org/abs/2604.10755
- **MedThinkVQA** — Yao Zonghai et al. (UMass). **ICLR 2026.** Expert-annotated multi-image diagnostic thinking benchmark. **8,067 cases** (720 test), **average 6.62 images per case**, vs "at most 1.43 images per case" in prior expert benchmarks. Best models: Claude-4.6-Opus 57.2%, Gemini-3-Pro 55.3%, GPT-5.2-xhigh 54.9%, GPT-5-mini 39.7%, GPT-5-nano 30.8%; best open Qwen3.5-397B 52.2%. **>70% of errors arise from image reading and cross-view integration.** Longer reasoning helps "only when visual grounding is already reliable." → https://arxiv.org/abs/2604.16506 · https://huggingface.co/datasets/bio-nlp-umass/MedThinkVQA
- **MedSG-Bench** — Yue Jingkun et al. (SJTU / Shanghai AI Lab). **NeurIPS 2025.** "**The first benchmark tailored for Medical Image Sequences Grounding.**" Eight VQA-style tasks in two paradigms: **Image Difference Grounding** (change regions) and **Image Consistency Grounding** (shared semantics). **76 public datasets, 10 imaging modalities, 9,630 QA pairs.** Companion instruction-tuning set **MedSG-188K** and model **MedSeq-Grounder**. Finds even Qwen2.5-VL and HuatuoGPT-vision have substantial limitations. → https://arxiv.org/abs/2505.11852 · https://huggingface.co/MedSG-Bench
- **MEDSYN** — Chen Boqi, Liu Xudong, … Qiu Jianing. **Findings of ACL 2026.** **Multilingual, multimodal benchmark of highly complex clinical cases with up to 7 distinct visual clinical evidence (CE) types per case.** Evaluates 18 MLLMs on **differential diagnosis (DDx) generation and final diagnosis (FDx) selection**. Frontier models match/beat experts on DDx but show a much larger **DDx–FDx gap** than clinicians. Ablations attribute failure to **(i) overreliance on less discriminative textual CE (e.g., medical history) and (ii) a cross-modal CE utilization gap**; introduces **Evidence Sensitivity** to quantify it. Explicitly multilingual (a Chinese component is plausible but **the exact language mix is UNVERIFIED**). → https://aclanthology.org/2026.findings-acl.1183/ · https://github.com/jianing-lab/MEDSYN
- **PerMed-MM** — appears to be a multimodal benchmark with a deliberate **text-only ablation** ("it is natural to ask whether models can answer questions from text alone"). → https://aclanthology.org/anthology-files/pdf/ijcnlp/2025.ijcnlp-short.21.pdf **[details UNVERIFIED beyond this snippet]**
- **MedVQA-GI / ImageCLEFmedical 2025** — third edition of the MedVQA challenge at ImageCLEF; **VQA and synthetic image generation for the gastrointestinal tract**, GI endoscopy images, clinician-authored. → https://www.dei.unipd.it/~faggioli/temp/clef2025/paper_171.pdf · https://arxiv.org/abs/2507.21165
- **MedRepBench**, **MedImageEdu**, **OmniBrainBench** (CVPR 2026), **OralGPT-Omni / MMoral-Uni** (CVPR 2026) — further 2026 multimodal medical benchmarks confirming the field is dense and rapidly expanding. **[Only names/venues verified via search snippets; numbers UNVERIFIED]**

---

## 3. **Has anyone built a multimodal medical MCQ dataset from CHINESE ONLINE CASE REPORTS (病例/病案分享, iiyi.com / dxy.cn / 丁香园)?**

**Short answer: not as a published, peer-reviewed benchmark — but the source is not untouched, and one raw community artifact exists.**

### 3.1 What I found

| Candidate | Status |
|---|---|
| **`SnivellusSnape/iiyi_dataset`** on HuggingFace (owner display name: **Cuiyang Yong**) | **Exists.** The repo name `iiyi_dataset` matches **iiyi.com (爱爱医)**, the Chinese clinical case-sharing site. The README and images were uploaded across commits (one commit titled "Upload Images", remote file size 163 kB). **I could NOT read the README contents or verify any counts** — every direct fetch of huggingface.co, the raw URL, the HF API, and huggingface.ac.cn failed (network/redirect failures), and `r.jina.ai` proxy likewise failed. **Therefore I can state only: a community HF dataset named after iiyi.com exists. I cannot verify its size, modality coverage, MCQ format, quality control, licensing, or whether it is used by any paper. Do not cite it as prior art without checking it directly in a browser.** Links: https://huggingface.co/datasets/SnivellusSnape/iiyi_dataset · https://huggingface.co/SnivellusSnape |
| **Huatuo-26M** (FreedomIntelligence) | Includes **32.7M QA pairs crawled from public Chinese online medical Q&A forums** — proving forum crawling is established practice. But **text-only, no images, open QA not MCQ, and forum Q&A rather than case reports.** |
| **Zhongjing** (AAAI 2024) | Built on **real-world multi-turn Chinese online consultation** data. **Text-only.** |
| **OphthalWeChat** | Chinese **WeChat Official Account** posts + GPT-4o-mini-generated single-choice QA, **3,469 images**. Closest analogue, but **publisher-authored content, one specialty**, not case discussion. |
| **RJUA-QADatasets / RJUA-MedDQA** | Chinese clinical content, but **virtual patients rewritten by doctors** and **medical documents (screenshots/PDFs)** — not community case reports. |
| **WoundcareVQA** | Real **online telemedicine** patient queries + images, expert-annotated, **768 images**. Online-sourced patient images are publishable — but it is **wound care, multilingual (not Chinese-focused), not MCQ-only**. |
| **MMRareBench** | **PMC case reports** → 1,756 QA pairs, 7,958 images. **English.** The English version of the idea is done; the Chinese version is not. |

### 3.2 Skeptical reading

- **No paper I could verify** builds a multimodal Chinese medical MCQ benchmark from iiyi.com / dxy.cn / 丁香园 case threads. I searched in both English and Chinese (`iiyi dataset`, `爱爱医 病例 图片 数据集`, `丁香园 数据集`, `dxy.cn dataset`, `病例 多模态 问答 数据集`, `临床病例 图像 问答`) and the only hit pointing at iiyi is the unverifiable HF repo above.
- **Why the gap plausibly exists** (structural, not accidental): case-sharing forum images are **patient photographs / phone-camera clinical photos with faces, tattoos, identifying marks, and Chinese-language PII in the thread text**; they have **no ground-truth label beyond the thread's eventual consensus diagnosis** (often contested, sometimes never resolved); threads are **long, multi-author, and adversarially noisy** (differential debate, off-topic replies, guess-then-correct); and **image rights are unclear**. WeChat Official Accounts (OphthalWeChat) sidestep all of this because a publisher already curated the case. **This is the real reason the source is underused — and it is both the risk and the moat for a new project.**
- **Consequence for the proposal:** "nobody used iiyi.com" is **weak novelty** and a reviewer will ask about legality and label validity. "nobody built a *verified, image-dependent, adversarially controlled* Chinese case benchmark" is **strong novelty**.

---

## 4. Documented failure modes of VLMs on medical multimodal tasks

This is the strongest and best-quantified part of the literature. Note the near-total **English/MIMIC-CXR** concentration.

### 4.1 Text-modality dominance / language prior over the image

- **Restrepo, Ktena, Vakalopoulou, Christodoulidis, Ferrante — "On the Risk of Misleading Reports: Diagnosing Textual Biases in Multimodal Clinical AI."** **MICCAI 2025** (1st Workshop on MLLMs in Clinical Practice). Introduces **Selective Modality Shifting (SMS)**: **systematically swapping images or text between samples with opposing labels** to expose modality-specific bias. Six open-source VLMs (four generalist, two medically fine-tuned) on **MIMIC-CXR (chest X-ray)** and **FairVLMed (scanning laser ophthalmoscopy)**. Result: **"a marked dependency on text input, which persists despite the presence of complementary visual information."** Attention analysis **confirms image content is often overshadowed by text details.** → https://arxiv.org/abs/2508.00171
- **Same group — "Medical Context Distorts Decisions in Clinical Vision Language Models."** **May 2026.** Names **three failure modes: (1) modality over-reliance on text over images, (2) spurious reliance on irrelevant clinical history, (3) prompt sensitivity across semantically equivalent inputs.** On MIMIC-CXR, by manipulating image-text alignment, clinical history, and prompt formulation: **"VLM decisions are dominated by the text modality, even when visual evidence is available"**; **"VLMs are heavily influenced by irrelevant reports, while minor prompt changes can reverse correct image-based predictions."** → https://arxiv.org/abs/2605.17436
- **MEDSYN** (ACL Findings 2026): ablations attribute DDx–FDx failure to **"overreliance on less discriminative textual CE (e.g., medical history)"** plus a **cross-modal CE utilization gap**. → https://aclanthology.org/2026.findings-acl.1183/
- **Med-R2** (2026): **"models rely heavily on correct prompts to guess answers"**; even given explicit visual cues they **fail to align textual descriptions**. → https://arxiv.org/abs/2605.24492

### 4.2 Med-BiasX (MICCAI 2025) — language bias as a *measured, fixable* quantity

Zhu Huanjia, Liu Yishu, Zhou Chengju, Lu Guangming, Chen Bingzhi. **MICCAI 2025**, LNCS 15973, pp. 369–378. Diagnosis: medical language bias stems from (a) **imbalanced data distribution** and (b) **question shortcut dependence**. Method: **Energy-aware Confidence Constraint (ECC)** + **Distribution-aware Dependence Calibration (DDC)**. Critically for benchmark design, they **construct two new bias-sensitive datasets, `SLAKE-BIAS` and `VQA-RAD-BIAS`, using controlled out-of-distribution sampling.** Reviewer criticism is instructive: reviewer 1 notes **"imbalanced data distribution" here refers to question types (how/what) and their answers**, and that a "question-only" branch is enough to expose the shortcut. → https://papers.miccai.org/miccai-2025/0538-Paper5135.html · https://doi.org/10.1007/978-3-032-05185-1_36
- **CEDO** (follow-up, IJCAI 2025 style) — "Cause-Effect Driven Optimization for Robust Medical VQA with Language Biases." Builds **SLAKE-CP and VQA-RAD-CP**, changing the **question-type/answer distribution between train and test** (VQA-CP style). Illustrative numbers on **SLAKE-CP**: SAN (CVPR'16) All 26.02 / Open 48.30 / **Closed 6.42**; UpDn 31.45 / 59.90 / **6.42**; RUBi 33.88 / 60.30 / 10.64. On **VQA-RAD-CP**, RUBi reaches All 81.27 but **Open 60.62** — i.e. **high aggregate scores coexisting with near-chance closed-form performance**, the signature of a language-prior shortcut. → https://arxiv.org/abs/2506.17903
- Related: **"A Causal Approach to Mitigate Modality Preference Bias in Medical Visual Question Answering"** → https://dl.acm.org/doi/pdf/10.1145/3689096.3689459

### 4.3 OCR-mediated modality dominance / reading text burned into the image

- **"OCR-Mediated Modality Dominance in Vision-Language Models: Implications for Radiology AI Trustworthiness."** **medRxiv, 2026.** Reports OCR-mediated dominance, including behavior under **"stealth injection"** prompting with residual vulnerability. ⚠️ **I received HTTP 403 from medRxiv on fetch; the title, venue, and the "modality dominance"/"stealth injection" framing are from search snippets only — numbers UNVERIFIED.** → https://www.medrxiv.org/content/10.64898/2026.02.22.26346828v1.full.pdf
- **Design implication (mine, not a quoted finding):** any medical MCQ dataset built from web/forum images **must** report an explicit OCR/annotation-overlay control — radiology screenshots routinely carry patient name, date, laterality markers, and burned-in "R"/"L" labels that leak the answer.

### 4.4 Text-only / image-ablated answering

- **MedBench v4** (2025) provides the cleanest recent Chinese evidence of a **perception–reasoning gap**: multimodal models score **mean 47.5/100 vs 51.4+ for base LLMs** overall, with "solid perception yet weaker cross-modal reasoning." → https://arxiv.org/abs/2511.14439
- **PerMed-MM** explicitly frames the image-ablation question: *"it is natural to ask whether models can answer questions from text alone."* → https://aclanthology.org/anthology-files/pdf/ijcnlp/2025.ijcnlp-short.21.pdf **[UNVERIFIED beyond snippet]**
- **MedThinkVQA** (ICLR 2026) supplies the mechanism: **>70% of errors arise from image reading and cross-view integration**, and **"when early evidence extraction is weak, longer reasoning yields limited or unstable gains and can amplify misread cues."** → https://arxiv.org/abs/2604.16506
- **MMRareBench** (2026): **"capacity dilution effect"** — medical fine-tuning narrows the diagnostic gap but **erodes compositional multi-image capability.** → https://arxiv.org/abs/2604.10755

### 4.5 Distractor confusion, polarity reversal, negated-option attraction

- **CXR-ContraBench** (2026) is the reference work here: **negated-option attraction** as a **polarity reversal** — a model that says "No consolidation" while consolidation is visible **"emits a clinical statement contradicting the image."** Direct presence probe: **MedGemma 31.49%, Qwen2.5-VL 30.21%**; on the **135,754-record CheXpert protocol, >62% of presence questions answered with a negated option.** CoT **reduces presence-side reversals but amplifies absence-side contradictions** (i.e. CoT trades one distractor pathology for another). → https://arxiv.org/abs/2605.05810
- **"It is Too Many Options: Pitfalls of Multiple-Choice Questions in Generative AI and Medical Education"** — direct critique of MCQ as an evaluation instrument in medical education. → https://www.semanticscholar.org/author/Jaden-Stryker/2325728846 **[metadata only; UNVERIFIED]**
- **MCQ option-order / positional bias** (general, not medical-specific): "Embracing Positional Bias in Multiple-Choice Question Answering via Permutation Equivariant Neural Networks" (AAAI) → https://ojs.aaai.org/index.php/AAAI/article/view/40401/44362 ; "Mitigating Order Sensitivity in Large Language Models for Multiple-Choice Question Tasks" → https://zenodo.org/records/14043004 ; and a **medical** option-order study, "A novel method for analyzing output variations from LLM caused by different input option orders and improving applications of LLMs in medicine" → https://datadryad.org/dataset/doi:10.5061/dryad.7sqv9s582 **I found no single authoritative Chinese-medical option-order-bias number — treat as an OPEN measurement you could contribute.**

### 4.6 Leakage and contamination (this is the most damaging failure mode for a *new* benchmark)

- **"A Controlled Audit of Pretraining Contamination in Public Medical Vision-Language Benchmarks."** Xu Bruce Changlong, Wu Lan, Ryu Alexander. **June 2026.** Audits open VLMs on **SLAKE-En, PathVQA, VQA-RAD** and a public **OmniMedVQA** mirror using four detector families: image-side near-neighbour overlap **against PMC-OA-beta**, canonical-order exchangeability, cohort-relative **Min-K%++** tail enrichment, and cross-model top-K overlap. Findings: **SLAKE-En image-side overlap 19.8% (SigLIP-B-16) and 4.2% (SigLIP-SO400M), with out-of-domain controls producing 0/2000 flags.** Manual adjudication shows **same-modality, same-projection matches to different patients rather than verified pixel duplicates** → "source or distributional overlap rather than confirmed per-image memorization." Qwen2.5-VL on SLAKE-En shows a **canonical-order exchangeability signal surviving ordering ablation and external non-medical baselines**; on OmniMedVQA exchangeability fires for five medical and general VLMs while BLIP-2 stays clean. **Key methodological warning: cohort-relative Min-K%++ and cross-model top-K overlap collapse under an external pre-domain baseline (BLIP-2 reproduces the apparent signals without medical-VQA exposure), so these detectors are unreliable as standalone membership-inference signals on small medical-VLM cohorts.** → https://arxiv.org/abs/2606.10066
- **"Forensic Reproducibility Audit of a Radiology Vision-Language Model Benchmark: From Intended Protocol to Released Artifact"** — benchmark artifact ≠ intended protocol. → https://zenodo.org/records/21629849
- **MedXpertQA** addresses this by design: **"We perform data synthesis to mitigate data leakage risk."** → https://arxiv.org/abs/2501.18362

---

## 5. Is "yet another MCQ benchmark" considered saturated? What is the actual frontier?

### 5.1 The saturation case (well supported)

1. **Text-only Chinese medical MCQ is at ceiling.** CNMLEQA-10k: **Qwen2.5-32B 90.88%**; CNMLEQA-3k: **DeepSeek-R1 91.59%**. CAMEC (ACL 2026) reports **78.86% CMExam / 84.15% MedQA-CN / 78.51% CMMLU-Med / 74.40% CMB-exam**. CMExam's 2023 GPT-4 baseline was **61.6% vs human 71.6%** — the human ceiling has been passed.
2. **Public benchmark sets are contaminated.** SLAKE-En shows **19.8% image-side source overlap**; canonical-order exchangeability signals survive ablations. Any new *public* MCQ set is at contamination risk from day one.
3. **MCQ as an instrument is itself contested** ("It is Too Many Options"). Aggregate MCQ accuracy provably **hides polarity failures** (CXR-ContraBench: 96.6% after a deterministic negation fix vs 31.5% before — the metric, not the model, was the problem).
4. **The benchmark count is exploding.** In 2026 alone I verified: Med-R2, CXR-ContraBench, MMRareBench, MedThinkVQA, MEDSYN, MedSG-Bench, CNMLEQA, MedBench v4, OmniBrainBench, OralGPT-Omni/MMoral-Uni, MedRepBench, MedImageEdu. A "new MCQ set" paper without a methodological hook is a hard sell.

### 5.2 What the frontier actually looks like (verified patterns)

| Frontier axis | Exemplar | Verified signature |
|---|---|---|
| **Adversarial / misleading cues** | Med-R2 (2026) — 42,432 images, 110,406 QA, 31 tasks | "models rely heavily on correct prompts to guess answers" |
| **Negation / contradiction** | CXR-ContraBench (2026) | >62% negated-option selection on presence questions |
| **Image-text conflict & irrelevant context** | Restrepo et al. (2025, 2026) | decisions "dominated by the text modality"; minor prompt changes reverse correct image-based predictions |
| **Multi-image / report-grounded** | MMRareBench (7,958 images from PMC case reports), MedThinkVQA (6.62 images/case) | ">70% of errors arise from image reading and cross-view integration" |
| **Visual grounding on sequences** | MedSG-Bench (9,630 QA, 76 datasets, 10 modalities) | even Qwen2.5-VL / HuatuoGPT-vision fail |
| **Multi-evidence synthesis (DDx vs FDx)** | MEDSYN (up to 7 CE types/case) | all MLLMs show larger DDx–FDx gap than clinicians |
| **Leakage/contamination control** | Contamination audit (2026); MedXpertQA data synthesis | SLAKE-En 19.8% source overlap |
| **Real-world heterogeneous input** | WoundcareVQA (telemedicine, 768 images), OphthalWeChat (WeChat, 3,469 images) | real-world context, multiple exams per patient |

**Observation that matters most for the proposal:** the *hard-benchmark* frontier and the *Chinese* frontier are **almost disjoint**. Every adversarial/negation/multi-image benchmark I verified is **English and chest-X-ray-centric** (MIMIC-CXR, CheXpert, ReXVQA, OpenI). The Chinese multimodal benchmarks (TCM-Ladder, MedBench v4, OphthalWeChat, RJUA-MedDQA, WoundcareVQA-partial) are **not adversarial, not negation-controlled, and mostly not image-dependence-verified**.

---

## 6. Gap analysis — is it (a) done, (b) saturated, or (c) open?

### Verdict

**(a) Already done? — No, not the specific thing.** No verified peer-reviewed work builds a multimodal **Chinese** medical **MCQ** dataset **from online Chinese case reports (iiyi.com / dxy.cn / 丁香园)**. The nearest neighbours each miss on a different axis:
- TCM-Ladder — Chinese multimodal MCQ ✅, but **TCM / textbook / video**, not case reports.
- OphthalWeChat — Chinese images + Chinese single-choice MCQ ✅, but **one specialty**, **WeChat Official Accounts (publisher content)**, **GPT-4o-mini caption-derived**.
- RJUA-MedDQA — Chinese multimodal ✅, but **medical documents (screenshots/scanned PDFs)**, not case reports.
- MMRareBench — **case-report-derived multimodal** ✅, but **English, PMC**, not MCQ-only, not Chinese.
- MedBench v4 — Chinese multimodal ✅, but **closed/expert-curated platform**, not an open case-report corpus.

**(b) Saturated? — The MCQ *format* on exam-style text is saturated; the *source × language × adversarial-control* combination is not.** A paper whose contribution is "here are N more Chinese medical MCQs with images" will be rejected as incremental. A paper whose contribution is "a Chinese real-world case-report benchmark with *measured and controlled* image-dependence, negation, and text-prior shortcuts" is aligned with the 2026 frontier.

**(c) Opening? — Yes, but it is narrow and it is not about size.**

### Where exactly the opening is (in order of defensibility)

1. **Chinese + real-world case-report distribution + adversarial controls — the intersection nobody occupies.** Build the dataset **around failure modes, not around question count**. Concretely: every item ships with (i) the original forum image(s), (ii) a **text-only counterpart** (question + thread text, image withheld), (iii) an **image-swapped / image-shuffled** counterpart, (iv) a **negated-option** variant. Following Restrepo et al.'s **Selective Modality Shifting** and CXR-ContraBench's polarity probes, report **image-necessity (Δ = full − text-only)** as a first-class metric. **A Chinese multimodal medical MCQ set that can *prove* each item needs the image is genuinely new.**
2. **Ground the label in case-level evidence, not in a single model's guess.** MMRareBench's "evidence-grounded annotations" + "track-specific leakage control" is the template. For forum threads, this means: record the **final consensus diagnosis**, the **differential list raised in the thread**, and the **resolution status** — and expose **contested / unresolved cases as "unanswerable" items**. **Unanswerable and ambiguity-flagged items are a documented frontier axis (Med-R2, CXR-ContraBench) and are essentially absent from Chinese medical benchmarks.**
3. **Heterogeneous real images as a feature, not a bug.** Forum cases contain phone-camera clinical photos, dermatology close-ups, wound photos, imaging screenshots, lab-report photos, and multi-visit progressions **in the same thread**. Every verified Chinese multimodal benchmark is **modality-pure**; WoundcareVQA and MEDSYN show heterogeneity is what clinical reality looks like. A **multi-image temporal progression track** ("pre- vs post-treatment") mirrors MedSG-Bench's Image Difference Grounding but in a **Chinese, case-report** setting.
4. **OCR / burned-in-text control — an uncontested, cheap contribution.** The OCR-mediated modality dominance result is **medRxiv-only and English radiology**. Doing the OCR-control analysis on Chinese web medical images (where patient name, hospital, date, and laterality markers are burned in) would be a **first-of-kind Chinese finding**. ⚠️ Verify the medRxiv paper directly before citing its numbers.
5. **Publish the leakage/contamination audit for your own dataset.** The 2026 audit shows **SLAKE-En at 19.8% source overlap** and that **cohort-relative Min-K%++ detectors are unreliable on small medical-VLM cohorts.** Since case-report images are **already public on the open web**, a new dataset inherits maximal contamination risk. A **pre-registered image-side near-neighbour audit against PMC-OA-beta and Chinese web corpora**, plus a cutoff date and held-out recent cases, would be a differentiator rather than a liability.

### Honest risks a reviewer / supervisor will raise

| Risk | Severity | Mitigation |
|---|---|---|
| **Copyright / patient privacy** of iiyi.com / dxy.cn images. Chinese case-sharing sites have explicit terms; images often contain PII. | **High — potentially fatal** | Do not redistribute raw images; ship URLs + hashes + a re-crawl script; IRB/ethics review; face/tattoo/PII redaction; test on a sample before committing. **Get institutional legal/ethics sign-off before collecting.** |
| **Label validity** — forum "diagnosis" is often contested, sometimes never resolved. | **High** | Multi-annotator clinician adjudication; only admit items with ≥k agreeing clinicians; explicitly separate the **unanswerable/contested** stratum. |
| **Not actually multimodal** — models may answer from the thread text alone. | Medium | This is the *point*: measure and report it. Items that fail the image-necessity test should be **reported as a finding**, not silently dropped. |
| **"Just another MCQ set"** | Medium | Lead with the **failure-mode instrumentation and adversarial design**, not with size. |
| **Snowballing competition** — 2026 is producing benchmarks monthly. | Medium | Target a venue with a dataset track (NeurIPS D&B, ACL/EMNLP, MICCAI, Scientific Data) and move fast; **check the arXiv listings immediately before submission** for a Chinese case-report multimodal benchmark. |
| **The unverified `iiyi_dataset` HF repo** could already contain exactly this. | Medium | **Verify in a browser before writing the proposal.** If it contains case images but no MCQ/benchmark, it is a *resource* you can cite and build on, which strengthens rather than kills the proposal. |

### Suggested sharpened one-line framing

> *A Chinese, real-world case-report multimodal benchmark with per-item adversarial and image-necessity controls, testing whether medical VLMs reason from Chinese patient images or from the Chinese text prior they were trained on.*

That framing is defensible against every item in §1–§5. "A new multimodal Chinese medical MCQ dataset" on its own is not.

---

## Appendix A — Quick reference table

| Name | Year/Venue | Lang | Modalities | Task | Size | Source |
|---|---|---|---|---|---|---|
| CMExam | NeurIPS 2023 | ZH | text | MCQ | 60K+ | Chinese NMLE |
| CMB-Exam | NAACL 2024 | ZH | text | MCQ | 280,839 | mock/practice exams |
| CMB-Clin | NAACL 2024 | ZH | text | open QA | 74 cases | textbooks |
| MedBench v1 | AAAI-24 | ZH | text | mixed MCQ | 40,041 | exams + clinic cases |
| MedBench v4 | Nov 2025 | ZH | text+**multimodal**+agents | mixed, 3 tracks | **>700,000** | 500+ institutions, expert-curated |
| Zhongjing | AAAI 2024 | ZH | text | dialogue | — | online consultation |
| CNMLEQA | Sci Data 2026 | ZH | text | MCQ (5-opt) | 9,890 + 2,949 | PubMed/GitHub/MedExamLLM |
| Huatuo-26M | 2023 | ZH | text | open QA | 26M+ | encyclopedia/KG/**forums** |
| MLEC-QA | EMNLP 2021 | ZH | text | MCQ | — | Chinese NMLE |
| SLAKE | ISBI 2021 | **EN+ZH** | radiology imaging | open/closed VQA | **642 img / 14,028 QA** | MSD, CHAOS, ChestX-ray8 |
| VQA-RAD | 2018 | EN | radiology | VQA | 315 img / 3.5K QA | clinician-authored |
| PathVQA | 2020 | EN | pathology | VQA | ~5K img / 32.8K QA | pathology textbooks |
| GMAI-MMBench | NeurIPS 2024 | EN-mostly | 38 modalities | VQA | 284 datasets | aggregation |
| MedXpertQA | ICML 2025 | EN | text + MM images, records | MCQ/exam | 4,460 | specialty boards + synthesis |
| TCM-Ladder | NeurIPS 2025 | **ZH** | text+**image+video** | **MCQ**, fill-blank, dialogue, visual | **>52,000** | TCM textbooks + SHUTCM video |
| MedSG-Bench | NeurIPS 2025 | EN | 10 modalities, sequences | grounding VQA | 9,630 QA / 76 datasets | aggregation |
| OphthalWeChat | 2025 (Sci Direct) | **ZH+EN** | ophthalmic, 29 modalities | binary/**single-choice**/open | 3,469 img / 30,120 QA | **WeChat Official Accounts** + GPT-4o-mini |
| RJUA-QA | 2023 | ZH | text | QA | 2,129 | virtual patients, Renji Hospital |
| RJUA-MedDQA | KDD 2024 | ZH | **medical documents** (screenshots/PDF) | extraction + QA | **2,000 images** | Ant Group |
| WoundcareVQA | JBI 2025 | multilingual | wound photos | VQA | 477 patients / 768 img | **online telemedicine** |
| Med-R2 | 2026 | EN | medical imaging | **adversarial** stepwise VQA | 42,432 img / 110,406 QA | — |
| CXR-ContraBench | 2026 | EN | chest X-ray | **negation** probe MCQ | 135,754-record protocol | ReXVQA/OpenI/CheXpert |
| MMRareBench | 2026 | EN | multi-image | 4 clinical tracks | 1,756 QA / 7,958 img | **PMC case reports** |
| MedThinkVQA | ICLR 2026 | EN | multi-image (6.62/case) | diagnostic MCQ + step eval | 8,067 cases | expert-annotated |
| MEDSYN | ACL Findings 2026 | multilingual | up to 7 CE types | DDx + FDx | — | complex clinical cases |
| OMNI | 2025 | — | dental multi-view | classification | 4,166 img / 384 subjects | dentist-annotated |
| `iiyi_dataset` | unknown | ZH? | unknown | unknown | **UNVERIFIED** | **iiyi.com?** |

## Appendix B — Items I could NOT verify (do not cite as fact)

1. **`SnivellusSnape/iiyi_dataset`** — contents, size, format, license. All HF access paths failed from this environment.
2. **OCR-Mediated Modality Dominance** (medRxiv 2026) — HTTP 403; numbers unverified.
3. **TCM NMLE LLM evaluation** (Sci Rep 2026) — paywall; title/venue only.
4. **PerMed-MM** — only a snippet about a text-only ablation.
5. **MedRepBench, MedImageEdu, OmniBrainBench, OralGPT-Omni / MMoral-Uni** — names and venues from search snippets only.
6. **MEDSYN language composition** — described as "multilingual"; exact Chinese share unknown.
7. **OMNI language** — publication venue and Chinese/English composition of the dental OMNI dataset not confirmed.
8. **Zhongjing-OMNI** — a multimodal TCM dataset at `huggingface.co/datasets/CMLM/ZhongJing-OMNI` is referenced in search results and appears to be titled "First Multimodal Traditional Chinese Medicine Dataset" (GitHub: https://github.com/pariskang/ZhongJing-OMNI), but **HF was unreachable so I could not verify its size, modalities, or task format. If real, this is potentially the closest prior art to the proposal — verify first in a browser.**
9. **MCQ option-order/position bias specifically in Chinese medical MCQ** — no authoritative quantitative study found.
