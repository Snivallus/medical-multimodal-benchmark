"""Stage 2: turn selected cases into multiple-answer items.

One item per case, with four steps:

1. **Rewrite** — an LLM removes everything that names or implies the final
   diagnosis: the title, sentences such as 门诊以"…"收入院, imaging and pathology
   conclusions, intra-operative findings. Clinical facts and lab values stay.
2. **Audit** — two checks run on the rewritten text: a deterministic scan for a
   leftover imaging report, and a targeted question per correct diagnosis ("can
   this diagnosis be established from the text alone?"). A leftover report or an
   "A" verdict triggers a second rewrite on the stronger model; after two
   attempts the item is kept but flagged for review.
3. **Distractors** — near-miss diagnoses are retrieved from the corpus first, then
   an LLM picks three that a careless reader would pick and that are actually
   wrong for this case; the retrieval pool tops the list up if the model returns
   fewer than three.
4. **Assembly** — correct options plus distractors are shuffled with a seed
   derived from the case, so item order is reproducible and not biased.

Only cases with 1-3 diagnoses are used, because every diagnosis in ``诊断结果``
has to be a correct option and a 19-diagnosis case would be unanswerable.
Chronic comorbidities are moved out of the answer set.

Usage::

    python bench/build_items.py --limit 100
    python bench/build_items.py --limit 3 --dry-run     # print prompts only
"""

import argparse
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import llm
import medical
import retrieval

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS = REPO_ROOT / "iiyi_dataset" / "results" / "case_with_images.json"
CANDIDATES = REPO_ROOT / "bench" / "data" / "candidates.jsonl"
OUTPUT = REPO_ROOT / "bench" / "data" / "items.jsonl"

# Fields that describe the patient. Everything else (title, diagnosis, treatment)
# is withheld from the stem.
STEM_FIELDS = ["主诉", "现病史", "既往史", "个人史", "家族史", "查体", "辅助检查"]

# Fields that survive the rewrite. ``辅助检查`` is excluded on purpose: it is
# mostly an imaging report and is removed, so it says nothing about how much
# text the final stem can keep.
HISTORY_FIELDS = ["主诉", "现病史", "既往史", "个人史", "家族史", "查体"]

MIN_STEM_CHARS = 120
MAX_STEM_CHARS = 2500

# A case whose history and examination are this short has its whole content in
# the imaging report, so the rewritten stem is a fragment and the item degenerates
# into "guess from the picture". Measured on the corpus, the 10th percentile is
# 129 characters, so this gate removes roughly that tail.
MIN_HISTORY_CHARS = 150
MIN_IMAGES = 1
MAX_IMAGES = 4
MAX_DIAGNOSES = 3

# A distractor this similar to a correct option is probably a wording variant of
# the same disease rather than a genuinely different one.
GRAY_ZONE_SIMILARITY = 0.45

REWRITE_SYSTEM = "你是严谨的医学病例编辑，只做删改，不编造，只输出 JSON。"

REWRITE_PROMPT = """下面是一份中文病例的临床资料。请改写成一段"不含诊断结论"的病例描述。

必须删除：
1. 任何直接写出诊断或疾病名称的表述，例如"门诊以XX收入院""考虑XX""诊断为XX"；
2. 影像、病理、内镜、术中探查的结论性描述，例如"MRI示…""病理示…""术中见…"；可保留"已行相关影像检查，结果见附图"这类中性表述；
3. 任何提示性总结、鉴别诊断、治疗方案。

必须保留：
1. 起病时间与诱因、症状及演变、既往史、个人史、家族史；
2. 查体的阳性与阴性体征；
3. 辅助检查中的实验室数值（血常规、生化、凝血等）及其单位。

要求：
- 只删改，不新增任何医学信息，不改变数值与单位；
- 不要出现"该病例""本例"这类元叙述；
- 输出 JSON：{{"stem": "改写后的文本", "removed": ["删掉的要点", ...]}}

【临床资料】
{fields}
"""

AUDIT_SYSTEM = "你是临床医生，只输出 JSON。"

AUDIT_PROMPT = """下面是一段已去除影像结论的病例文本。

问题：仅凭这段文本，能否确立"诊断：{diagnosis}"？

判断标准（只选一个）：
A = 文本已经足够确立该诊断，不需要再看影像；
B = 方向接近，但不足以确定到这一具体诊断，必须看影像才能确认；
C = 这段文本完全不足以指向该诊断。

只输出 JSON：{{"verdict": "A", "why": "20 字内"}}

【病例文本】
{stem}
"""

DISTRACTOR_SYSTEM = "你是医学考试命题专家，只输出 JSON。"

DISTRACTOR_PROMPT = """下面是一道多选题的病例材料、正确诊断，以及从病例库中检索到的相近诊断候选池。

请从候选池中挑 3 个"粗心的读者很可能误选"的诊断作为干扰项。

硬性要求：
1. 不能是正确诊断本身，也不能与正确诊断互相包含；
2. 不能是正确诊断的同义表述、亚型、分期或严重程度变体（例如正确诊断为"急性脑梗死"时，不能给"腔隙性脑梗死""脑梗塞灶"；正确诊断为"前交叉韧带断裂"时，不能给"前交叉韧带损伤"）；
3. 不能在本病例中真实成立；
4. 尽量来自同一系统或同一部位，与原诊断只差关键的一点（部位对侧、相邻结构、性质不同）；
5. 三个干扰项之间不能重复。

【病例材料】
{stem}

【正确诊断】
{correct}

【候选池】
{pool}

输出 JSON：{{"distractors": [{{"diagnosis": "…", "why": "为什么容易误选，20 字内"}}]}}
"""


# ------------------------------------------------------------------
# Case loading and selection
# ------------------------------------------------------------------
def load_corpus():
    """Load the case corpus and attach parsed diagnoses."""

    with open(CORPUS, encoding="utf-8") as f:
        cases = json.load(f)["cases"]

    for case in cases:
        case["_diagnoses"] = medical.parse_diagnoses(case.get("诊断结果", ""))

    return cases


def load_candidates(path):
    """Load the screening output as ``{uid: record}``."""

    records = {}

    if not Path(path).exists():
        return records

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records[record["uid"]] = record

    return records


def eligible(case, record):
    """Return ``(ok, reason)`` for whether a case can become an item."""

    diagnoses = case["_diagnoses"]
    answers, _ = medical.split_answer_set(diagnoses)
    images = case.get("images") or []

    if not answers:
        return False, "no answer diagnosis (only comorbidities)"
    if len(diagnoses) > MAX_DIAGNOSES:
        return False, f"too many diagnoses ({len(diagnoses)})"
    if not (MIN_IMAGES <= len(images) <= MAX_IMAGES):
        return False, f"image count {len(images)}"

    history_chars = sum(
        len(medical.normalize_whitespace(str(case.get(field, ""))))
        for field in HISTORY_FIELDS
    )

    if history_chars < MIN_HISTORY_CHARS:
        return False, "history too short"

    if not record.get("has_visual_modality", True):
        return False, "no visual modality"

    if record.get("llm_score") is not None and record["llm_score"] < 2:
        return False, f"llm score {record['llm_score']}"

    return True, ""


def select_cases(corpus, candidates, limit):
    """Pick the cases that will be turned into items."""

    chosen = []
    skipped = {}

    for case in corpus:
        uid = f"with_images:{case['case_idx']}"
        record = candidates.get(uid, {})

        ok, reason = eligible(case, record)

        if not ok:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue

        chosen.append((case, record))

    # Cases the screening pass liked best come first; ties keep corpus order.
    chosen.sort(key=lambda pair: (-(pair[1].get("llm_score") or 0), pair[0]["case_idx"]))

    return (chosen if limit is None else chosen[:limit]), skipped


# ------------------------------------------------------------------
# Prompt building
# ------------------------------------------------------------------
def render_fields(case):
    """Render the clinical fields of a case for the rewrite prompt."""

    lines = []

    for field in STEM_FIELDS:
        value = medical.normalize_whitespace(str(case.get(field, "")))
        if value:
            lines.append(f"【{field}】{value}")

    return "\n".join(lines)


def stable_seed(uid):
    """Stable seed derived from the case uid."""

    return int(hashlib.sha256(uid.encode("utf-8")).hexdigest()[:8], 16)


# ------------------------------------------------------------------
# LLM steps
# ------------------------------------------------------------------
def rewrite_stem(case, model=None):
    """Ask the model for a de-identified version of the case text."""

    data = llm.chat_json(
        [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": REWRITE_PROMPT.format(fields=render_fields(case))},
        ],
        model=model,
        temperature=0.2,
        max_tokens=2048,
        timeout=(10, 300),
    )

    stem = medical.normalize_whitespace(data.get("stem") or "")
    removed = [str(item) for item in (data.get("removed") or [])]

    return stem, removed


def audit_diagnosis(stem, diagnosis, model=None):
    """Ask whether the text alone is enough to establish one diagnosis.

    An open-ended "what is the diagnosis?" question turned out to be unreliable:
    the model often answers with a related disease, and string similarity cannot
    tell "related but different" (肩关节脱位 vs 肩关节盂骨折, ratio 0.62) from
    "same disease worded differently" (前交叉韧带损伤 vs 完全断裂, ratio 0.64).
    Asking about one specific diagnosis at a time avoids that ambiguity.
    """

    data = llm.chat_json(
        [
            {"role": "system", "content": AUDIT_SYSTEM},
            {"role": "user", "content": AUDIT_PROMPT.format(stem=stem, diagnosis=diagnosis)},
        ],
        model=model,
        temperature=0.0,
        max_tokens=96,
        timeout=(10, 300),
    )

    verdict = str(data.get("verdict") or "").strip().upper()[:1]

    return verdict, str(data.get("why") or "").strip()


def make_distractors(stem, diagnoses, pool, model=None):
    """Ask the model to pick three hard negatives from the retrieved pool."""

    data = llm.chat_json(
        [
            {"role": "system", "content": DISTRACTOR_SYSTEM},
            {
                "role": "user",
                "content": DISTRACTOR_PROMPT.format(
                    stem=stem,
                    correct="；".join(diagnoses),
                    pool="\n".join(f"- {entry['name']}" for entry in pool),
                ),
            },
        ],
        model=model,
        temperature=0.4,
        max_tokens=512,
        timeout=(10, 300),
    )

    picked = []

    for entry in data.get("distractors") or []:
        if isinstance(entry, dict) and entry.get("diagnosis"):
            picked.append(
                {
                    "diagnosis": str(entry["diagnosis"]).strip(),
                    "why": str(entry.get("why") or "").strip(),
                }
            )
        elif isinstance(entry, str):
            picked.append({"diagnosis": entry.strip(), "why": ""})

    return picked


def complete_distractors(picked, diagnoses, pool, needed=3):
    """Top the distractor list up from the retrieval pool.

    The model sometimes returns fewer than three usable options. Without this,
    every such item would land in review, so the pool (already filtered for
    near-duplicates) fills the gap deterministically.
    """

    chosen = [entry for entry in picked if entry.get("diagnosis")][:needed]
    names = [entry["diagnosis"] for entry in chosen]

    for entry in pool:

        if len(chosen) >= needed:
            break

        name = entry["name"]

        if retrieval.is_too_close(name, diagnoses):
            continue

        if any(retrieval.is_too_close(name, [existing]) for existing in names):
            continue

        chosen.append({"diagnosis": name, "why": "retrieval fallback"})
        names.append(name)

    return chosen


# ------------------------------------------------------------------
# Assembly and validation
# ------------------------------------------------------------------
def assemble_options(diagnoses, distractors, seed):
    """Shuffle correct options and distractors into a lettered option list."""

    options = [{"text": name, "correct": True} for name in diagnoses]
    options += [{"text": entry["diagnosis"], "correct": False} for entry in distractors]

    random.Random(seed).shuffle(options)

    for index, option in enumerate(options):
        option["id"] = chr(ord("A") + index)

    correct_ids = [option["id"] for option in options if option["correct"]]

    return options, correct_ids


def validate(item):
    """Return the list of problems that block an item from being ``ready``."""

    problems = []
    warnings = []

    stem = item["stem"]
    options = item["options"]
    correct = [o["text"] for o in options if o["correct"]]
    wrong = [o["text"] for o in options if not o["correct"]]

    if len(stem) < MIN_STEM_CHARS:
        problems.append("stem too short")
    if len(stem) > MAX_STEM_CHARS:
        problems.append("stem too long")

    if not (1 <= len(correct) <= MAX_DIAGNOSES):
        problems.append(f"{len(correct)} correct options")
    if len(wrong) < 3:
        problems.append(f"only {len(wrong)} distractors")

    texts = [medical.normalize_whitespace(o["text"]) for o in options]
    if len(set(texts)) != len(texts):
        problems.append("duplicate option text")

    for name in wrong:
        if retrieval.is_too_close(name, correct):
            problems.append(f"distractor too close to an answer: {name}")
            continue

        # A very similar but not identical distractor is often a same-disease
        # wording variant, which makes the answer ambiguous. Flag it for review
        # instead of rejecting it, so a human decides.
        closest = max(retrieval.similarity(name, answer) for answer in correct)

        if closest >= GRAY_ZONE_SIMILARITY:
            warnings.append(f"possibly ambiguous distractor: {name} ({closest:.2f})")

    if item["audit"]["leaked"]:
        problems.append("text still leaks the diagnosis")

    if item["audit"].get("failed"):
        problems.append("audit could not run")

    item["warnings"] = warnings

    return problems


def build_item(case, record, pool_entries, model_rewrite=None, model_fast=None, model_judge=None):
    """Build one item, or return ``None`` when the case has to be dropped."""

    uid = f"with_images:{case['case_idx']}"
    diagnoses = case["_diagnoses"]
    answers, background = medical.split_answer_set(diagnoses)
    seed = stable_seed(uid)

    # 1-2: rewrite, then check that the text no longer gives the answer. The
    # first attempt uses the fast model; if the text still leaks, the second
    # attempt escalates to the stronger model, which follows the deletion rules
    # much more reliably.
    attempts = 0
    stem = ""
    removed = []
    checks = []
    leaked = True
    reasons = []
    audit_failed = False

    while attempts < 2:
        attempts += 1

        if attempts == 1:
            model = model_rewrite or model_fast
        else:
            model = model_judge or model_fast

        stem, removed = rewrite_stem(case, model=model)
        reasons = []
        audit_failed = False

        # Deterministic guard: a report left in place is a leak no matter what
        # the model claims in its "removed" list.
        if medical.mentions_imaging_conclusion(stem):
            reasons.append("imaging report still present")

        # Targeted audit, one verdict per correct diagnosis.
        checks = []

        for diagnosis in answers:
            try:
                verdict, why = audit_diagnosis(stem, diagnosis, model=model_fast)
            except Exception as error:
                # An audit that cannot run must not be read as a clean result.
                verdict, why = None, str(error)[:120]
                audit_failed = True

            checks.append({"diagnosis": diagnosis, "verdict": verdict, "why": why})

            if verdict == "A":
                reasons.append(f"text alone establishes: {diagnosis}")

        leaked = bool(reasons)

        if not leaked and not audit_failed:
            break

    # 3: distractors from the retrieved pool. This step needs actual medical
    # judgement, so it runs on the stronger model.
    distractors = complete_distractors(
        make_distractors(stem, answers, pool_entries, model=model_judge),
        answers,
        pool_entries,
    )

    options, correct_ids = assemble_options(answers, distractors, seed)

    item = {
        "item_id": "",  # assigned by the driver
        "uid": uid,
        "case_id": case["case_idx"],
        "source_url": case.get("url", ""),
        "images": case.get("images") or [],
        "modalities": record.get("modalities", []),
        "llm_score": record.get("llm_score"),
        "stem": stem,
        "stem_chars": len(stem),
        "options": options,
        "correct_ids": correct_ids,
        "main_diagnosis": answers[0],
        "diagnoses": answers,
        "comorbidities": background,
        "distractor_pool": [entry["name"] for entry in pool_entries],
        "explanation": {
            "诊断依据": case.get("诊断依据", ""),
            "治疗方案": case.get("治疗方案", ""),
            "治疗经过": case.get("治疗经过", ""),
        },
        "audit": {
            "checks": checks,
            "leaked": leaked,
            "failed": audit_failed,
            "attempts": attempts,
            "reasons": reasons,
        },
        "removed": removed,
        "status": "ready",
    }

    problems = validate(item)

    if problems:
        item["status"] = "needs_review"
        item["problems"] = problems

    return item


# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build multiple-answer items from cases.")

    parser.add_argument("--candidates", default=str(CANDIDATES), help="screening output")
    parser.add_argument("--output", default=str(OUTPUT), help="items JSONL (ready items only)")
    parser.add_argument(
        "--review",
        default=None,
        help="where rejected items go (default: <output>_rejected.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="stop when the output file holds this many items (counts existing ones)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print prompts, call no model")
    parser.add_argument("--no-resume", action="store_true", help="rebuild existing items")
    parser.add_argument("--workers", type=int, default=4, help="cases processed in parallel")
    parser.add_argument(
        "--model-rewrite",
        default=None,
        help="model for the first rewrite attempt (defaults to --model-fast)",
    )
    parser.add_argument(
        "--model-fast",
        default=None,
        help="model for the audit (defaults to LLM_MODEL)",
    )
    parser.add_argument(
        "--model-judge",
        default=None,
        help="model for distractor selection and the escalated rewrite "
        "(defaults to LLM_MODEL)",
    )

    args = parser.parse_args()

    corpus = load_corpus()
    candidates = load_candidates(args.candidates)

    # Select more cases than needed: some are dropped when the text keeps
    # leaking the diagnosis or when the distractor pool is too thin.
    all_eligible, skipped = select_cases(corpus, candidates, None)
    selected = all_eligible[: args.limit * 3]

    if args.dry_run:
        selected = selected[:3]

    print(f"corpus: {len(corpus)} cases, screened: {len(candidates)}")
    print(f"eligible: {len(all_eligible)} | taking {len(selected)}")
    print("skipped:", dict(sorted(skipped.items(), key=lambda kv: -kv[1])))

    if not selected:
        raise SystemExit("nothing to build; run bench/screen_cases.py first")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    review_path = Path(args.review) if args.review else output.with_name(output.stem + "_rejected.jsonl")

    done = set()
    used_ids = set()
    existing = 0

    # A case that failed review must not be rebuilt on every run: it would burn
    # the same API calls and produce the same flag again. Both files count as
    # "already handled".
    for path in (output, review_path):

        if not path.exists() or args.no_resume:
            continue

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                done.add(record["uid"])
                if record.get("item_id"):
                    used_ids.add(record["item_id"])

                if path == output and record.get("status") == "ready":
                    existing += 1

    # Neighbour pool for distractors, computed once.
    corpus_index = retrieval.build_corpus_index(corpus)

    def work(position, case, record):
        """Build one item and stamp its stable id."""

        pool = retrieval.top_similar(
            case["_diagnoses"][0],
            corpus_index,
            k=20,
            exclude_case_ids=[case["case_idx"]],
            exclude_names=case["_diagnoses"],
        )

        started = time.time()
        item = build_item(
            case,
            record,
            pool,
            model_rewrite=args.model_rewrite,
            model_fast=args.model_fast,
            model_judge=args.model_judge,
        )
        item["elapsed_s"] = round(time.time() - started, 1)

        return item

    def assign_id(item, position):
        """Give the item a readable id that cannot collide with an existing one.

        Positions follow the candidate ordering, which can shift when more cases
        are screened, so an id already present in the file is extended instead of
        being reused for a different case.
        """

        item_id = f"iiyi-{position:04d}"

        while item_id in used_ids:
            item_id += "b"

        used_ids.add(item_id)
        item["item_id"] = item_id

        return item

    written = 0
    print(f"existing ready items: {existing} | already rejected: {len(done) - existing} | target: {args.limit}")

    pending = [
        (position, case, record)
        for position, (case, record) in enumerate(selected)
        if f"with_images:{case['case_idx']}" not in done
    ]

    with open(output, "a" if not args.no_resume else "w", encoding="utf-8") as f, \
            open(review_path, "a" if not args.no_resume else "w", encoding="utf-8") as review_file:

        if args.dry_run:
            for position, case, record in pending[:3]:
                pool = retrieval.top_similar(
                    case["_diagnoses"][0],
                    corpus_index,
                    k=20,
                    exclude_case_ids=[case["case_idx"]],
                    exclude_names=case["_diagnoses"],
                )
                print("=" * 70)
                print(f"[{position}] case {case['case_idx']} | {case['标题']}")
                print(f"diagnoses: {case['_diagnoses']}")
                print(f"images: {len(case.get('images') or [])} | modalities: {record.get('modalities')}")
                print("\n--- rewrite prompt ---")
                print(REWRITE_PROMPT.format(fields=render_fields(case))[:1200])
                print("\n--- pool ---")
                print(", ".join(entry["name"] for entry in pool[:10]))

        else:
            # Submit everything up front and write each item the moment it is
            # finished. Waiting for a whole batch would let one slow call hold
            # back seven finished items, which is what stalled the first run.
            executor = ThreadPoolExecutor(max_workers=max(1, args.workers))
            futures = {
                executor.submit(work, position, case, record): (position, case)
                for position, case, record in pending
            }

            try:
                for future in as_completed(futures):

                    if existing + written >= args.limit:
                        break

                    position, case = futures[future]

                    try:
                        item = future.result()
                    except Exception as error:
                        print(f"[FAIL] case {case['case_idx']}: {error}")
                        continue

                    assign_id(item, position)

                    # Only clean items belong in the deliverable; flagged ones go
                    # to the review file and are not rebuilt later.
                    if item["status"] == "ready":
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        f.flush()
                        written += 1
                        flag = "ok"
                    else:
                        review_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                        review_file.flush()
                        flag = "review: " + ",".join(item.get("problems", []))

                    print(
                        f"[{item['item_id']}] case {item['case_id']:4d} | "
                        f"{item['stem_chars']:4d} chars | {item['elapsed_s']:5.1f}s | {flag}",
                        flush=True,
                    )
            finally:
                # Drop whatever is still queued once enough items exist.
                executor.shutdown(wait=False, cancel_futures=True)

        print(f"\nwrote {written} new items (total {existing + written}) -> {output}")

    if not args.dry_run:
        report(output, review_path)


def report(path, review_path=None):
    """Print a short quality report over the item file and the reject file."""

    total = 0
    option_letters = {}
    correct_counts = {}
    leaks = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            leaks += 1 if item["audit"]["leaked"] else 0
            correct_counts[len(item["correct_ids"])] = correct_counts.get(len(item["correct_ids"]), 0) + 1

            for option_id in item["correct_ids"]:
                option_letters[option_id] = option_letters.get(option_id, 0) + 1

    print(f"\nitems: {total} | residual leaks: {leaks}")
    print(f"correct options per item: {dict(sorted(correct_counts.items()))}")
    print(f"answer position distribution: {dict(sorted(option_letters.items()))}")

    if review_path and Path(review_path).exists():

        reasons = {}

        with open(review_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for problem in item.get("problems", []):
                    reasons[problem] = reasons.get(problem, 0) + 1

        print(f"rejected items: {sum(reasons.values()) or 'see file'} -> {review_path}")
        print(f"reject reasons: {dict(sorted(reasons.items(), key=lambda kv: -kv[1]))}")


if __name__ == "__main__":
    sys.exit(main())
