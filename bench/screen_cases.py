"""Stage 1: turn the cleaned case corpus into a candidate pool.

The stage runs in two passes:

1. **Rule pass** (always). Every case is tagged with the examination modalities
   its text mentions, e.g. ``CT``, ``MRI``, ``病理``, ``内镜``. This pass is
   recall-oriented: it only decides which bucket a case belongs to.

2. **LLM pass** (optional, needs an API key). Each case is scored 0-3 on how
   much its final diagnosis depends on a visual examination. The model reads the
   case text only; the question is whether the diagnosis could be settled from
   history and physical examination alone.

Output is JSONL, one record per case, appended incrementally so the pass can be
resumed:

    {"case_id": 12, "title": "...", "source": "with_images",
     "modalities": ["CT", "病理"], "image_count": 2,
     "llm_score": 3, "llm_reason": "..."}

Cases with ``llm_score >= 2`` are the candidates for item construction.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import llm
import medical

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUTS = [
    REPO_ROOT / "iiyi_dataset" / "results" / "case_with_images.json",
]
TEXT_ONLY_INPUT = REPO_ROOT / "iiyi_dataset" / "results" / "case_without_images.json"
DEFAULT_OUTPUT = REPO_ROOT / "bench" / "data" / "candidates.jsonl"

# Fields that describe the patient rather than the final verdict.
SCAN_FIELDS = [
    "标题", "关键词", "主诉", "现病史", "既往史", "查体",
    "辅助检查", "诊断依据", "治疗方案", "治疗经过", "诊断结果",
]

SCORE_RUBRIC = """评分标准（只判断“最终诊断”对可视化检查的依赖程度，与病例文字长短无关）：
3 = 几乎只能靠影像/病理定诊断，例如骨折、韧带断裂、肿瘤良恶性、占位性质；
2 = 影像/病理提供关键信息，但病史与查体也给出明显方向；
1 = 影像只用于排除或补充，诊断主要靠病史、查体或化验；
0 = 诊断与影像无关，临床表现或化验即可确定，或病例没有可用影像。"""

PROMPT_TEMPLATE = """你是医学评测集的筛选助手。下面是一份已脱敏的中文病例文本。
请判断：要得出该病例的最终诊断，医生是否必须查看影像/病理等可视化检查？

{case_text}

{rubric}

只输出 JSON，不要解释：
{{"score": 0, "modalities": ["CT"], "reason": "不超过 40 字"}}

其中 modalities 填该病例真正依赖的检查类型，可多选，取值范围：
CT、MRI、X线、超声、内镜、病理、皮肤镜、临床照片、眼底、心电图、核医学、无。
"""


def detect_modalities(case):
    """Return the modalities mentioned anywhere in the case text."""

    text = "\n".join(str(case.get(field, "")) for field in SCAN_FIELDS)

    return medical.detect_modalities(text)


def build_case_text(case):
    """Render the case as the plain text the screening model reads.

    Only the fields a clinician would see are included, in a fixed order, so the
    prompt is stable across runs.
    """

    lines = []

    for field in ["标题", "主诉", "现病史", "既往史", "个人史", "家族史",
                  "查体", "辅助检查", "诊断依据", "诊断结果"]:
        value = str(case.get(field, "")).strip()
        if value:
            lines.append(f"【{field}】{value}")

    return "\n".join(lines)


def screen_with_llm(case, dry_run=False, model=None):
    """Score one case with the LLM and return ``(score, modalities, reason)``."""

    prompt = PROMPT_TEMPLATE.format(
        case_text=build_case_text(case),
        rubric=SCORE_RUBRIC,
    )

    if dry_run:
        print(prompt)
        return None, [], None

    data = llm.chat_json(
        [
            {"role": "system", "content": "你是严谨的医学病例审核助手，只输出 JSON。"},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=0.0,
        max_tokens=256,
    )

    try:
        score = int(data.get("score"))
    except (TypeError, ValueError):
        score = None

    modalities = [m for m in (data.get("modalities") or []) if isinstance(m, str)]

    return score, modalities, str(data.get("reason") or "")


def load_done_ids(path):
    """Read the case uids already present in the output file."""

    done = set()

    if not path.exists():
        return done

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                done.add(record["uid"])
            except (json.JSONDecodeError, KeyError):
                # A half-written last line after a crash is simply ignored.
                continue

    return done


def load_cases(paths):
    """Load cases from one or more ``{"cases": [...]}`` files."""

    cases = []

    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        source = "with_images" if "with_images" in Path(path).name else "without_images"

        for case in data.get("cases", []):
            case["_source"] = source
            # Both files number their cases from zero, so the source has to be
            # part of the identity.
            case["_uid"] = f"{source}:{case.get('case_idx')}"
            cases.append(case)

    return cases


def build_record(case, use_llm, dry_run=False, model=None):
    """Build one candidate record for a case."""

    case_id = case["case_idx"]
    modalities = detect_modalities(case)
    images = case.get("images") or []

    record = {
        "uid": case["_uid"],
        "case_id": case_id,
        "title": case.get("标题", ""),
        "source": case["_source"],
        "modalities": modalities,
        "has_visual_modality": bool(set(modalities) & medical.VISUAL_MODALITIES),
        "image_count": len(images),
        "llm_score": None,
        "llm_reason": None,
        "diagnosis": case.get("诊断结果", ""),
    }

    if use_llm:
        score, llm_modalities, reason = screen_with_llm(case, dry_run=dry_run, model=model)
        record["llm_score"] = score
        record["llm_reason"] = reason
        # The model's own modality judgement wins when it has one.
        if llm_modalities:
            record["llm_modalities"] = llm_modalities

    return record


def main():
    parser = argparse.ArgumentParser(description="Screen cases into a candidate pool.")

    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help="case file to read; repeatable, defaults to results/case_with_images.json",
    )
    parser.add_argument("--include-text-only", action="store_true",
                        help="also screen cases without images")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="JSONL output")
    parser.add_argument("--use-llm", action="store_true", help="run the 0-3 scoring pass")
    parser.add_argument("--limit", type=int, default=None, help="stop after N cases")
    parser.add_argument("--workers", type=int, default=4, help="parallel LLM calls")
    parser.add_argument("--model", default=None, help="model id (defaults to LLM_MODEL)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the prompt instead of calling the model")
    parser.add_argument("--no-resume", action="store_true", help="ignore existing output")

    args = parser.parse_args()

    inputs = [Path(p) for p in args.input] if args.input else list(DEFAULT_INPUTS)

    if args.include_text_only and TEXT_ONLY_INPUT.exists():
        inputs.append(TEXT_ONLY_INPUT)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.no_resume and output.exists():
        output.unlink()

    done = load_done_ids(output)
    cases = load_cases(inputs)

    print(f"loaded {len(cases)} cases from {len(inputs)} file(s), {len(done)} already screened")

    pending = [case for case in cases if case["_uid"] not in done]

    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"pending: {len(pending)}")

    if args.use_llm and not args.dry_run and not llm.is_configured():
        raise SystemExit(
            "no API key: set LLM_API_KEY (or DEEPSEEK_API_KEY), "
            "or drop --use-llm to run the rule pass only"
        )

    def work(case):
        """Screen one case; a failure is reported instead of killing the run."""

        try:
            return build_record(case, use_llm=args.use_llm, dry_run=args.dry_run, model=args.model)
        except Exception as error:
            # The record is not written, so the next run retries this case
            # instead of silently keeping an unscored one.
            return {"uid": case["_uid"], "error": str(error)[:200]}

    written = 0
    failed = 0

    def persist(record):
        """Write a record, or count it as failed."""

        nonlocal written, failed

        if "error" in record:
            failed += 1
            print(f"  [FAIL] {record['uid']}: {record['error']}")
            return

        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        written += 1

        if written % 50 == 0:
            print(f"  screened {written}/{len(pending)}", flush=True)

    with open(output, "a", encoding="utf-8") as f:

        if args.workers > 1 and args.use_llm and not args.dry_run:

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(work, case): case for case in pending}

                for future in as_completed(futures):
                    persist(future.result())

        else:
            for case in pending:
                persist(work(case))

    print(f"wrote {written} records -> {output}" + (f" | failed: {failed}" if failed else ""))

    summarise(output)


def summarise(path):
    """Print a short report over the candidate file."""

    total = 0
    with_images = 0
    visual = 0
    scored = 0
    by_score = {}
    by_modality = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            with_images += 1 if record.get("image_count") else 0
            visual += 1 if record.get("has_visual_modality") else 0

            score = record.get("llm_score")
            if score is not None:
                scored += 1
                by_score[score] = by_score.get(score, 0) + 1

            for modality in record.get("modalities") or []:
                by_modality[modality] = by_modality.get(modality, 0) + 1

    print(f"\nrecords: {total} | with images: {with_images} | visual modality: {visual}")

    if scored:
        print(f"scored: {scored} | by score: {dict(sorted(by_score.items()))}")
        print(f"candidates (score >= 2): {sum(v for k, v in by_score.items() if k >= 2)}")

    print("modalities:", dict(sorted(by_modality.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
