"""Evaluate a multimodal model on the built items.

Three variants are derived from each item, so the same item is asked in the same
way and only the evidence changes:

* ``photo``   — images + de-identified text (the main condition)
* ``text_b``  — de-identified text only (masked images, the 主指标 control)
* ``text_a``  — the original case text with its imaging conclusions intact, images
  masked (the strict, proposal-literal control)

Per item and variant the model selects one or more option ids; the score is the
partial-credit rule from :func:`medical.score_selection`. The headline number is

    delta_acc_i = score(photo) - score(text_b)

averaged over items, plus the main-diagnosis hit rate.

Any OpenAI-compatible vision endpoint works, so the same script runs against a
local vLLM server and against a hosted API:

    python bench/evaluate.py --model Qwen3-VL-30B-A3B-Instruct-A --limit 10
    python bench/evaluate.py --base-url http://localhost:8000/v1 --model Qwen/Qwen3-VL-8B-Instruct
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path

from PIL import Image

import llm
import medical

REPO_ROOT = Path(__file__).resolve().parent.parent
ITEMS = REPO_ROOT / "bench" / "data" / "items.jsonl"
CONFLICTS = REPO_ROOT / "bench" / "data" / "conflict_items.jsonl"
IMAGE_ROOT = REPO_ROOT / "bench" / "data" / "images"
# Optional per-run pixel cap. GLM-4.1V on vLLM 0.11 hangs while encoding
# images above roughly 4096 visual tokens; a cap keeps that model usable.
IMAGE_MAX_PIXELS = None
# ``text_a`` needs the verbatim case text. The frozen export of the source
# records is what ships in the repository, so the crawl corpus is never required
# to reproduce the benchmark on another machine.
CASES = REPO_ROOT / "bench" / "data" / "cases.jsonl"
RESULTS = REPO_ROOT / "bench" / "results" / "eval_results.jsonl"

# Fields that make up the original, verbatim case text of variant ``text_a``.
ORIGINAL_FIELDS = ["主诉", "现病史", "既往史", "个人史", "家族史", "查体", "辅助检查"]

SYSTEM_PROMPT = "你是临床医生，正在参加一次病例诊断测试。只输出 JSON。"

PROMPT = """请根据下面提供的信息，选出所有成立的诊断（可多选，至少选一个）。

【病例信息】
{stem}

【选项】
{options}

只输出 JSON：{{"selected": ["A"], "reason": "一句话"}}
"""

CONFLICT_PROMPT = """下面是一段病例文本及其给出的诊断，以及配给这段文本的医学影像。

请判断：这些影像是否支持文本给出的诊断？

A = 支持：影像与文本的诊断一致
B = 反驳：影像与文本的诊断相互矛盾
C = 信息不足：仅凭这些影像无法判断文本的诊断是否成立

只输出 JSON：{{"verdict": "A", "why": "20 字内"}}

【文本给出的诊断】{claimed}

【病例文本】
{stem}
"""

# The conflict task uses letters; the dataset stores the Chinese label.
CONFLICT_LABELS = {"A": "支持", "B": "反驳", "C": "信息不足"}
CONFLICT_LETTERS = {label: letter for letter, label in CONFLICT_LABELS.items()}


def load_items(path, limit=None, exclude=None):
    """Load items from the JSONL file, optionally skipping some ids.

    ``exclude`` drops a few ids from a run without touching the dataset or the
    manifest, which is handy for a one-off sanity check.
    """

    items = []
    skipped = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            if exclude and item["item_id"] in exclude:
                skipped += 1
                continue

            items.append(item)

            if limit is not None and len(items) >= limit:
                break

    if skipped:
        print(f"[info] excluded {skipped} item(s)", flush=True)

    return items


def read_jsonl(path):
    """Load a results file, ignoring a torn last line from an interrupted run."""

    rows = []

    if not Path(path).exists():
        return rows

    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    f"[warn] {path}:{number} is not valid JSON; skipped "
                    "(a killed run can leave a half-written line)",
                    flush=True,
                )

    return rows


def write_jsonl(path, rows):
    """Rewrite a results file atomically.

    A plain ``open(path, "w")`` truncates the file before writing; a crash in
    the middle would destroy the rows that ``--resume`` is supposed to keep.
    """

    path = Path(path)
    temporary = Path(str(path) + ".tmp")

    with open(temporary, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

    os.replace(temporary, path)


def ensure_trailing_newline(path):
    """Terminate a torn last line before appending a resumed record.

    ``read_jsonl`` skips the fragment, but opening the same file in append mode
    would glue the next JSON object onto it, creating a second invalid line and
    losing the newly completed answer if the run is killed again.
    """

    path = Path(path)

    if not path.exists() or path.stat().st_size == 0:
        return

    with open(path, "rb+") as handle:
        handle.seek(-1, os.SEEK_END)

        if handle.read(1) != b"\n":
            handle.write(b"\n")


def load_corpus_text():
    """Return ``{case_idx: original case text}`` for variant ``text_a``.

    Reads the frozen ``data/cases.jsonl`` export rather than the crawl corpus:
    the corpus is not shipped, and a missing one used to degrade ``text_a`` into
    ``text_b`` silently, which quietly invalidates the control.
    """

    texts = {}

    if not CASES.exists():
        return texts

    with open(CASES, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            case = json.loads(line)
            parts = []

            for field in ORIGINAL_FIELDS:
                value = medical.normalize_whitespace(str(case.get(field, "")))
                if value:
                    parts.append(f"【{field}】{value}")

            texts[case["case_idx"]] = "\n".join(parts)

    return texts


def image_mime(data, path):
    """Guess the image type from the file signature, not from the extension.

    The crawl output stores everything as ``.jpg`` even when the bytes are PNG
    (129 of the 154 shipped images), and a data URL with the wrong type is
    decoded differently — or rejected — by a strict inference server.
    """

    if data.startswith(b"\xff\xd8"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"GIF8"):
        return "gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    if data.startswith(b"BM"):
        return "bmp"

    suffix = Path(path).suffix.lower().lstrip(".") or "jpeg"

    return "jpeg" if suffix in ("jpg", "jpeg") else suffix


def image_payload(paths, image_root=None):
    """Encode images as data URLs for an OpenAI-compatible vision request."""

    root = Path(image_root) if image_root else IMAGE_ROOT
    payload = []

    for relative in paths:

        file_path = root / relative

        if not file_path.exists():
            print(f"[warn] missing image: {file_path}")
            continue

        raw = file_path.read_bytes()
        mime = image_mime(raw[:16], file_path)

        if IMAGE_MAX_PIXELS:
            with Image.open(BytesIO(raw)) as image:
                width, height = image.size
                pixels = width * height

                if pixels > IMAGE_MAX_PIXELS:
                    scale = (IMAGE_MAX_PIXELS / pixels) ** 0.5
                    size = (max(1, round(width * scale)), max(1, round(height * scale)))
                    resized = image.convert("RGB").resize(size, Image.LANCZOS)
                    buffer = BytesIO()
                    resized.save(buffer, format="JPEG", quality=95)
                    raw = buffer.getvalue()
                    mime = "jpeg"

        data = base64.b64encode(raw).decode("ascii")

        payload.append({"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{data}"}})

    return payload


def build_variant_messages(item, variant, original_text):
    """Build the chat messages for one variant."""

    options = "\n".join(f"{option['id']}. {option['text']}" for option in item["options"])

    if variant == "text_a":
        # The strict control: the original text, imaging conclusions included.
        stem = original_text or item["stem"]
    else:
        stem = item["stem"]

    text = PROMPT.format(stem=stem, options=options)

    if variant == "photo":
        content = [{"type": "text", "text": text}]
        content += image_payload(item["images"])
    else:
        content = [{"type": "text", "text": text}]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


THINK_OPEN_RE = re.compile(r"<think\b", re.I)
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.S | re.I)
ANSWER_BLOCK_RE = re.compile(r"<answer\b[^>]*>(.*?)</answer\s*>", re.S | re.I)


def visible_answer(text):
    """Drop a thinking block and keep the model's final answer.

    GLM-4.1V-Thinking writes ``<think>...</think><answer>{...}</answer>``. Its
    reasoning often quotes the prompt's JSON example, so parsing the raw reply
    can score that example instead of the actual answer. An opening ``<think>``
    without a closing tag means the reply was cut off before the answer.
    """

    text = text or ""

    if THINK_OPEN_RE.search(text) and not re.search(r"</think\s*>", text, re.I):
        return ""

    text = THINK_BLOCK_RE.sub("", text)
    match = ANSWER_BLOCK_RE.search(text)

    if match:
        return match.group(1)

    return text


def parse_selection(text):
    """Extract the selected option ids from the model answer."""

    text = visible_answer(text)
    data = llm.extract_json(text, last=True) or {}

    selected = data.get("selected") or data.get("answers") or []

    if isinstance(selected, str):
        selected = [selected]

    cleaned = []

    for value in selected:
        if not isinstance(value, str):
            continue
        # Accept "A", "A.", "A. 右膝…" and multi-letter answers like "A,C".
        for part in value.replace("，", ",").split(","):
            letter = part.strip().strip("。.").upper()[:1]
            # Only real option ids: "中".isalpha() is True, so a bare isalpha()
            # would let non-Latin characters through as answers.
            if letter and letter in "ABCDEF" and letter not in cleaned:
                cleaned.append(letter)

    # A reply without JSON sometimes still starts with a letter list.
    if not cleaned:
        head = (text or "").strip()[:40].upper()

        for part in re.split(r"[,，、]", head):
            letter = part.strip().strip("。.").upper()[:1]

            if letter and letter in "ABCDEF" and letter not in cleaned:
                cleaned.append(letter)

    return cleaned


def parse_verdict(text):
    """Return ``A``/``B``/``C`` for a conflict answer, or ``None``.

    Thinking models put a long chain of thought in front of the answer, so the
    JSON object is preferred. A model that answers with the Chinese label instead
    of the letter is still answering the same question; only a reply without a
    recognisable verdict counts as a parse failure.
    """

    text = visible_answer(text)
    data = llm.extract_json(text, last=True) or {}

    raw = data.get("verdict")

    if raw is not None:
        value = str(raw).strip()

        if value[:1].upper() in CONFLICT_LABELS:
            return value[:1].upper()

        for label, letter in CONFLICT_LETTERS.items():
            if value.startswith(label):
                return letter

        return None

    # No JSON. Accept a bare letter or label only when that is all the answer
    # is: a label mentioned somewhere inside a chain of thought is not a verdict.
    lines = [line.strip() for line in (text or "").strip().splitlines() if line.strip()]

    if lines:
        tail = lines[-1].strip("。.!！ ")

        if tail[:1].upper() in CONFLICT_LABELS:
            return tail[:1].upper()

        for label, letter in CONFLICT_LETTERS.items():
            if tail == label or tail.startswith(label):
                return letter

    return None


def evaluate_item(item, variants, model, original_text, retries=3, max_tokens=256):
    """Evaluate one item on every requested variant."""

    # ``--model`` is optional; the request falls back to LLM_MODEL. Resolve it
    # here too, otherwise every row records ``null`` as the model that produced it.
    model = model or llm.model_name()

    correct_ids = item["correct_ids"]
    valid_ids = {option["id"] for option in item["options"]}

    # The main diagnosis has to be located by text, because option ids are in
    # shuffled order and the first correct id is not necessarily the main one.
    main_option_id = next(
        (
            option["id"]
            for option in item["options"]
            if option["text"] == item["main_diagnosis"]
        ),
        None,
    )

    records = []

    for variant in variants:

        messages = build_variant_messages(item, variant, original_text)
        selected = []
        parsed = []
        answer = None
        finish_reason = None
        error = None
        started = time.time()

        for attempt in range(retries):
            try:
                answer, finish_reason = llm.chat(
                    messages,
                    model=model,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    timeout=(10, 900),
                    # llm.chat already retries transport errors; without this
                    # the 3x3 nesting can hold one request for hours.
                    retries=1,
                    return_meta=True,
                )
                parsed = parse_selection(answer)
                # Ignore ids that do not exist in this item; they would be
                # counted as false positives.
                selected = [option_id for option_id in parsed if option_id in valid_ids]
                error = None
                break
            except Exception as exc:
                error = str(exc)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)

        score = medical.score_selection(correct_ids, selected) if selected else 0.0
        hit = medical.main_diagnosis_hit(main_option_id, selected)

        records.append(
            {
                "item_id": item["item_id"],
                "case_id": item["case_id"],
                "variant": variant,
                "model": model,
                "selected": selected,
                "score": score,
                "main_hit": hit,
                # Raw reply kept for the audit trail: a JSON parse failure is a
                # zero score under the protocol, so the only way to tell it apart
                # from a wrong answer later is to keep what the model said.
                "answer": answer,
                "parse_ok": bool(parsed),
                # "length" means the reply hit --max-tokens. A thinking model
                # can spend the whole budget before it reaches the JSON.
                "finish_reason": finish_reason,
                "truncated": finish_reason == "length",
                "latency_s": round(time.time() - started, 2),
                # A call that never succeeded must not be counted as a wrong
                # answer: it would silently deflate the score of that variant.
                "excluded": error is not None,
                "error": error,
            }
        )

    return records


def summarise(records):
    """Aggregate scores and print the headline numbers."""

    # Failed calls are reported separately, never averaged in.
    failed = sum(1 for record in records if record.get("excluded"))
    records = [record for record in records if not record.get("excluded")]

    by_variant = {}

    for record in records:
        by_variant.setdefault(record["variant"], []).append(record["score"])

    print("\nvariant     n    mean score   main-diagnosis hit")

    for variant, scores in sorted(by_variant.items()):
        hits = [r["main_hit"] for r in records if r["variant"] == variant]
        mean = sum(scores) / len(scores)
        print(f"{variant:<10} {len(scores):3d}    {mean:.3f}          {sum(hits)/len(hits):.3f}")

    if failed:
        print(f"excluded (call failed): {failed}")

    truncated = sum(1 for record in records if record.get("truncated"))

    if truncated:
        print(f"truncated by --max-tokens: {truncated}")

    # Per-item delta between the photo condition and the masked-text condition.
    photo = {r["item_id"]: r["score"] for r in records if r["variant"] == "photo"}
    text = {r["item_id"]: r["score"] for r in records if r["variant"] == "text_b"}

    shared = sorted(set(photo) & set(text))

    if shared:
        deltas = [photo[item_id] - text[item_id] for item_id in shared]
        mean_delta = sum(deltas) / len(deltas)
        positive = sum(1 for d in deltas if d > 0)
        zero = sum(1 for d in deltas if d == 0)
        negative = sum(1 for d in deltas if d < 0)

        print(f"\ndelta_acc (photo - text_b): {mean_delta:+.3f}")
        print(f"items with delta>0: {positive}/{len(shared)} | delta=0: {zero} | delta<0: {negative}")


def evaluate_conflict(record, model, retries=3, max_tokens=8192):
    """Ask one model whether the images support the diagnosis claimed by the text."""

    model = model or llm.model_name()

    content = [
        {
            "type": "text",
            "text": CONFLICT_PROMPT.format(
                stem=record["stem"],
                claimed=record["claimed_diagnosis"],
            ),
        }
    ]
    content += image_payload(record["images"])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    answer = None
    finish_reason = None
    error = None
    started = time.time()

    for attempt in range(retries):
        try:
            answer, finish_reason = llm.chat(
                messages,
                model=model,
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=(10, 900),
                retries=1,
                return_meta=True,
            )
            error = None
            break
        except Exception as exc:
            error = str(exc)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    verdict = parse_verdict(answer) if answer is not None else None

    return {
        "conflict_id": record["conflict_id"],
        "source_item": record["source_item"],
        "label": record["label"],
        "model": model,
        # A reply that cannot be parsed is a wrong answer, not a missing one:
        # dropping it from the denominator would reward a model for ignoring the
        # answer format. Only a failed call is excluded.
        "predicted": verdict,
        "predicted_label": CONFLICT_LABELS.get(verdict),
        "correct": verdict is not None and verdict == CONFLICT_LETTERS.get(record["label"]),
        "answer": answer,
        "parse_ok": verdict is not None,
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
        "latency_s": round(time.time() - started, 2),
        "excluded": error is not None,
        "error": error,
    }


def summarise_conflicts(records):
    """Print accuracy, per-label precision/recall/F1 and the confusion matrix."""

    labels = ["支持", "反驳", "信息不足"]
    # A parse failure has no label; it is a wrong answer and needs its own
    # column so the matrix still adds up to the denominator.
    columns = labels + ["未解析"]
    matrix = {label: {other: 0 for other in columns} for label in labels}
    failed = 0
    scored = 0
    correct = 0

    for record in records:

        if record.get("excluded"):
            failed += 1
            continue

        scored += 1
        predicted = record.get("predicted_label") or "未解析"
        matrix[record["label"]][predicted] += 1
        correct += 1 if record["correct"] else 0

    if not scored:
        print("\nno conflict answer could be scored")
        return

    print(f"\nconflict accuracy: {correct/scored:.3f} ({correct}/{scored})")
    print(f"excluded (call failed): {failed}")

    truncated = sum(
        1 for record in records
        if record.get("truncated") and not record.get("excluded")
    )

    if truncated:
        print(f"truncated by --max-tokens: {truncated}")

    header = "label\\pred  " + "  ".join(f"{label:>6}" for label in columns)
    print("\n" + header)

    for label in labels:
        row = "  ".join(f"{matrix[label][other]:>6}" for other in columns)
        print(f"{label:<10} {row}")

    print("\nlabel       precision  recall     f1")

    for label in labels:
        true_positive = matrix[label][label]
        predicted_positive = sum(matrix[truth][label] for truth in labels)
        actual_positive = sum(matrix[label][other] for other in columns)
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"{label:<10}  {precision:.3f}      {recall:.3f}      {f1:.3f}")


def main():
    global IMAGE_ROOT, IMAGE_MAX_PIXELS

    parser = argparse.ArgumentParser(description="Evaluate a model on the built items.")
    parser.add_argument("--items", default=str(ITEMS), help="items JSONL")
    parser.add_argument("--conflicts", default=None, help="conflict items JSONL (--mode conflicts)")
    parser.add_argument("--output", default=str(RESULTS), help="per-answer results JSONL")
    parser.add_argument("--model", default=None, help="model id (defaults to LLM_MODEL)")
    parser.add_argument("--base-url", default=None, help="override LLM_BASE_URL")
    parser.add_argument("--api-key", default=None, help="override LLM_API_KEY")
    parser.add_argument(
        "--variants",
        default="photo,text_b",
        help="comma separated: photo, text_b, text_a",
    )
    parser.add_argument(
        "--mode",
        default="items",
        choices=["items", "conflicts"],
        help="items: multiple-answer diagnosis task; conflicts: three-class image/text check",
    )
    parser.add_argument("--limit", type=int, default=None, help="evaluate the first N items/records")
    parser.add_argument(
        "--exclude",
        default="",
        help="comma separated item ids to skip",
    )
    parser.add_argument(
        "--image-root",
        default=str(IMAGE_ROOT),
        help="directory holding the case image folders (defaults to bench/data/images)",
    )
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=None,
        help="downscale images above this many pixels before encoding (GLM workaround)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="generation budget per call (raise it for thinking models)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep finished rows in --output and only ask the missing ones",
    )
    args = parser.parse_args()

    IMAGE_ROOT = Path(args.image_root)
    IMAGE_MAX_PIXELS = args.max_image_pixels

    if args.base_url:
        os.environ["LLM_BASE_URL"] = args.base_url
    if args.api_key:
        os.environ["LLM_API_KEY"] = args.api_key

    # Without a key every call fails the same way, and the per-variant retry
    # turns that into a row of zeros marked "excluded" — a missing key looks
    # like a model that scores 0. Refuse to start instead.
    if not llm.is_configured():
        raise SystemExit(
            "no API key: set LLM_API_KEY (or DEEPSEEK_API_KEY), or pass --api-key. "
            "A local server that ignores auth still needs a placeholder value."
        )

    if args.mode == "conflicts":
        return run_conflicts(args)

    variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = set(variants) - {"photo", "text_b", "text_a"}

    if unknown:
        raise SystemExit(f"unknown variant(s): {sorted(unknown)}")

    items = load_items(
        args.items,
        args.limit,
        exclude={name.strip() for name in args.exclude.split(",") if name.strip()},
    )
    originals = load_corpus_text() if "text_a" in variants else {}

    if not items:
        raise SystemExit(f"no items in {args.items}")

    if "text_a" in variants:
        # Without the verbatim text the variant is just text_b under another name,
        # which would look like a valid control in the results file.
        absent = [item["item_id"] for item in items if not originals.get(item["case_id"])]

        if absent:
            raise SystemExit(
                f"text_a needs the original case text, missing for {len(absent)} item(s) "
                f"({', '.join(absent[:5])}); is {CASES.name} present and complete?"
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    existing = read_jsonl(output) if args.resume else []

    if args.resume:
        ensure_trailing_newline(output)

    # Only a successful call is a finished row: a row marked ``excluded`` is a
    # failed call and has to be retried on resume.
    done = {
        (record["item_id"], record["variant"])
        for record in existing
        if not record.get("excluded")
    }

    if existing:
        print(f"[resume] {len(done)} finished (item, variant) pair(s) in {output}")

    all_records = list(existing)
    started = time.time()
    mode = "a" if args.resume else "w"

    with open(output, mode, encoding="utf-8") as handle:

        for index, item in enumerate(items, 1):

            todo = [name for name in variants if (item["item_id"], name) not in done]

            if not todo:
                continue

            records = evaluate_item(
                item,
                todo,
                args.model,
                originals.get(item["case_id"], ""),
                max_tokens=args.max_tokens,
            )

            for record in records:
                all_records.append(record)
                # Flush per answer: a run of several hundred calls should not
                # lose everything to one crash.
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

            flags = " ".join(f"{r['variant']}={r['score']:.2f}" for r in records)
            print(f"[{index:3d}/{len(items)}] {item['item_id']} | {flags}", flush=True)

    # A resumed run can hold two rows for the same key (one retried); the later
    # row wins, but a failed retry must not overwrite an earlier success.
    deduped = {}
    for record in all_records:
        key = (record["item_id"], record["variant"])
        previous = deduped.get(key)

        if previous is not None and not previous.get("excluded") and record.get("excluded"):
            continue

        deduped[key] = record

    all_records = list(deduped.values())

    write_jsonl(output, all_records)

    summarise(all_records)
    print(f"\nresults -> {output} | {time.time() - started:.0f}s")


def run_conflicts(args):
    """Evaluate the three-class 图文冲突 task."""

    path = args.conflicts if args.conflicts else str(CONFLICTS)

    records = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.limit is not None:
        records = records[: args.limit]

    if not records:
        raise SystemExit(f"no conflict records in {path}")

    print(f"conflict records: {len(records)} | model: {args.model or llm.model_name()}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    existing = read_jsonl(output) if args.resume else []

    if args.resume:
        ensure_trailing_newline(output)

    done = {
        record["conflict_id"]
        for record in existing
        if not record.get("excluded")
    }

    if existing:
        print(f"[resume] {len(done)} finished conflict row(s) in {output}")

    results = list(existing)
    started = time.time()
    mode = "a" if args.resume else "w"

    with open(output, mode, encoding="utf-8") as handle:

        for index, record in enumerate(records, 1):

            if record["conflict_id"] in done:
                continue

            result = evaluate_conflict(
                record,
                args.model,
                max_tokens=args.max_tokens,
            )
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()

            if index % 5 == 0 or index == len(records):
                print(
                    f"  [{index}/{len(records)}] {record['conflict_id']} -> {result['predicted']}",
                    flush=True,
                )

    deduped = {}
    for result in results:
        key = result["conflict_id"]
        previous = deduped.get(key)

        if previous is not None and not previous.get("excluded") and result.get("excluded"):
            continue

        deduped[key] = result

    results = list(deduped.values())

    write_jsonl(output, results)

    summarise_conflicts(results)
    print(f"\nresults -> {output} | {time.time() - started:.0f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
