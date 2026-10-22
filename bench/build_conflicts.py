"""Stage 3: build the 图文冲突 (selective modality shifting) items.

Each item yields three variants that share its image set, so only the pairing
between image and text changes:

* **支持** — the item's own text and its own diagnosis; the image should support it.
* **反驳** — another case's text, whose diagnosis is a near miss of this case's
  diagnosis (same organ system, different disease), so the image contradicts the
  claimed diagnosis.
* **信息不足** — another case's text from a different organ system, so the image
  neither supports nor contradicts the claimed diagnosis.

The partner text is assembled from the partner case's history and examination
(``辅助检查`` is left out: it holds that case's own imaging report, which would be
a second, unrelated image description in the prompt). The claimed diagnosis is
stored next to the text, because the evaluator has to tell the model which
diagnosis the text stands for.

The label is fixed by how the pair was constructed, not guessed by a model. The
pairing is combinatorial, so every record still needs human review.

Usage::

    python bench/build_conflicts.py            # after build_items.py
"""

import argparse
import json
from pathlib import Path

import medical
import retrieval

REPO_ROOT = Path(__file__).resolve().parent.parent
ITEMS = REPO_ROOT / "bench" / "data" / "items.jsonl"
CORPUS = REPO_ROOT / "iiyi_dataset" / "results" / "case_with_images.json"
OUTPUT = REPO_ROOT / "bench" / "data" / "conflict_items.jsonl"

# Text fields of a partner case that go into the prompt.
PARTNER_FIELDS = ["主诉", "现病史", "既往史", "个人史", "家族史", "查体"]

# A distractor partner must be this similar to count as "same organ system";
# below it the pair would really be 信息不足, not 反驳.
MIN_CONTRADICTION_SIMILARITY = 0.15

# The 信息不足 partner must be this unrelated.
MAX_UNRELATED_SIMILARITY = 0.15


def load_items(path):
    """Load the built items."""

    items = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    return items


def load_corpus():
    """Load the case corpus with parsed diagnoses and a renderable text."""

    with open(CORPUS, encoding="utf-8") as f:
        cases = json.load(f)["cases"]

    for case in cases:
        diagnoses = medical.parse_diagnoses(case.get("诊断结果", ""))
        answers, _ = medical.split_answer_set(diagnoses)
        case["_answers"] = answers or diagnoses
        case["_text"] = "\n".join(
            part
            for part in (
                f"【{field}】{medical.normalize_whitespace(str(case.get(field, '')))}"
                for field in PARTNER_FIELDS
                if medical.normalize_whitespace(str(case.get(field, "")))
            )
        )

    return cases


def nearest_case(query_diagnoses, cases, pool_names):
    """Find a corpus case whose diagnosis is a same-system near miss.

    Candidates come from the retrieval pool first (already curated as near
    misses); when nothing in the pool matches a corpus case, the closest
    different diagnosis in the corpus is used.
    """

    query = set(retrieval.normalize(name) for name in query_diagnoses)

    wanted = {retrieval.normalize(name) for name in pool_names} - query

    for case in cases:
        for name in case["_answers"]:
            if retrieval.normalize(name) in wanted:
                return case, name

    best = None
    best_name = None
    best_score = MIN_CONTRADICTION_SIMILARITY

    for case in cases:
        for name in case["_answers"]:
            if retrieval.normalize(name) in query:
                continue
            if retrieval.is_too_close(name, query_diagnoses):
                continue

            score = max(retrieval.similarity(query_name, name) for query_name in query_diagnoses)

            if score > best_score:
                best, best_name, best_score = case, name, score

    return best, best_name


def unrelated_case(query_diagnoses, cases):
    """Find a corpus case from a different organ system."""

    best = None
    best_name = None
    best_score = 1.0

    for case in cases:
        for name in case["_answers"]:

            if retrieval.is_too_close(name, query_diagnoses):
                continue

            score = max(retrieval.similarity(query_name, name) for query_name in query_diagnoses)

            if score < best_score:
                best, best_name, best_score = case, name, score

    if best_score <= MAX_UNRELATED_SIMILARITY:
        return best, best_name

    return None, None


def make_variant(item, text, claimed, label, rationale):
    """Build one conflict record sharing the item's images."""

    return {
        "conflict_id": "",
        "source_item": item["item_id"],
        "source_uid": item["uid"],
        "case_id": item["case_id"],
        "images": item["images"],
        "stem": text,
        "claimed_diagnosis": claimed,
        "label": label,
        "rationale": rationale,
        "options": ["支持", "反驳", "信息不足"],
        "answer": label,
        "status": "needs_review",
    }


def main():
    parser = argparse.ArgumentParser(description="Build 图文冲突 items.")
    parser.add_argument("--items", default=str(ITEMS), help="built items JSONL")
    parser.add_argument("--output", default=str(OUTPUT), help="conflict items JSONL")
    args = parser.parse_args()

    items = load_items(args.items)

    if not items:
        raise SystemExit(f"no items found in {args.items}; run bench/build_items.py first")

    cases = load_corpus()

    records = []
    counts = {"支持": 0, "反驳": 0, "信息不足": 0}
    skipped = {"反驳": 0, "信息不足": 0}

    for item in items:

        # 支持: the item's own text and its own diagnosis.
        records.append(
            make_variant(
                item,
                item["stem"],
                item["main_diagnosis"],
                "支持",
                "影像与文本来自同一病例",
            )
        )
        counts["支持"] += 1

        # 反驳: a same-system case claiming a near-miss diagnosis.
        partner, claimed = nearest_case(
            item["diagnoses"],
            cases,
            item.get("distractor_pool") or [],
        )

        if partner is None:
            skipped["反驳"] += 1
        else:
            records.append(
                make_variant(
                    item,
                    partner["_text"],
                    claimed,
                    "反驳",
                    f"影像属于「{item['main_diagnosis']}」，文本却主张「{claimed}」",
                )
            )
            counts["反驳"] += 1

        # 信息不足: a case from an unrelated system.
        other, claimed = unrelated_case(item["diagnoses"], cases)

        if other is None:
            skipped["信息不足"] += 1
        else:
            records.append(
                make_variant(
                    item,
                    other["_text"],
                    claimed,
                    "信息不足",
                    f"影像与文本分属不同系统（{item['main_diagnosis']} vs {claimed}）",
                )
            )
            counts["信息不足"] += 1

    for index, record in enumerate(records):
        record["conflict_id"] = f"conf-{index:04d}"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"items: {len(items)} -> conflict records: {len(records)} -> {output}")
    print("labels:", counts)
    print("skipped pairs:", skipped)


if __name__ == "__main__":
    main()
