"""Check the frozen cross-case conflict set and its text-only control."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
LABELS = ("支持", "反驳", "信息不足")


def read_rows(data, filename):
    with (data / filename).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_pairs(data):
    items = read_rows(data, "items.jsonl")
    cases = read_rows(data, "cases.jsonl")
    conflicts = read_rows(data, "conflict_items.jsonl")
    by_id = {item["item_id"]: item for item in items}
    sources = {case["case_idx"]: case for case in cases}
    require(len(items) == len(by_id) == len(cases) == len(sources) == 100,
            "expected 100 distinct questions and source cases")
    require({item["case_id"] for item in items} == set(sources),
            "question and source case IDs differ")
    for item in items:
        item_id = item["item_id"]
        case = sources[item["case_id"]]
        require(item["source_url"] == case["url"] and item["images"] == case["images"],
                f"{item_id}: source record and question disagree")
        options = item["options"]
        scored_indices = [option["id"] for option in options if option["correct"]]
        scored_diagnoses = {option["text"] for option in options if option["correct"]}
        require(
            [option["id"] for option in options] == list("ABCDEF"[:len(options)])
            and len(options) == len(scored_indices) + 3
            and len(options) == len({option["text"] for option in options})
            and scored_indices == item["correct_ids"]
            and scored_diagnoses == set(item["diagnoses"])
            and item["main_diagnosis"] in scored_diagnoses,
            f"{item_id}: invalid options or gold answers",
        )
        require(item.get("trap_types"), f"{item_id}: missing trap types")

    require(len(conflicts) == len({row["conflict_id"] for row in conflicts}) == 300,
            "expected 300 distinct conflict questions")
    groups = defaultdict(list)
    image_root = (data / "images").resolve()
    for row in conflicts:
        label, row_id = row["label"], row["conflict_id"]
        anchor = by_id[row["text_item"]]
        image_case = row["case_id"]
        require(row["text_case_id"] == anchor["case_id"] and row["stem"] == anchor["stem"],
                f"{row_id}: text provenance does not match its diagnosis question")
        require(label in LABELS and row["answer"] == label and row["options"] == list(LABELS),
                f"{row_id}: invalid label")
        require(bool(row["claimed_diagnosis"].strip() and row["rationale"].strip()),
                f"{row_id}: missing claim or evidence")
        require((image_case == row["text_case_id"]) == (label == "支持"),
                f"{row_id}: support must use its own image and other labels must swap cases")
        require(row["source_uid"] == f"with_images:{image_case}",
                f"{row_id}: image source UID mismatch")
        image_item = by_id.get(row["source_item"])
        if image_item:
            require(
                image_case == image_item["case_id"]
                and row["image_source_url"] == image_item["source_url"]
                and set(row["images"]) <= set(image_item["images"]),
                f"{row_id}: image provenance does not match its diagnosis question",
            )
        else:
            require(
                label != "支持"
                and row["source_item"] == f"iiyi-corpus-{image_case:04d}"
                and bool(row["image_source_url"])
                and all(p.startswith(f"conflict_sources/case_{image_case:04d}/") for p in row["images"]),
                f"{row_id}: supplemental image has no matching source",
            )
        require(1 <= len(row["images"]) <= 4, f"{row_id}: invalid image count")
        for path in row["images"]:
            image = (image_root / path).resolve()
            require(image.is_relative_to(image_root) and image.is_file(),
                    f"{row_id}: missing image {path}")
        groups[row["pair_id"]].append(row)

    require(len(groups) == 100, "expected 100 matched groups")
    text_labels = defaultdict(Counter)
    for pair_id, group in groups.items():
        require(Counter(row["label"] for row in group) == Counter(LABELS),
                f"{pair_id}: expected one record per label")
        require(len({row["case_id"] for row in group}) == len(LABELS),
                f"{pair_id}: expected three distinct image cases")
        text_keys = {(row["stem"], row["claimed_diagnosis"]) for row in group}
        require(len(text_keys) == 1, f"{pair_id}: text varies with the label")
        text_labels[text_keys.pop()].update(row["label"] for row in group)

    best_text_only = sum(max(counts.values()) for counts in text_labels.values())
    text_cases = {row["text_item"] for row in conflicts}
    require(text_cases == set(by_id), "not every diagnosis question has a matched group")
    print(f"100 diagnosis questions · 300 conflict questions · {len(text_cases)} text cases")
    print(f"Best deterministic text-only score: {best_text_only}/{len(conflicts)} "
          f"({best_text_only / len(conflicts):.1%})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    try:
        check_pairs(args.data)
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f"Conflict pair check failed: {error}") from error


if __name__ == "__main__":
    main()
