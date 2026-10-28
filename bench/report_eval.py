"""Aggregate the VLM evaluation runs into the tables used by ``RESULTS.md``.

Reads one directory per model::

    bench/results/<model>/items.jsonl
    bench/results/<model>/conflicts.jsonl

and writes a Markdown summary. The statistics are fixed by ``RESULTS.md``:
per-item paired delta between ``photo`` and ``text_b``, a percentile bootstrap
over items (1000 resamples) and an exact paired sign test. Items judged ``T3``
in ``TEXT_AUDIT.md`` (the image is necessary) are reported as a sensitivity
analysis next to the full 84-item run.

    python bench/report_eval.py
    python bench/report_eval.py --results-dir bench/results --out bench/results/summary.md
"""

import argparse
import json
import random
import re
from collections import defaultdict
from math import comb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH = REPO_ROOT / "bench"
ITEMS = BENCH / "data" / "items.jsonl"
CONFLICTS = BENCH / "data" / "conflict_items.jsonl"
AUDIT = BENCH / "TEXT_AUDIT.md"
RESULTS = BENCH / "results"

VARIANTS = ["photo", "text_b", "text_a"]
CONFLICT_LABELS = ["支持", "反驳", "信息不足"]
CONFLICT_COLUMNS = CONFLICT_LABELS + ["未解析"]
BOOTSTRAP_ROUNDS = 1000
CI = 2.5


def load_jsonl(path):
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
                print(f"[warn] {path}:{number} is not valid JSON; skipped")

    return rows


def load_levels():
    """Return ``{item_id: T1|T2|T3}`` parsed from the audit table."""

    text = AUDIT.read_text(encoding="utf-8")
    found = dict(re.findall(r"^\| `(iiyi-[0-9a-z]+)` \| \d+ \| (T[123]|BROKEN) \|", text, re.M))

    return found


def dedup(records, key):
    """Last row wins, except that a failed retry cannot erase a success."""

    table = {}

    for record in records:
        row_key = key(record)
        previous = table.get(row_key)

        if previous is not None and not previous.get("excluded") and record.get("excluded"):
            continue

        table[row_key] = record

    return list(table.values())


def mean(values):
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(values, rounds=BOOTSTRAP_ROUNDS, seed=0):
    """Percentile bootstrap 95% CI for the mean."""

    if not values:
        return 0.0, 0.0

    rng = random.Random(seed)
    size = len(values)
    means = []

    for _ in range(rounds):
        sample = [values[rng.randrange(size)] for _ in range(size)]
        means.append(sum(sample) / size)

    means.sort()

    low = means[int(round(CI / 100 * rounds)) - 1]
    high = means[int(round((100 - CI) / 100 * rounds)) - 1]

    return low, high


def sign_test(values):
    """Exact two-sided sign test over the non-zero paired differences."""

    positive = sum(1 for value in values if value > 0)
    negative = sum(1 for value in values if value < 0)
    total = positive + negative

    if not total:
        return 1.0, positive, negative

    tail = sum(comb(total, index) for index in range(min(positive, negative) + 1))
    p_value = min(1.0, 2 * tail / (2 ** total))

    return p_value, positive, negative


def paired(records, left="photo", right="text_b"):
    """Return ``[(item_id, difference)]`` for items with both variants scored."""

    scores = defaultdict(dict)

    for record in records:
        if record.get("excluded"):
            continue
        scores[record["item_id"]][record["variant"]] = record["score"]

    pairs = []

    for item_id, row in scores.items():
        if left in row and right in row:
            pairs.append((item_id, row[left] - row[right]))

    pairs.sort()

    return pairs


def tertiles(rows):
    """Split ``[(value, item_id)]`` into three roughly equal groups."""

    ordered = sorted(rows)
    size = len(ordered)
    cuts = [0, size // 3, 2 * size // 3, size]

    names = ["low", "mid", "high"]
    groups = []

    for index, name in enumerate(names):
        chunk = ordered[cuts[index]: cuts[index + 1]]
        groups.append((name, chunk))

    return groups


def summarise_items(records, items, levels):
    """Return the item-level metrics for one model."""

    variants = {}

    for variant in VARIANTS:
        rows = [
            record
            for record in records
            if record["variant"] == variant and not record.get("excluded")
        ]

        if not rows:
            continue

        variants[variant] = {
            "n": len(rows),
            "score": mean([record["score"] for record in rows]),
            "main": mean([1.0 if record["main_hit"] else 0.0 for record in rows]),
            "parse_failed": sum(1 for record in rows if record.get("parse_ok") is False),
            "truncated": sum(1 for record in rows if record.get("truncated")),
        }

    pairs = paired(records)
    deltas = [value for _, value in pairs]
    low, high = bootstrap_ci(deltas)
    p_value, positive, negative = sign_test(deltas)

    t3 = {item_id for item_id, diff in pairs if levels.get(item_id) == "T3"}
    t3_deltas = [diff for item_id, diff in pairs if item_id in t3]
    t3_low, t3_high = bootstrap_ci(t3_deltas)

    return {
        "variants": variants,
        "pairs": pairs,
        "delta": mean(deltas),
        "ci": (low, high),
        "p": p_value,
        "positive": positive,
        "zero": sum(1 for value in deltas if value == 0),
        "negative": negative,
        "t3_n": len(t3_deltas),
        "t3_delta": mean(t3_deltas),
        "t3_ci": (t3_low, t3_high),
    }


def summarise_conflicts(records):
    """Return accuracy, per-label F1 and the confusion matrix for one model."""

    matrix = {label: {column: 0 for column in CONFLICT_COLUMNS} for label in CONFLICT_LABELS}
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
        correct += 1 if record.get("correct") else 0

    f1 = {}

    for label in CONFLICT_LABELS:
        true_positive = matrix[label][label]
        predicted_positive = sum(matrix[truth][label] for truth in CONFLICT_LABELS)
        actual_positive = sum(matrix[label][column] for column in CONFLICT_COLUMNS)
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1[label] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )

    return {
        "n": scored,
        "failed": failed,
        "truncated": sum(
            1 for record in records if record.get("truncated") and not record.get("excluded")
        ),
        "accuracy": correct / scored if scored else 0.0,
        "f1": f1,
        "matrix": matrix,
    }


def stratified(items, pairs, variant_scores):
    """Group the paired deltas by modality, text_b difficulty and stem length."""

    by_id = {item["item_id"]: item for item in items}

    modality = defaultdict(list)

    for item_id, diff in pairs:
        for name in by_id[item_id]["modalities"]:
            modality[name].append(diff)

    modality_rows = [
        (name, len(values), mean(values))
        for name, values in sorted(modality.items(), key=lambda row: -len(row[1]))
    ]

    difficulty = tertiles([(variant_scores[item_id], item_id) for item_id, _ in pairs])
    difficulty_rows = []

    for name, chunk in difficulty:
        values = [dict(pairs)[item_id] for _, item_id in chunk]
        difficulty_rows.append((name, len(values), mean(values)))

    length = tertiles([(by_id[item_id]["stem_chars"], item_id) for item_id, _ in pairs])
    length_rows = []

    for name, chunk in length:
        values = [dict(pairs)[item_id] for _, item_id in chunk]
        length_rows.append((name, len(values), mean(values)))

    return modality_rows, difficulty_rows, length_rows


def model_dirs(results_dir):
    root = Path(results_dir)

    if not root.exists():
        return []

    # ``_smoke`` is a scratch directory; it must never leak into the published
    # summary just because it happens to contain an items.jsonl.
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith("_") and (path / "items.jsonl").exists()
    )


def fmt(value):
    return f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser(description="Aggregate the VLM evaluation runs.")
    parser.add_argument("--results-dir", default=str(RESULTS))
    parser.add_argument("--out", default=str(RESULTS / "summary.md"))
    args = parser.parse_args()

    items = load_jsonl(ITEMS)
    levels = load_levels()
    item_ids = {item["item_id"] for item in items}

    if set(levels) != item_ids:
        missing = sorted(item_ids - set(levels))
        print(f"[warn] {len(missing)} item(s) missing a T-level: {missing[:5]}")

    lines = ["# 评测汇总", ""]
    overview = []

    for directory in model_dirs(args.results_dir):
        name = directory.name
        records = dedup(load_jsonl(directory / "items.jsonl"), lambda r: (r["item_id"], r["variant"]))
        conflicts = dedup(load_jsonl(directory / "conflicts.jsonl"), lambda r: r["conflict_id"])

        if not records:
            continue

        metrics = summarise_items(records, items, levels)
        conflict_metrics = summarise_conflicts(conflicts) if conflicts else None

        variants = metrics["variants"]

        print(
            f"[{name}] photo={fmt(variants.get('photo', {}).get('score', 0))} "
            f"text_b={fmt(variants.get('text_b', {}).get('score', 0))} "
            f"delta={metrics['delta']:+.3f}"
        )

        overview.append(
            {
                "name": name,
                "variants": variants,
                "delta": metrics["delta"],
                "ci": metrics["ci"],
                "t3": (metrics["t3_n"], metrics["t3_delta"], metrics["t3_ci"]),
                "conflict": conflict_metrics,
            }
        )

        lines += [f"## {name}", "", "### 主表", ""]
        lines += [
            "| 变体 | n | mean score | 主诊断命中 | 解析失败 | 截断 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]

        for variant in VARIANTS:
            row = variants.get(variant)

            if not row:
                continue

            lines.append(
                f"| `{variant}` | {row['n']} | {fmt(row['score'])} | "
                f"{fmt(row['main'])} | {row['parse_failed']} | {row['truncated']} |"
            )

        lines += [
            "",
            f"- $\\Delta\\mathrm{{acc}}$ (`photo` - `text_b`): **{metrics['delta']:+.3f}** "
            f"(95% CI [{metrics['ci'][0]:+.3f}, {metrics['ci'][1]:+.3f}], "
            f"sign test $p$ = {metrics['p']:.3f})",
            f"- $\\Delta > 0$: {metrics['positive']} | $= 0$: {metrics['zero']} | "
            f"$< 0$: {metrics['negative']} (n = {len(metrics['pairs'])})",
            f"- T3 子集: $\\Delta\\mathrm{{acc}}$ = {metrics['t3_delta']:+.3f} "
            f"(95% CI [{metrics['t3_ci'][0]:+.3f}, {metrics['t3_ci'][1]:+.3f}], "
            f"n = {metrics['t3_n']})",
            "",
        ]

        if metrics["pairs"]:
            text_b = {
                record["item_id"]: record["score"]
                for record in records
                if record["variant"] == "text_b" and not record.get("excluded")
            }
            modality_rows, difficulty_rows, length_rows = stratified(
                items, metrics["pairs"], text_b
            )

            lines += ["### 分层", "", "| 模态 | n | $\\Delta\\mathrm{acc}$ |", "| --- | ---: | ---: |"]

            for label, count, value in modality_rows:
                lines.append(f"| {label} | {count} | {value:+.3f} |")

            lines += [
                "",
                "| `text_b` 难度 | n | $\\Delta\\mathrm{acc}$ |",
                "| --- | ---: | ---: |",
            ]

            for label, count, value in difficulty_rows:
                lines.append(f"| {label} | {count} | {value:+.3f} |")

            lines += [
                "",
                "| 题干长度 | n | $\\Delta\\mathrm{acc}$ |",
                "| --- | ---: | ---: |",
            ]

            for label, count, value in length_rows:
                lines.append(f"| {label} | {count} | {value:+.3f} |")

            lines.append("")

        if conflict_metrics:
            lines += [
                "### 图文冲突",
                "",
                f"accuracy = **{conflict_metrics['accuracy']:.3f}** "
                f"({conflict_metrics['n']} scored, {conflict_metrics['failed']} call failures, "
                f"{conflict_metrics['truncated']} truncated)",
                "",
                "| label | precision | recall | F1 |",
                "| --- | ---: | ---: | ---: |",
            ]

            for label in CONFLICT_LABELS:
                true_positive = conflict_metrics["matrix"][label][label]
                predicted_positive = sum(
                    conflict_metrics["matrix"][truth][label] for truth in CONFLICT_LABELS
                )
                actual_positive = sum(
                    conflict_metrics["matrix"][label][column] for column in CONFLICT_COLUMNS
                )
                precision = true_positive / predicted_positive if predicted_positive else 0.0
                recall = true_positive / actual_positive if actual_positive else 0.0
                lines.append(
                    f"| {label} | {precision:.3f} | {recall:.3f} | "
                    f"{conflict_metrics['f1'][label]:.3f} |"
                )

            lines += ["", "| 真值 \\ 预测 | " + " | ".join(CONFLICT_COLUMNS) + " |",
                      "| --- | " + " | ".join("---:" for _ in CONFLICT_COLUMNS) + " |"]

            for label in CONFLICT_LABELS:
                cells = " | ".join(
                    str(conflict_metrics["matrix"][label][column])
                    for column in CONFLICT_COLUMNS
                )
                lines.append(f"| {label} | {cells} |")

            lines.append("")

    lines += [
        "## 横向对比",
        "",
        "| 模型 | `text_b` | `photo` | `text_a` | $\\Delta\\mathrm{acc}$ | 95% CI | T3 $\\Delta$ | 冲突 acc |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]

    for row in overview:
        variants = row["variants"]
        text_b = variants.get("text_b", {}).get("score", 0.0)
        photo = variants.get("photo", {}).get("score", 0.0)
        text_a = variants.get("text_a", {}).get("score", 0.0)
        conflict = fmt(row["conflict"]["accuracy"]) if row["conflict"] else "n/a"
        lines.append(
            f"| {row['name']} | {fmt(text_b)} | {fmt(photo)} | {fmt(text_a)} | "
            f"{row['delta']:+.3f} | [{row['ci'][0]:+.3f}, {row['ci'][1]:+.3f}] | "
            f"{row['t3'][1]:+.3f} (n={row['t3'][0]}) | {conflict} |"
        )

    lines.append("")

    text = "\n".join(lines)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")

    print(f"\nsummary -> {args.out}")


if __name__ == "__main__":
    main()
