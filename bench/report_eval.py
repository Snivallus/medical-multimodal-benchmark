"""Aggregate model runs against the dataset version they evaluated.

By default, read bench/results/_v2/<model>/{items,conflicts}.jsonl.
Pass --version v1 to reproduce the archived report against its frozen Git
snapshot and matching text audit. Only an audit for the selected item set
can supply a T3 subgroup.
"""

import argparse
import json
import random
import re
import subprocess
from collections import defaultdict
from hashlib import sha256
from math import comb, fsum, isclose
from pathlib import Path

import medical

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH = REPO_ROOT / "bench"
ITEMS = BENCH / "data" / "items.jsonl"
CONFLICTS = BENCH / "data" / "conflict_items.jsonl"
AUDIT = BENCH / "TEXT_AUDIT.md"
ARCHIVE_RESULTS = BENCH / "results"
RESULTS = ARCHIVE_RESULTS / "_v2"
ARCHIVE_REV = "12b40fe6c67b6c71df100fca3d47bb5c39a38961"

VARIANTS = ["photo", "text_b", "text_a"]
CONFLICT_LABELS = ["支持", "反驳", "信息不足"]
CONFLICT_COLUMNS = CONFLICT_LABELS + ["未解析"]
BOOTSTRAP_ROUNDS = 1000
CI = 2.5


def parse_jsonl(text, source):
    rows = []

    for number, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source}:{number}: invalid JSON: {error.msg}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{source}:{number}: expected a JSON object")
        rows.append(row)

    return rows


def load_jsonl(path):
    if not path.exists():
        return []
    return parse_jsonl(path.read_text(encoding="utf-8"), path)


def reference_bytes(path, override, version):
    if override is not None or version == "v2":
        source = override if override is not None else path
        return source.read_bytes(), str(source)

    relative = path.relative_to(REPO_ROOT).as_posix()
    revision_path = f"{ARCHIVE_REV}:{relative}"
    try:
        contents = subprocess.run(
            ["git", "show", revision_path],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise ValueError(
            f"cannot read archived v1 file {relative} from {ARCHIVE_REV}. "
            f"Fetch the source commit with `git fetch origin {ARCHIVE_REV}`, "
            "or supply the archived files with --items, --conflicts and --audit."
        ) from error
    return contents, revision_path


def load_reference(path, override, version, id_key):
    contents, source = reference_bytes(path, override, version)
    rows = parse_jsonl(contents.decode("utf-8"), source)
    by_id = {row[id_key]: row for row in rows}
    if not by_id or len(by_id) != len(rows):
        raise ValueError(f"{source}: expected nonempty data with unique {id_key} values")
    return by_id, sha256(contents).hexdigest()


def load_levels(contents, items):
    """Read levels only when the entire audit matches the chosen item set."""
    matches = re.findall(
        r"^\| `(iiyi-[0-9a-z-]+)` \| (\d+) \| (T[123]|BROKEN) \|",
        contents.decode("utf-8"),
        re.M,
    )
    levels = {item_id: level for item_id, _, level in matches}
    if len(levels) != len(matches) or levels.keys() != items.keys():
        missing = sorted(items.keys() - levels.keys())
        extra = sorted(levels.keys() - items.keys())
        raise ValueError(f"audit does not match item IDs (missing: {missing[:5]}, extra: {extra[:5]})")
    for item_id, size, _ in matches:
        if int(size) != len(items[item_id]["correct_ids"]):
            raise ValueError(f"audit answer count does not match the data for {item_id}")
    return levels


def check_provenance(row, version, digest, custom, source, row_id):
    row_version = row.get("dataset_version")
    row_digest = row.get("dataset_sha256")
    if row_version is None and row_digest is None and version == "v1":
        return  # Archived evaluations predate provenance fields.
    version_matches = row_version == version or (custom and row_version == "custom")
    if row_digest != digest or not version_matches:
        raise ValueError(
            f"{source}: {row_id} was not evaluated against the selected {version} data. "
            "Select the matching dataset and results directory, or rerun the evaluation."
        )


def check_item_results(records, items, version, digest, custom, source):
    for row in records:
        item_id = row["item_id"]
        check_provenance(row, version, digest, custom, source, item_id)
        item = items.get(item_id)
        if item is None or row["case_id"] != item["case_id"] or row["variant"] not in VARIANTS:
            raise ValueError(f"{source}: {item_id} does not match the selected diagnosis dataset")
        selected = row["selected"]
        options = {option["id"]: option["text"] for option in item["options"]}
        main_id = next(key for key, value in options.items() if value == item["main_diagnosis"])
        if (
            not set(selected) <= options.keys()
            or not isclose(row["score"], medical.score_selection(item["correct_ids"], selected))
            or row["main_hit"] != medical.main_diagnosis_hit(main_id, selected)
        ):
            raise ValueError(f"{source}: {item_id} score does not match the selected answer key")


def check_conflict_results(records, conflicts, version, digest, custom, source):
    for row in records:
        row_id = row["conflict_id"]
        check_provenance(row, version, digest, custom, source, row_id)
        if row.get("input_mode", "photo") != "photo":
            raise ValueError(f"{source}: {row_id} is a text-only control, not an image-based conflict result")
        conflict = conflicts.get(row_id)
        if (
            conflict is None
            or row["source_item"] != conflict["source_item"]
            or row["label"] != conflict["label"]
            or row["correct"] != (row["predicted_label"] == conflict["label"])
        ):
            raise ValueError(f"{source}: {row_id} does not match the selected conflict dataset")


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
    return fsum(values) / len(values) if values else 0.0


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


def summarise_items(records, levels):
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
    p_value, positive, negative = sign_test(deltas)

    t3_deltas = [diff for item_id, diff in pairs if levels.get(item_id) == "T3"]

    return {
        "variants": variants,
        "pairs": pairs,
        "delta": mean(deltas) if deltas else None,
        "ci": bootstrap_ci(deltas) if deltas else None,
        "p": p_value,
        "positive": positive,
        "zero": sum(1 for value in deltas if value == 0),
        "negative": negative,
        "t3_n": len(t3_deltas),
        "t3_delta": mean(t3_deltas) if t3_deltas else None,
        "t3_ci": bootstrap_ci(t3_deltas) if t3_deltas else None,
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
        if path.is_dir()
        and not path.name.startswith("_")
        and any((path / f"{mode}.jsonl").is_file() for mode in ("items", "conflicts"))
    )


def fmt(value):
    return f"{value:.3f}" if value is not None else "n/a"


def signed(value):
    return f"{value:+.3f}" if value is not None else "n/a"


def interval(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci is not None else "n/a"


def main():
    parser = argparse.ArgumentParser(description="Aggregate the VLM evaluation runs.")
    parser.add_argument("--version", choices=("v1", "v2"), default="v2", help="dataset version (default: v2)")
    parser.add_argument("--results-dir", type=Path, help="directory containing one subdirectory per model")
    parser.add_argument("--out", type=Path, help="Markdown output (default: <results-dir>/summary.md)")
    parser.add_argument("--items", type=Path, help="override the selected version's diagnostic items JSONL")
    parser.add_argument("--conflicts", type=Path, help="override the selected version's conflict items JSONL")
    parser.add_argument("--audit", type=Path, help="matching text audit Markdown (default: v1 archive only)")
    args = parser.parse_args()

    results_dir = args.results_dir or (ARCHIVE_RESULTS if args.version == "v1" else RESULTS)
    output = args.out or (results_dir / "summary.md")
    try:
        items, item_digest = load_reference(ITEMS, args.items, args.version, "item_id")
        conflict_items, conflict_digest = load_reference(
            CONFLICTS, args.conflicts, args.version, "conflict_id"
        )
        levels = {}
        if args.audit is not None or args.version == "v1":
            audit_bytes, _ = reference_bytes(AUDIT, args.audit, args.version)
            levels = load_levels(audit_bytes, items)
    except (KeyError, ValueError, OSError) as error:
        parser.error(str(error))

    lines = ["# 评测汇总", ""]
    overview = []

    for directory in model_dirs(results_dir):
        name = directory.name
        item_path = directory / "items.jsonl"
        conflict_path = directory / "conflicts.jsonl"
        try:
            records = load_jsonl(item_path)
            conflicts = load_jsonl(conflict_path)
            check_item_results(
                records, items, args.version, item_digest, args.items is not None, item_path
            )
            check_conflict_results(
                conflicts,
                conflict_items,
                args.version,
                conflict_digest,
                args.conflicts is not None,
                conflict_path,
            )
        except (KeyError, ValueError, OSError) as error:
            parser.error(str(error))
        records = dedup(records, lambda r: (r["item_id"], r["variant"]))
        conflicts = dedup(conflicts, lambda r: r["conflict_id"])

        if not records and not conflicts:
            continue

        metrics = summarise_items(records, levels)
        conflict_metrics = summarise_conflicts(conflicts) if conflicts else None

        variants = metrics["variants"]

        print(
            f"[{name}] photo={fmt(variants.get('photo', {}).get('score'))} "
            f"text_b={fmt(variants.get('text_b', {}).get('score'))} "
            f"delta={signed(metrics['delta'])}"
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

        lines.append("")
        if metrics["pairs"]:
            lines += [
                f"- $\\Delta\\mathrm{{acc}}$ (`photo` - `text_b`): **{signed(metrics['delta'])}** "
                f"(95% CI {interval(metrics['ci'])}, sign test $p$ = {metrics['p']:.3f})",
                f"- $\\Delta > 0$: {metrics['positive']} | $= 0$: {metrics['zero']} | "
                f"$< 0$: {metrics['negative']} (n = {len(metrics['pairs'])})",
            ]
        else:
            lines.append("- `photo` 与 `text_b` 的配对差值不可用（无可评分配对）。")
        if metrics["t3_n"]:
            lines.append(
                f"- T3 子集: $\\Delta\\mathrm{{acc}}$ = {signed(metrics['t3_delta'])} "
                f"(95% CI {interval(metrics['t3_ci'])}, n = {metrics['t3_n']})"
            )
        elif levels:
            lines.append("- T3 子集不可用（无可评分的 T3 配对）。")
        else:
            lines.append("- T3 子集不可用（未提供与本版对应的文本充分性审查）。")
        lines.append("")

        if metrics["pairs"]:
            text_b = {
                record["item_id"]: record["score"]
                for record in records
                if record["variant"] == "text_b" and not record.get("excluded")
            }
            modality_rows, difficulty_rows, length_rows = stratified(
                items.values(), metrics["pairs"], text_b
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

    if not overview:
        parser.error(f"no evaluation rows under {results_dir}")

    lines += [
        "## 横向对比",
        "",
        "| 模型 | `text_b` | `photo` | `text_a` | $\\Delta\\mathrm{acc}$ | 95% CI | T3 $\\Delta$ | 冲突 acc |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]

    for row in overview:
        variants = row["variants"]
        text_b = variants.get("text_b", {}).get("score")
        photo = variants.get("photo", {}).get("score")
        text_a = variants.get("text_a", {}).get("score")
        conflict = fmt(row["conflict"]["accuracy"]) if row["conflict"] else "n/a"
        t3 = f"{signed(row['t3'][1])} (n={row['t3'][0]})" if row["t3"][0] else "n/a"
        lines.append(
            f"| {row['name']} | {fmt(text_b)} | {fmt(photo)} | {fmt(text_a)} | "
            f"{signed(row['delta'])} | {interval(row['ci'])} | {t3} | {conflict} |"
        )

    lines.append("")

    text = "\n".join(lines)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8", newline="\n")

    print(f"\nsummary -> {output}")


if __name__ == "__main__":
    main()
