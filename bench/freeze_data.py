"""Package the frozen dataset into ``bench/data/``.

The pipeline writes JSONL into ``bench/data/`` while it runs, but the images live
next to the crawl output (790 MB in total) and nothing records which slice of them
belongs to the shipped items. This script closes that gap and makes the directory
self-contained:

1. every image referenced by ``items.jsonl`` or ``conflict_items.jsonl`` is copied
   from the crawl output into ``bench/data/images/<case>/``;
2. the source case records of the shipped items are exported to ``cases.jsonl``,
   so a reviewer or a report can read the original text without the corpus;
3. ``MANIFEST.json`` records counts, sizes and sha256 of every shipped file, which
   is what tells a fresh server clone that it has the dataset the experiment used.

Re-run it after the human review changes ``items.jsonl``; ``--prune`` additionally
deletes images that are no longer referenced.

Usage::

    python bench/freeze_data.py                    # sync images, write manifest
    python bench/freeze_data.py --prune            # also drop unreferenced images
    python bench/freeze_data.py --dry-run          # report what would change
    python bench/freeze_data.py --verify           # check a fresh clone
"""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "bench" / "data"
CORPUS = REPO_ROOT / "iiyi_dataset" / "results" / "case_with_images.json"
IMAGE_ROOT = REPO_ROOT / "iiyi_dataset" / "results" / "images"

# Files of the frozen dataset, in manifest order. ``cases.jsonl`` and the manifest
# itself are written here; the rest are produced by the pipeline.
DATA_FILES = ["items.jsonl", "conflict_items.jsonl", "candidates.jsonl", "items_rejected.jsonl"]


def load_jsonl(path):
    """Read a JSONL file, returning an empty list when it does not exist."""

    path = Path(path)

    if not path.exists():
        return []

    records = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


TEXT_SUFFIXES = (".jsonl", ".json", ".md", ".txt")


def sha256_of(path):
    """Hex sha256 of a file, read in chunks because images can be several MB.

    Text files are hashed with their line endings normalised to ``\\n``. Without
    that the hash depends on how the checkout happened: git rewrites line endings
    on Windows (``core.autocrlf``) and leaves them alone elsewhere, so the same
    commit produced two different hashes and ``--verify`` failed on the clone.
    """

    path = Path(path)
    normalize = path.suffix.lower() in TEXT_SUFFIXES
    digest = hashlib.sha256()
    carry = b""

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            if not normalize:
                digest.update(chunk)
                continue

            data = carry + chunk
            # A CRLF split across two chunks would otherwise be missed.
            if data.endswith(b"\r"):
                carry = b"\r"
                data = data[:-1]
            else:
                carry = b""

            digest.update(data.replace(b"\r\n", b"\n"))

    if carry:
        digest.update(carry)

    return digest.hexdigest()


def load_corpus(path):
    """Index the crawl corpus by uid, the key the pipeline uses everywhere."""

    with open(path, encoding="utf-8") as f:
        corpus = json.load(f)

    cases = corpus["cases"] if isinstance(corpus, dict) else corpus
    index = {}

    for case in cases:
        index[f"with_images:{case['case_idx']}"] = case

    return index


def referenced_images(records):
    """Union of image paths over all records, order-stable for a stable manifest."""

    seen = set()

    for record in records:
        for relative in record.get("images") or []:
            seen.add(relative)

    return sorted(seen)


def copy_images(images, image_root, data_images, prune, dry_run):
    """Copy referenced images into the data directory, optionally pruning extras."""

    copied, skipped, missing = 0, 0, []

    for relative in images:
        source = Path(image_root) / relative
        target = data_images / relative

        if not source.exists():
            missing.append(relative)
            continue

        if target.exists() and target.stat().st_size == source.stat().st_size:
            skipped += 1
            continue

        copied += 1

        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    pruned = 0

    if prune and data_images.exists():
        keep = {str(Path(relative)) for relative in images}

        for path in sorted(data_images.rglob("*")):
            if path.is_file() and str(path.relative_to(data_images)) not in keep:
                pruned += 1
                if not dry_run:
                    path.unlink()
                    if not any(path.parent.iterdir()):
                        path.parent.rmdir()

    return copied, skipped, missing, pruned


def write_cases(cases_index, uids, path, dry_run):
    """Export the original case records of the shipped items.

    The crawl corpus is not part of the repository, so on a clone without it
    ``cases_index`` is empty. Writing in that case would truncate the tracked
    export and silently take ``text_a`` with it, so nothing is written and the
    existing file is kept.
    """

    records = [cases_index[uid] for uid in uids if uid in cases_index]

    if not records:
        return 0

    if not dry_run:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(records)


def count_lines(path):
    """Number of non-empty lines, used for the manifest counts."""

    path = Path(path)

    if not path.exists():
        return 0

    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def build_manifest(data_dir, images, corpus_path, dry_run):
    """Collect counts and hashes of everything the directory ships."""

    files = {}

    for name in DATA_FILES + ["cases.jsonl"]:
        path = data_dir / name
        if path.exists():
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256_of(path)}

    image_entries = []

    for relative in images:
        path = data_dir / "images" / relative
        if path.exists():
            image_entries.append(
                {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_of(path)}
            )

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {
            "corpus": str(corpus_path.relative_to(REPO_ROOT)).replace("\\", "/")
            if corpus_path.is_relative_to(REPO_ROOT)
            else str(corpus_path),
            "corpus_sha256": sha256_of(corpus_path) if corpus_path.exists() else None,
        },
        "counts": {
            "items": count_lines(data_dir / "items.jsonl"),
            "conflict_items": count_lines(data_dir / "conflict_items.jsonl"),
            "candidates": count_lines(data_dir / "candidates.jsonl"),
            "items_rejected": count_lines(data_dir / "items_rejected.jsonl"),
            "cases": count_lines(data_dir / "cases.jsonl"),
            "images": len(image_entries),
        },
        "files": files,
        "images": image_entries,
    }

    if not dry_run:
        with open(data_dir / "MANIFEST.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.write("\n")

    return manifest


def verify_manifest(data_dir):
    """Check every file the manifest lists against its recorded sha256."""

    manifest_path = Path(data_dir) / "MANIFEST.json"

    if not manifest_path.exists():
        print(f"[error] no manifest at {manifest_path}")
        return 1

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    checked, bad = 0, []

    for name, meta in manifest["files"].items():
        path = Path(data_dir) / name
        checked += 1
        if not path.exists() or sha256_of(path) != meta["sha256"]:
            bad.append(name)

    for entry in manifest["images"]:
        path = Path(data_dir) / "images" / entry["path"]
        checked += 1
        if not path.exists() or sha256_of(path) != entry["sha256"]:
            bad.append(entry["path"])

    print(f"checked {checked} file(s) against {manifest_path}")

    if bad:
        print(f"[error] {len(bad)} mismatch(es):")
        for name in bad[:10]:
            print(f"  {name}")
        return 1

    print(f"ok: {manifest['counts']['items']} items, {manifest['counts']['images']} images")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Freeze bench/data into a self-contained dataset.")
    parser.add_argument("--data", default=str(DATA), help="frozen dataset directory")
    parser.add_argument("--corpus", default=str(CORPUS), help="crawl output with the source cases")
    parser.add_argument("--image-root", default=str(IMAGE_ROOT), help="crawl output with the images")
    parser.add_argument("--prune", action="store_true", help="delete images no record references")
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="re-hash the files already in the data directory without copying from the crawl",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="only check the directory against MANIFEST.json (no crawl output needed)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data)
    data_images = data_dir / "images"
    corpus_path = Path(args.corpus)

    if args.verify:
        raise SystemExit(verify_manifest(data_dir))

    items = load_jsonl(data_dir / "items.jsonl")
    conflicts = load_jsonl(data_dir / "conflict_items.jsonl")

    if not items:
        raise SystemExit(f"no items in {data_dir / 'items.jsonl'}; run bench/build_items.py first")

    cases_index = load_corpus(corpus_path) if corpus_path.exists() else {}
    images = referenced_images(items + conflicts)

    uids = []
    for record in items:
        uid = record.get("uid")
        if uid and uid not in uids:
            uids.append(uid)

    if args.manifest_only:
        # The images in the data directory are already the ones to ship (they may
        # have been re-downloaded at full size); only re-hash them.
        missing = [relative for relative in images if not (data_images / relative).exists()]
        copied, skipped, pruned = 0, len(images) - len(missing), 0
    else:
        copied, skipped, missing, pruned = copy_images(
            images, Path(args.image_root), data_images, args.prune, args.dry_run
        )

    exported = write_cases(cases_index, uids, data_dir / "cases.jsonl", args.dry_run)
    manifest = build_manifest(data_dir, images, corpus_path, args.dry_run)

    if not exported:
        exported = count_lines(data_dir / "cases.jsonl")

    print(f"items={len(items)} conflicts={len(conflicts)} cases={exported} images={len(images)}")
    print(f"images copied={copied} unchanged={skipped} pruned={pruned}")
    print(f"dataset size={sum(entry['bytes'] for entry in manifest['images']) / 1048576:.1f} MB (images)")

    if missing:
        print(f"[error] {len(missing)} referenced image(s) missing under {args.image_root}:")
        for relative in missing[:10]:
            print(f"  {relative}")
        raise SystemExit(1)

    if not cases_index:
        print(f"[info] no crawl corpus at {corpus_path}; kept the existing cases.jsonl as is")


if __name__ == "__main__":
    main()
