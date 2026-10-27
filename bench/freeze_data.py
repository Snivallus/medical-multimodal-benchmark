"""Rebuild and verify ``bench/data/MANIFEST.json``.

``bench/data/`` is the frozen dataset that ships with the repository: the items,
the conflict pairs, the source case records and the images they reference. The
directory is maintained in place (hand-cropped images, removed items), so the only
part left to do mechanically is to keep the manifest in step with it:

* counts, sizes and sha256 of every shipped file, which is what tells a fresh
  clone that it has the dataset the experiment used;
* ``--prune`` deletes images that no record references any more;
* ``--verify`` checks a clone against the manifest, no key and no corpus needed.

Everything here reads and writes ``bench/data/`` only. The pipeline that produced
the items, and the crawl corpus it read from, are not part of the repository, so
there is nothing to import from them and nothing to copy in.

Run it after any hand edit of the data directory, otherwise the manifest and the
items no longer describe the same dataset.

Usage::

    python bench/freeze_data.py                 # recompute the manifest
    python bench/freeze_data.py --prune         # also drop unreferenced images
    python bench/freeze_data.py --dry-run       # report changes without writing
    python bench/freeze_data.py --verify        # check a fresh clone
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "bench" / "data"

# Files of the frozen dataset, in manifest order. ``images/`` is hashed separately
# below, and ``MANIFEST.json`` describes the rest without hashing itself.
DATA_FILES = ["items.jsonl", "conflict_items.jsonl", "cases.jsonl"]


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


def referenced_images(records):
    """Union of image paths over all records, order-stable for a stable manifest."""

    seen = set()

    for record in records:
        for relative in record.get("images") or []:
            seen.add(relative)

    return sorted(seen)


def count_lines(path):
    """Number of non-empty lines, used for the manifest counts."""

    path = Path(path)

    if not path.exists():
        return 0

    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def prune_images(images, data_images, dry_run):
    """Delete images under ``data_images`` that no record references."""

    if not Path(data_images).exists():
        return 0

    keep = {str(Path(relative)) for relative in images}
    pruned = 0

    for path in sorted(Path(data_images).rglob("*")):
        if not path.is_file() or str(path.relative_to(data_images)) in keep:
            continue

        pruned += 1

        if not dry_run:
            path.unlink()
            if not any(path.parent.iterdir()):
                path.parent.rmdir()

    return pruned


def build_manifest(data_dir, images, dry_run):
    """Collect counts and hashes of everything the directory ships."""

    files = {}

    for name in DATA_FILES:
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
        "counts": {
            "items": count_lines(data_dir / "items.jsonl"),
            "conflict_items": count_lines(data_dir / "conflict_items.jsonl"),
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
    parser = argparse.ArgumentParser(
        description="Recompute or verify bench/data/MANIFEST.json."
    )
    parser.add_argument("--data", default=str(DATA), help="frozen dataset directory")
    parser.add_argument("--prune", action="store_true", help="delete images no record references")
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="kept for old notes: recomputing the manifest is now the default",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="only check the directory against MANIFEST.json (no image scan)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data)

    if args.verify:
        raise SystemExit(verify_manifest(data_dir))

    items = load_jsonl(data_dir / "items.jsonl")

    if not items:
        raise SystemExit(f"no items in {data_dir / 'items.jsonl'}")

    conflicts = load_jsonl(data_dir / "conflict_items.jsonl")
    images = referenced_images(items + conflicts)
    data_images = data_dir / "images"

    missing = [relative for relative in images if not (data_images / relative).exists()]
    pruned = prune_images(images, data_images, args.dry_run) if args.prune else 0
    manifest = build_manifest(data_dir, images, args.dry_run)

    print(
        f"items={manifest['counts']['items']} conflicts={manifest['counts']['conflict_items']} "
        f"cases={manifest['counts']['cases']} images={manifest['counts']['images']}"
    )
    print(f"images pruned={pruned}")
    print(f"dataset size={sum(entry['bytes'] for entry in manifest['images']) / 1048576:.1f} MB (images)")

    # ``cases.jsonl`` ships the original case text behind ``text_a``. Nothing can
    # regenerate it any more (the corpus is not in the repository), so a stale row
    # set shows up here instead of as a failed evaluation run.
    known = {record.get("case_idx") for record in load_jsonl(data_dir / "cases.jsonl")}
    absent = [item["item_id"] for item in items if item.get("case_id") not in known]

    if absent:
        print(
            f"[warn] {len(absent)} item(s) have no row in cases.jsonl; "
            f"variant text_a would fail on them: {', '.join(absent[:5])}"
        )

    if missing:
        print(f"[error] {len(missing)} referenced image(s) missing under {data_images}:")
        for relative in missing[:10]:
            print(f"  {relative}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
