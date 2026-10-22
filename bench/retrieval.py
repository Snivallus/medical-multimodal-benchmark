"""Diagnosis-level neighbour retrieval for distractor generation.

Distractors must be *plausible* rather than random, so each case is compared
against the diagnoses of the whole corpus. Character n-grams are enough here:
Chinese diagnosis names share body parts and disease suffixes ("…骨折",
"…韧带损伤"), which is exactly the kind of near-miss a careless reader picks.

No embedding model is needed, so the step stays reproducible offline.
"""

import re

# The corpus writes the same disease several ways. Without this map, "急性脑梗塞"
# would be offered as a distractor for "急性脑梗死", i.e. a second correct answer.
SYNONYM_MAP = (
    ("梗塞", "梗死"),
    ("前十字韧带", "前交叉韧带"),
    ("后十字韧带", "后交叉韧带"),
    ("格林-巴利", "格林巴利"),
    ("返流", "反流"),
)


def normalize(name):
    """Normalize a diagnosis name for comparison.

    Bracketed staging is dropped, synonyms are folded, and every separator is
    removed, so that ``格林-巴利综合征`` and ``格林巴利综合征`` collapse onto one
    key.
    """

    name = (name or "").strip()
    name = re.sub(r"[（(][^）)]*[）)]", "", name)   # drop staging in brackets

    for variant, canonical in SYNONYM_MAP:
        name = name.replace(variant, canonical)

    name = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", name)

    return name


def char_ngrams(text, n=2):
    """Return the set of character n-grams of ``text``."""

    text = normalize(text)

    if len(text) < n:
        return {text} if text else set()

    return {text[i:i + n] for i in range(len(text) - n + 1)}


def similarity(a, b):
    """Jaccard similarity over character bigrams."""

    grams_a = char_ngrams(a)
    grams_b = char_ngrams(b)

    if not grams_a or not grams_b:
        return 0.0

    return len(grams_a & grams_b) / len(grams_a | grams_b)


def is_too_close(candidate, correct):
    """Return ``True`` when a candidate would be a duplicate of a correct option.

    Duplicates are detected on the normalized string, on containment, and on a
    high n-gram overlap, because "右膝前交叉韧带断裂" and "前交叉韧带断裂" are the
    same answer spelled differently.
    """

    norm_candidate = normalize(candidate)

    for name in correct:
        norm_name = normalize(name)

        if not norm_candidate or not norm_name:
            continue

        if norm_candidate == norm_name:
            return True

        if norm_candidate in norm_name or norm_name in norm_candidate:
            return True

        if similarity(norm_candidate, norm_name) >= 0.8:
            return True

    return False


def build_corpus_index(cases):
    """Collect every diagnosis in the corpus, with the case ids it came from.

    Each case is expected to carry a ``_diagnoses`` list, as produced by
    :func:`medical.parse_diagnoses`.

    Returns ``{normalized_diagnosis: {"name": str, "case_ids": set}}``.
    """

    index = {}

    for case in cases:
        for name in (case.get("_diagnoses") or []):
            key = normalize(name)

            if not key:
                continue

            entry = index.setdefault(key, {"name": name, "case_ids": set()})
            entry["case_ids"].add(case.get("case_idx"))

    return index


def top_similar(query, index, k=20, exclude_case_ids=(), exclude_names=()):
    """Return the ``k`` most similar diagnoses from the index.

    ``exclude_case_ids`` drops diagnoses that belong to the query case itself,
    which would otherwise be correct-but-wrong-looking options.
    """

    scored = []

    for key, entry in index.items():

        if entry["case_ids"] & set(exclude_case_ids):
            continue

        if exclude_names and is_too_close(entry["name"], exclude_names):
            continue

        score = similarity(query, entry["name"])

        if score <= 0:
            continue

        scored.append((score, entry["name"]))

    scored.sort(key=lambda pair: (-pair[0], pair[1]))

    # Keep the list short but varied: at most two entries per n-gram score band.
    picked = []
    seen_names = set()

    for score, name in scored:
        key = normalize(name)

        if key in seen_names:
            continue

        picked.append({"name": name, "score": round(score, 3)})
        seen_names.add(key)

        if len(picked) >= k:
            break

    return picked
