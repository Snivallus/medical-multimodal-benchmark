"""Medical text helpers shared by the benchmark pipeline.

Three jobs live here:

* **modality detection** — which examinations a case text mentions;
* **diagnosis parsing** — turning one ``诊断结果`` string into a list of
  diagnoses, which is what the answer key and the distractors are built from;
* **imaging redaction** — removing imaging, pathology and intra-operative
  conclusions from the case text, so that the text-only control cannot answer
  from a sentence like "MRI 示前交叉韧带断裂".
"""

import re

# ------------------------------------------------------------------
# Modalities
# ------------------------------------------------------------------
def token(text):
    """Match an ASCII abbreviation next to Chinese text.

    Python's ``\\b`` treats CJK characters as word characters, so ``\\bCT\\b``
    would not match ``胸部CT``. An ASCII-only lookaround is what is needed.
    """

    return rf"(?<![A-Za-z]){re.escape(text)}(?![A-Za-z])"


MODALITY_PATTERNS = {
    "CT": [token("CT"), token("CTA"), "PET-CT"],
    "MRI": [token("MRI"), token("MR"), "磁共振", "核磁", token("DWI"), token("T1WI"), token("T2WI")],
    "X线": ["X线", "X光", "平片", "胸片", "正侧位片", "正位片", token("DR")],
    "超声": ["超声", "彩超", "B超", "多普勒"],
    "内镜": [
        "胃镜", "肠镜", "支气管镜", "喉镜", "宫腔镜", "腹腔镜",
        "关节镜", "膀胱镜", "胆道镜", "纵隔镜", "鼻内镜",
    ],
    "病理": ["病理", "活检", "免疫组化", "细胞学", "涂片"],
    "皮肤镜": ["皮肤镜"],
    "临床照片": ["皮损", "皮疹", "水疱", "溃疡面", "外观照片", "患处照片"],
    "眼底": ["眼底", "视网膜", "荧光血管造影", token("OCT")],
    "心电图": ["心电图", token("ECG"), token("EKG")],
    "核医学": [token("PET"), token("SPECT"), "核素", "骨扫描", "碘131"],
}

# Modalities whose evidence is an image the model can actually be shown.
VISUAL_MODALITIES = set(MODALITY_PATTERNS) - {"心电图"}

# Terms that name a visual examination. They decide whether a sentence is about
# evidence the model would have to look at.
MODALITY_TERMS = [p for patterns in MODALITY_PATTERNS.values() for p in patterns]
MODALITY_TERMS += ["影像", "造影", "摄片", "平扫", "增强扫描", "阅片"]

# Intra-operative and pathological findings are visual evidence even when no
# examination name appears.
STRONG_MARKERS = ("术中见", "术中所见", "术中探查", "探查见", "镜下见", "病理")

# Report-like phrasing; it only counts together with a modality.
WEAK_MARKERS = ("示", "提示", "显示", "报告", "可见")


def detect_modalities(text):
    """Return the modalities mentioned in ``text``, in pattern order."""

    return [
        name
        for name, patterns in MODALITY_PATTERNS.items()
        if any(re.search(pattern, text) for pattern in patterns)
    ]


# ------------------------------------------------------------------
# Diagnoses
# ------------------------------------------------------------------
# "1." / "2、" / "(1)" / "①" prefixes used by the site.
ENUM_PREFIX = re.compile(r"^\s*(?:[（(]?\d{1,2}[）)、.．]|[①-⑳])\s*")
TRAILING_PUNCT = "。；;，,、 \t\n"


def parse_diagnoses(text):
    """Split a ``诊断结果`` string into an ordered list of diagnoses.

    The site separates diagnoses with ``；`` or a numbered list, but also leaves
    commas inside a single diagnosis (``高血压病2级，很高危组``), so commas are
    only used as separators when no semicolon is present at all.
    """

    text = (text or "").strip()

    if not text:
        return []

    if re.search(r"[；;\n]", text):
        parts = re.split(r"[；;\n]", text)
    else:
        parts = [text]

    # A numbered list inside one segment, e.g. "1.xxx 2.yyy".
    expanded = []
    for part in parts:
        pieces = re.split(r"(?<=[^\d])\s+(?=\d{1,2}[.、])", part.strip())
        expanded.extend(pieces)

    diagnoses = []

    for part in expanded:
        part = ENUM_PREFIX.sub("", part).strip(TRAILING_PUNCT)
        if part and part not in diagnoses:
            diagnoses.append(part)

    return diagnoses


# Chronic background conditions that the site lists next to the real diagnosis.
# They must not become correct options: nobody would "select" them as the
# diagnosis for this admission, and they inflate the answer set, which distorts
# the partial-credit score.
COMORBIDITY_PATTERNS = (
    "高血压", "糖尿病", "冠心病", "高脂血症", "高脂蛋白血症", "脂肪肝",
    "陈旧性心肌梗死", "脑梗死后遗症", "前列腺增生", "骨质疏松", "贫血",
    "低蛋白血症", "电解质紊乱", "心律失常", "心功能不全", "心功能II",
    "心功能III", "心功能IV",
)


def is_comorbidity(diagnosis):
    """Return ``True`` when a diagnosis looks like a background condition."""

    return any(pattern in diagnosis for pattern in COMORBIDITY_PATTERNS)


def split_answer_set(diagnoses):
    """Split parsed diagnoses into ``(answers, comorbidities)``."""

    answers = [name for name in diagnoses if not is_comorbidity(name)]
    background = [name for name in diagnoses if is_comorbidity(name)]

    return answers, background


# ------------------------------------------------------------------
# Imaging conclusions
# ------------------------------------------------------------------
# Sentence delimiters used when scanning the rewritten text for leftovers.
SENTENCE_SPLIT = re.compile(r"(?<=[。！？；\n])")

# Markers that mean the segment is worth a second look. The weak ones only
# trigger a redaction when a modality is also named, so ordinary prose such as
# "可见轻度水肿" in a physical exam is not deleted.
STRONG_MARKERS = ("术中见", "术中所见", "术中探查", "探查见", "镜下见", "病理")
WEAK_MARKERS = ("示", "提示", "显示", "报告", "可见")


def is_imaging_segment(segment):
    """Return ``True`` when a segment states a visual conclusion.

    Intra-operative and pathological findings always count. Otherwise an
    examination must be named *and* the sentence must read like a report, so
    ordinary prose such as "查体可见轻度水肿" survives.
    """

    if not segment.strip():
        return False

    if any(marker in segment for marker in STRONG_MARKERS):
        return True

    if not any(re.search(pattern, segment) for pattern in MODALITY_TERMS):
        return False

    has_weak = any(marker in segment for marker in WEAK_MARKERS)
    has_colon = "：" in segment or ":" in segment

    return has_weak or has_colon


def normalize_whitespace(text):
    """Collapse whitespace without touching line breaks."""

    text = (text or "").replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]

    return "\n".join(line for line in lines if line)


# Sentences that only say "an examination was done" are the intended replacement
# for a removed report and must not be flagged as a leftover report.
PLACEHOLDER_MARKERS = ("见附图", "无文字报告", "已行相关影像检查")

def mentions_imaging_conclusion(text):
    """Return ``True`` when ``text`` still states an imaging conclusion.

    This is the deterministic guard behind the rewrite step: an LLM may claim to
    have removed the report while the sentence is still there, so the rewritten
    text is re-checked with the same rules used by the offline redactor.
    """

    for sentence in SENTENCE_SPLIT.split(text or ""):

        if not sentence.strip():
            continue

        if any(marker in sentence for marker in PLACEHOLDER_MARKERS):
            continue

        if is_imaging_segment(sentence):
            return True

    return False


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------
def score_selection(correct, selected):
    """Score one multiple-answer item with partial credit and a wrong-answer penalty.

    ``clip((TP - FP) / |G|, 0, 1)``: selecting everything scores zero, missing a
    secondary diagnosis costs part of the credit, and a single-diagnosis item
    reduces to plain 0/1 accuracy so ``Delta acc`` keeps its usual scale.
    """

    correct = set(correct)
    selected = set(selected)

    if not correct:
        raise ValueError("an item needs at least one correct option")

    true_positive = len(correct & selected)
    false_positive = len(selected - correct)

    return max(0.0, min(1.0, (true_positive - false_positive) / len(correct)))


def main_diagnosis_hit(main_option_id, selected):
    """Return ``True`` when the option holding the main diagnosis was selected.

    The caller must pass the *option id* of the main diagnosis: correct option
    ids are in shuffled option order, so their first element is not the main
    diagnosis.
    """

    return bool(main_option_id) and main_option_id in set(selected)
