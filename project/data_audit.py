#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iiyi 数据集体检脚本 (A 组数据治理第一步)
==========================================

目的: 把"人工抽样看到的问题"变成可复现的全量统计, 产出选题答辩要用的数据证据表.

检查项:
  1. 图片 magic byte 与扩展名是否一致 (现有数据中大量 .jpg 实为 PNG/WebP),
     统计真实格式分布, 并给出可解码率.
  2. 重复病例检测 (同一病例被多次入库 -> 训练/测试泄漏), 按标题与内容指纹聚类.
  3. 诊断字段健康度: 多诊断比例、含不确定措辞比例、空诊断.
  4. 文本可解性代理指标: 辅助检查 是否已写出诊断结论 (长度 + 关键词 + 诊断结果子串命中),
     用于估计"遮住图片仍可作答"的样本比例.
  5. 图像可用性分级 (可选, 需要 Pillow): 分辨率过小的判为低清废图.

用法:
    python project/data_audit.py --root iiyi_dataset/results --out project/audit_report.json
    python project/data_audit.py --root iiyi_dataset/results --no-pillow   # 不依赖 Pillow

注意: 本脚本只读数据, 不修改任何原始文件; 输出报告写到 --out.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------------------------------------------------------
# 1. magic byte 嗅探: 不依赖第三方库即可判断真实容器格式
# --------------------------------------------------------------------------

MAGIC_SIGNATURES = [
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
    (b"II*\x00", "tiff"),
    (b"MM\x00*", "tiff"),
]


def sniff_format(head: bytes) -> str:
    """根据文件头返回真实格式名, 无法识别返回 'unknown'."""
    for sig, name in MAGIC_SIGNATURES:
        if head.startswith(sig):
            return name
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1", b"ftypavif"):
        return head[8:12].decode("ascii", "replace")
    return "unknown"


EXT_TO_FORMATS = {
    ".jpg": {"jpeg"},
    ".jpeg": {"jpeg"},
    ".png": {"png"},
    ".gif": {"gif"},
    ".bmp": {"bmp"},
    ".webp": {"webp"},
    ".tif": {"tiff"},
    ".tiff": {"tiff"},
}


def image_extension_mismatch(path: Path) -> dict:
    """读取文件头, 判断扩展名与真实格式是否一致."""
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError as e:
        return {"path": str(path), "status": "unreadable", "error": str(e)}

    real = sniff_format(head)
    ext = path.suffix.lower()
    expected = EXT_TO_FORMATS.get(ext, set())
    compatible = real in expected
    return {
        "path": str(path),
        "declared_ext": ext,
        "real_format": real,
        "compatible": compatible,
        "size": path.stat().st_size,
    }


# --------------------------------------------------------------------------
# 2. 诊断字段健康度
# --------------------------------------------------------------------------

HEDGE_WORDS = ["可能", "疑似", "待排", "考虑", "不除外", "待查", "待定", "性质不明"]
MODALITY_KEYWORDS = [
    "MRI", "磁共振", "CT", "超声", "彩超", "彩色多普勒", "X线", "X光", "DR",
    "数字化摄影", "钼靶", "平片", "病理", "活检", "内镜", "内窥镜", "胃镜",
    "肠镜", "喉镜", "支气管镜", "关节镜", "膀胱镜", "腹腔镜", "造影", "DSA",
    "心电图", "肌电图", "脑电图",
]


def normalize(s: str) -> str:
    """去掉空白与标点, 用于内容指纹."""
    return re.sub(r"[\s，。；、,.;:：!！?？\-—()（）\[\]【】]", "", s or "")


def diagnose_case_health(case: dict) -> dict:
    dx = (case.get("诊断结果") or "").strip()
    aux = (case.get("辅助检查") or "").strip()
    findings = {
        "case_idx": case.get("case_idx"),
        "empty_diagnosis": dx == "",
        "multi_label": "；" in dx,
        "hedged": any(w in dx for w in HEDGE_WORDS),
        "empty_aux_exam": aux == "",
        "aux_exam_len": len(aux),
        "aux_exam_long": len(aux) > 200,
        "n_images": len(case.get("images") or []),
    }
    # 文本可解性代理: 主要诊断名是否已在 辅助检查 / 诊断依据 / 查体 中逐字出现
    primary = re.split(r"[；;，,、]", dx)[0].strip() if dx else ""
    haystack = " ".join(
        (case.get(k) or "") for k in ("辅助检查", "诊断依据", "查体", "主诉", "现病史")
    )
    findings["primary_dx_in_text"] = bool(primary) and primary in haystack
    # 更宽松的代理: 诊断的关键片段是否出现 (去掉"左/右/侧"等修饰后仍匹配)
    core = re.sub(r"^(左|右|双侧|双|陈旧性|急性|慢性)", "", primary)
    findings["core_dx_in_text"] = bool(core) and len(core) >= 2 and core in haystack
    findings["modalities_in_aux"] = sorted(
        {k for k in MODALITY_KEYWORDS if k in aux}
    )
    return findings


# --------------------------------------------------------------------------
# 3. 重复病例聚类
# --------------------------------------------------------------------------

def cluster_duplicates(cases: list[dict]) -> dict:
    """按标题精确匹配 + 内容指纹分组, 找出同一病例被多次入库的情况."""
    by_title = defaultdict(list)
    by_fingerprint = defaultdict(list)

    for c in cases:
        title = normalize(c.get("标题") or "")
        if title:
            by_title[title].append(c.get("case_idx"))

        fp_src = normalize(c.get("辅助检查") or "") + "|" + normalize(c.get("诊断结果") or "")
        if fp_src.strip("|"):
            by_fingerprint[fp_src].append(c.get("case_idx"))

    title_dups = {k: v for k, v in by_title.items() if len(v) > 1}
    fp_dups = {k: v for k, v in by_fingerprint.items() if len(v) > 1}

    # 去重后独立病例数 = 总数 - Σ(每簇多余份数)
    redundant_by_title = sum(len(v) - 1 for v in title_dups.values())
    redundant_by_fp = sum(len(v) - 1 for v in fp_dups.values())
    dup_case_ids = {i for ids in title_dups.values() for i in ids}
    dup_case_ids |= {i for ids in fp_dups.values() for i in ids}

    return {
        "n_cases": len(cases),
        "n_unique_titles": len(by_title),
        "titles_occurring_multiple_times": len(title_dups),
        "cases_involved_in_title_duplicates": sum(len(v) for v in title_dups.values()),
        "duplicate_title_examples": [
            {"title": k, "case_idx": v} for k, v in list(title_dups.items())[:20]
        ],
        "n_content_fingerprint_dup_groups": len(fp_dups),
        "cases_involved_in_fingerprint_duplicates": sum(len(v) for v in fp_dups.values()),
        "estimated_independent_cases_by_title": len(cases) - redundant_by_title,
        "estimated_independent_cases_by_fingerprint": len(cases) - redundant_by_fp,
        "all_duplicate_case_idx_sample": sorted(dup_case_ids)[:50],
    }


# --------------------------------------------------------------------------
# 4. 图像清单体检
# --------------------------------------------------------------------------

def audit_images(root: Path, cases: list[dict], use_pillow: bool) -> dict:
    stats = Counter()
    per_case = {}
    broken_examples = []
    resolutions = []
    pillow = None
    if use_pillow:
        try:
            from PIL import Image  # type: ignore
            pillow = Image
        except ImportError:
            print("[warn] Pillow 不可用, 跳过分辨率统计", file=sys.stderr)

    listed = 0
    missing = 0
    for c in cases:
        idx = c.get("case_idx")
        imgs = c.get("images") or []
        ok_for_case = 0
        case_records = []
        for rel in imgs:
            listed += 1
            p = root / rel
            if not p.exists():
                missing += 1
                stats["missing_file"] += 1
                continue
            info = image_extension_mismatch(p)
            if info.get("status") == "unreadable":
                stats["unreadable"] += 1
                continue
            stats["decodable_by_magic"] += 1
            stats[f"format:{info['real_format']}"] += 1
            if not info["compatible"]:
                stats["ext_mismatch"] += 1
                if len(broken_examples) < 20:
                    broken_examples.append(
                        {k: info[k] for k in ("path", "declared_ext", "real_format")}
                    )
            else:
                stats["ext_ok"] += 1

            if pillow is not None:
                try:
                    with pillow.open(p) as im:
                        w, h = im.size
                    resolutions.append((w, h, str(rel)))
                    if min(w, h) < 300:
                        stats["low_res_lt300"] += 1
                    ok_for_case += 1
                except Exception:
                    stats["pillow_open_failed"] += 1
            else:
                ok_for_case += 1

        per_case[idx] = {"n_images": len(imgs), "usable": ok_for_case}
        if imgs and ok_for_case == 0:
            stats["cases_with_zero_usable_image"] += 1

    res_sorted = sorted(resolutions, key=lambda t: min(t[0], t[1]))
    return {
        "listed_image_refs": listed,
        "missing_files": missing,
        "stats": dict(stats),
        "cases_with_images": sum(1 for c in cases if c.get("images")),
        "cases_with_zero_usable_image": stats["cases_with_zero_usable_image"],
        "extension_mismatch_examples": broken_examples,
        "resolution_min_10": [
            {"w": w, "h": h, "path": p} for w, h, p in res_sorted[:10]
        ],
        "resolution_max_5": [
            {"w": w, "h": h, "path": p} for w, h, p in res_sorted[-5:]
        ],
        "per_case_usable": per_case if len(per_case) <= 500 else "omitted (too many cases)",
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def load_cases(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="iiyi_dataset/results",
                    help="包含 case_with_images.json / case_without_images.json 与 images/ 的目录")
    ap.add_argument("--out", default="project/audit_report.json")
    ap.add_argument("--no-pillow", action="store_true",
                    help="不统计分辨率 (无需安装 Pillow)")
    ap.add_argument("--skip-images", action="store_true",
                    help="只做文本侧与重复检测 (图片全量嗅探较慢)")
    args = ap.parse_args()

    root = Path(args.root)
    report: dict = {"root": str(root)}

    with_img = root / "case_with_images.json"
    without_img = root / "case_without_images.json"

    if with_img.exists():
        cases = load_cases(with_img)
        report["with_images"] = {
            "n_cases": len(cases),
            "duplicates": cluster_duplicates(cases),
        }
        health = [diagnose_case_health(c) for c in cases]
        report["with_images"]["health"] = {
            "empty_diagnosis": sum(h["empty_diagnosis"] for h in health),
            "multi_label_diagnosis": sum(h["multi_label"] for h in health),
            "hedged_diagnosis": sum(h["hedged"] for h in health),
            "empty_aux_exam": sum(h["empty_aux_exam"] for h in health),
            "aux_exam_gt_200_chars": sum(h["aux_exam_long"] for h in health),
            "primary_dx_verbatim_in_text": sum(h["primary_dx_in_text"] for h in health),
            "core_dx_verbatim_in_text": sum(h["core_dx_in_text"] for h in health),
            "aux_exam_len_buckets": dict(Counter(
                "empty" if h["aux_exam_len"] == 0 else
                "1-49" if h["aux_exam_len"] < 50 else
                "50-99" if h["aux_exam_len"] < 100 else
                "100-199" if h["aux_exam_len"] < 200 else
                "200+"
                for h in health
            )),
            "modality_keyword_case_counts": dict(Counter(
                k for h in health for k in h["modalities_in_aux"]
            ).most_common()),
        }
        if not args.skip_images:
            report["with_images"]["images"] = audit_images(
                root, cases, use_pillow=not args.no_pillow
            )
            # 关键交叉统计: 图片可用 且 文本未泄漏 的候选池
            img_audit = report["with_images"]["images"]["per_case_usable"]
            if isinstance(img_audit, dict):
                leaked = {h["case_idx"] for h in health if h["core_dx_in_text"]}
                usable = [i for i, r in img_audit.items() if r["usable"] > 0]
                report["with_images"]["candidate_pool"] = {
                    "usable_image_and_no_text_leak": len(
                        [i for i in usable if i not in leaked]
                    ),
                    "usable_image_total": len(usable),
                }

    if without_img.exists():
        cases2 = load_cases(without_img)
        report["without_images"] = {
            "n_cases": len(cases2),
            "duplicates": cluster_duplicates(cases2),
        }
        health2 = [diagnose_case_health(c) for c in cases2]
        report["without_images"]["health"] = {
            "empty_diagnosis": sum(h["empty_diagnosis"] for h in health2),
            "multi_label_diagnosis": sum(h["multi_label"] for h in health2),
            "primary_dx_verbatim_in_text": sum(h["primary_dx_in_text"] for h in health2),
            "core_dx_verbatim_in_text": sum(h["core_dx_in_text"] for h in health2),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 控制台摘要
    print(f"[ok] 报告已写入 {out}\n")
    if "with_images" in report:
        w = report["with_images"]
        print(f"case_with_images.json: {w['n_cases']} 例")
        d = w["duplicates"]
        print(f"  唯一标题数: {d['n_unique_titles']}"
              f"  重复标题数: {d['titles_occurring_multiple_times']}"
              f"  涉及病例: {d['cases_involved_in_title_duplicates']}")
        h = w["health"]
        print(f"  多诊断(含；): {h['multi_label_diagnosis']}"
              f"  不确定措辞: {h['hedged_diagnosis']}"
              f"  空诊断: {h['empty_diagnosis']}")
        print(f"  主要诊断逐字已在文本中出现: {h['primary_dx_verbatim_in_text']}"
              f"  (宽松口径: {h['core_dx_verbatim_in_text']})")
        print(f"  辅助检查长度分布: {h['aux_exam_len_buckets']}")
        if "images" in w:
            s = w["images"]["stats"]
            listed = w["images"]["listed_image_refs"]
            print(f"  图片引用 {listed} 条, 扩展名/格式不符 {s.get('ext_mismatch', 0)}"
                  f" ({100.0 * s.get('ext_mismatch', 0) / max(listed, 1):.1f}%)")
            print(f"  实际格式分布: "
                  f"{ {k.split(':')[1]: v for k, v in s.items() if k.startswith('format:')} }")
            print(f"  全部图片不可用的病例: {w['images']['cases_with_zero_usable_image']}")
            if "candidate_pool" in w:
                print(f"  候选池(图片可用且文本未泄漏): "
                      f"{w['candidate_pool']['usable_image_and_no_text_leak']}"
                      f" / {w['candidate_pool']['usable_image_total']}")
    if "without_images" in report:
        print(f"\ncase_without_images.json: {report['without_images']['n_cases']} 例 (纯文本对照集)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
