#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SEMI_AR_BS_TAG_RE = re.compile(r"semi_ar(?:_bs)?(\d+)")
_SEMI_AR_LABEL_RE = re.compile(r"semi_ar\s*\((\d+)\)")
_PERM_FILE_RE = re.compile(r"(success|failed)_perms_([A-Za-z0-9_.-]+)_sh\d+_of\d+\.jsonl$")

def format_method_label(method: str, block_size: int | None = None) -> str:
    if method == "semi_ar" and block_size is not None:
        return f"semi_ar ({block_size})"
    return method

def block_size_from_label(label: str | None) -> int | None:
    if not label:
        return None
    m = _SEMI_AR_LABEL_RE.search(label)
    if m:
        return int(m.group(1))
    m = _SEMI_AR_BS_TAG_RE.search(label)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None

def normalize_method_label(label: str | None, block_hint: int | None = None) -> str | None:
    if not label:
        return None
    label = label.strip()
    if not label:
        return None
    block = block_size_from_label(label)
    if block is None:
        block = block_hint
    if label.startswith("semi_ar"):
        return format_method_label("semi_ar", block)
    return label

def method_matches(label: str | None, query: str | None) -> bool:
    if not label or not query:
        return False
    label = label.strip()
    query = query.strip()
    if not label or not query:
        return False
    if label == query:
        return True
    if "(" not in query:
        base = query
        return label == base or label.startswith(f"{base} ")
    return False

def method_tag_from_label(label: str) -> str:
    block = block_size_from_label(label)
    if block is not None:
        return f"semi_ar_bs{block}"
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "-", label).strip("-_.")
    return cleaned or "method"

def extract_method_from_tag(tag: str) -> tuple[str | None, int | None]:
    base = tag
    idx = base.find("_sh")
    if idx != -1:
        base = base[:idx]
    block_hint = block_size_from_label(base)
    label = normalize_method_label(base, block_hint)
    return label, block_hint

def load_perm_records(
    in_dir: str,
    statuses: List[str],
    method_include: Dict[str, List[str]] | None = None,
    method_exclude: Dict[str, List[str]] | None = None,
) -> List[Dict[str, Any]]:
    method_include = method_include or {}
    method_exclude = method_exclude or {}
    items: List[Dict[str, Any]] = []
    for status in statuses:
        pattern = os.path.join(in_dir, f"{status}_perms_*.jsonl")
        paths = sorted(glob.glob(pattern))
        pattern_re = re.compile(rf"{status}_perms_([a-zA-Z0-9_]+)_")
        for path in paths:
            base = os.path.basename(path)
            method_key = ""
            m_key = _PERM_FILE_RE.match(base)
            if m_key:
                method_key = normalize_method_key(m_key.group(2))
            include_patterns = method_include.get(method_key, [])
            if include_patterns and not any(p in base for p in include_patterns):
                continue
            exclude_patterns = method_exclude.get(method_key, [])
            if exclude_patterns and any(p in base for p in exclude_patterns):
                continue
            method_hint = None
            block_hint = None
            m = pattern_re.search(base)
            if m:
                tag = m.group(1)
                method_hint, block_hint = extract_method_from_tag(tag)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if "permutation" not in obj:
                        continue
                    label = normalize_method_label(obj.get("method"), block_hint)
                    if label is None:
                        label = method_hint
                    if label is None and block_hint is not None:
                        label = format_method_label("semi_ar", block_hint)
                    if label is None:
                        label = obj.get("method") or "unknown"
                    obj["method"] = label
                    obj["status"] = status
                    items.append(obj)
    return items

def normalize_method_key(tag: str) -> str:
    if not tag:
        return ""
    if tag.startswith("semi_ar"):
        return "semi_ar"
    if tag.startswith("halton"):
        return "halton"
    if tag.startswith("conv"):
        return "conv"
    parts = tag.split("_", 1)
    return parts[0]

def parse_method_filter_args(values: List[str] | None) -> Dict[str, List[str]]:
    parsed: Dict[str, List[str]] = {}
    if not values:
        return parsed
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Method filter '{raw}' must use format method=substring")
        method, substr = raw.split("=", 1)
        method = method.strip()
        substr = substr.strip()
        if not method or not substr:
            raise ValueError(f"Invalid method filter entry: {raw}")
        parsed.setdefault(method, []).append(substr)
    return parsed

def parse_perm(s: str) -> np.ndarray:
    return np.fromstring(s, sep=" ", dtype=np.int32)

def compute_ar_similarity_stats(perm: np.ndarray) -> Dict[str, float]:
    L = int(perm.size)
    steps = np.arange(L, dtype=np.int32)
    diff1 = perm - steps
    diff2 = np.diff(diff1) if L > 1 else np.zeros(0, dtype=np.int32)

    def summarize(arr: np.ndarray, prefix: str) -> Dict[str, float]:
        if arr.size == 0:
            return {
                f"{prefix}_mean": 0.0,
                f"{prefix}_std": 0.0,
                f"{prefix}_max": 0.0,
                f"{prefix}_abs_mean": 0.0,
                f"{prefix}_abs_std": 0.0,
                f"{prefix}_abs_max": 0.0,
            }
        abs_arr = np.abs(arr)
        return {
            f"{prefix}_mean": float(np.mean(arr)),
            f"{prefix}_std": float(np.std(arr)),
            f"{prefix}_max": float(np.max(arr)),
            f"{prefix}_abs_mean": float(np.mean(abs_arr)),
            f"{prefix}_abs_std": float(np.std(abs_arr)),
            f"{prefix}_abs_max": float(np.max(abs_arr)),
        }

    stats = {}
    stats.update(summarize(diff1, "diff1"))
    stats.update(summarize(diff2, "diff2"))
    stats["length"] = L
    return stats

def infer_final_dir(in_dir: str) -> str:
    norm = os.path.normpath(in_dir)
    base = os.path.basename(norm)
    root = os.path.dirname(norm) if base == "raw" else norm
    if not root:
        root = "."
    return os.path.join(root, "final")

def plot_histograms(df: pd.DataFrame, metric: str, status: str, out_pdf: str, out_png: str, bins: int = 40):
    if df.empty:
        return False
    plt.figure(figsize=(6,4))
    for method, g in df.groupby("method"):
        values = g[metric].dropna().values
        if values.size == 0:
            continue
        plt.hist(values, bins=bins, alpha=0.5, label=method, density=True)
    plt.xlabel(metric)
    plt.ylabel("Density")
    plt.title(f"{status.capitalize()} permutations - {metric}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.close()
    return True

def method_filter_set(methods: List[str] | None, semi_ar_block_size: int | None) -> List[str] | None:
    if not methods:
        return None
    normalized: List[str] = []
    for m in methods:
        block_hint = semi_ar_block_size if m.strip().startswith("semi_ar") else None
        norm = normalize_method_label(m, block_hint)
        if norm is None:
            raise ValueError(f"Unable to interpret method name: {m}")
        if norm not in normalized:
            normalized.append(norm)
    return normalized

def main():
    ap = argparse.ArgumentParser(description="Analyze permutation similarity to left-to-right AR order.")
    ap.add_argument("--in_dir", type=str, default="results_n100/raw",
                    help="Directory containing success_perms_*.jsonl files.")
    ap.add_argument("--raw_out_dir", type=str, default=None,
                    help="Directory to write CSV (default: in_dir).")
    ap.add_argument("--final_out_dir", type=str, default=None,
                    help="Directory to write plots (default: sibling final/).")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="Optional list of methods to include (e.g., confidence random margin conv \"semi_ar (16)\" halton).")
    ap.add_argument("--semi_ar_block_size", type=int, default=None,
                    help="Block size hint used when filtering with method 'semi_ar'.")
    ap.add_argument("--csv_name", type=str, default="perm_ar_similarity_metrics.csv",
                    help="CSV filename for metric dump.")
    ap.add_argument("--status", type=str, choices=["success", "failed", "both"], default="both",
                    help="Which permutation outcomes to include.")
    ap.add_argument("--hist_bins", type=int, default=40, help="Histogram bins.")
    ap.add_argument("--seed", type=int, default=1234, help="Shuffle seed when sampling (unused currently).")
    ap.add_argument("--method_file_filter", action="append", default=[],
                    help="Limit permutation JSONLs for specific methods, format method=substring (e.g., confidence=_of04).")
    ap.add_argument("--method_file_exclude", action="append", default=[],
                    help="Exclude permutation JSONLs for specific methods, format method=substring (e.g., confidence=_of08).")
    args = ap.parse_args()

    if args.status == "both":
        statuses = ["success", "failed"]
    else:
        statuses = [args.status]

    include_map = parse_method_filter_args(args.method_file_filter)
    exclude_map = parse_method_filter_args(args.method_file_exclude)

    items = load_perm_records(args.in_dir, statuses, include_map, exclude_map)
    if not items:
        raise FileNotFoundError(f"No permutation records (statuses={statuses}) found under {args.in_dir}")

    method_filters = method_filter_set(args.methods, args.semi_ar_block_size)
    if method_filters:
        items = [obj for obj in items if any(method_matches(obj.get("method"), mf) for mf in method_filters)]
        if not items:
            raise ValueError("No success permutations left after method filtering.")

    rows: List[Dict[str, Any]] = []
    for obj in items:
        perm = parse_perm(obj["permutation"])
        stats = compute_ar_similarity_stats(perm)
        row = {
            "method": obj.get("method", "unknown"),
            "idx": obj.get("idx"),
            "trial": obj.get("trial"),
            "status": obj.get("status", "success"),
        }
        row.update(stats)
        rows.append(row)

    df = pd.DataFrame(rows)

    raw_out_dir = args.raw_out_dir or args.in_dir
    os.makedirs(raw_out_dir, exist_ok=True)
    csv_path = os.path.join(raw_out_dir, args.csv_name)
    df.to_csv(csv_path, index=False)
    print(f"[CSV] wrote metrics to {csv_path}")

    final_out_dir = args.final_out_dir or infer_final_dir(args.in_dir)
    os.makedirs(final_out_dir, exist_ok=True)

    hist_metrics: List[Tuple[str, str]] = [
        ("diff1_abs_mean", "Global deviation |perm-step| mean"),
        ("diff2_abs_mean", "Local deviation |Δ(perm-step)| mean"),
    ]
    for status in statuses:
        df_status = df[df["status"] == status]
        if df_status.empty:
            print(f"[WARN] No data for status={status}; skipping plots.")
            continue
        for metric, desc in hist_metrics:
            pdf_path = os.path.join(final_out_dir, f"perm_{metric}_{status}_hist.pdf")
            png_path = os.path.join(final_out_dir, f"perm_{metric}_{status}_hist.png")
            if plot_histograms(df_status, metric, status, pdf_path, png_path, bins=args.hist_bins):
                print(f"[PLOT] saved {pdf_path} and {png_path}")

if __name__ == "__main__":
    main()
