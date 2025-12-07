#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, glob, json, argparse, random, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SEMI_AR_BS_TAG_RE = re.compile(r"semi_ar(?:_bs)?(\d+)")
_SEMI_AR_LABEL_RE = re.compile(r"semi_ar\s*\((\d+)\)")
_PERM_FILE_RE = re.compile(r"success_perms_([A-Za-z0-9_.-]+)_sh\d+_of\d+\.jsonl$")

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

def parse_method_filter_args(values):
    parsed = {}
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

def load_success_perms(in_dir: str, method_include=None, method_exclude=None):
    """results/raw/success_perms_*.jsonl 파일들을 모아 [dict,…] 리스트로 반환."""
    method_include = method_include or {}
    method_exclude = method_exclude or {}
    pattern = os.path.join(in_dir, "success_perms_*.jsonl")
    paths = sorted(glob.glob(pattern))
    items = []
    pattern_re = re.compile(r"success_perms_([a-zA-Z0-9_]+)_")
    for p in paths:
        base = os.path.basename(p)
        method_key = ""
        m_key = _PERM_FILE_RE.match(base)
        if m_key:
            method_key = normalize_method_key(m_key.group(1))
        include_patterns = method_include.get(method_key, [])
        if include_patterns and not any(s in base for s in include_patterns):
            continue
        exclude_patterns = method_exclude.get(method_key, [])
        if exclude_patterns and any(s in base for s in exclude_patterns):
            continue
        method_hint = None
        block_hint = None
        m = pattern_re.search(base)
        if m:
            tag = m.group(1)
            method_hint, block_hint = extract_method_from_tag(tag)
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # 기대 포맷: {"idx": int, "trial": int, "permutation": "0 5 2 ..."}
                    if "permutation" in obj:
                        label = normalize_method_label(obj.get("method"), block_hint)
                        if label is None:
                            label = method_hint
                        if label is None and block_hint is not None:
                            label = format_method_label("semi_ar", block_hint)
                        if label is None:
                            label = obj.get("method") or "unknown"
                        obj["method"] = label
                        items.append(obj)
                except Exception:
                    pass
    return items

def parse_perm(s: str) -> np.ndarray:
    """공백 구분 permutation 문자열 -> np.array[int]"""
    return np.fromstring(s, sep=" ", dtype=np.int32)

def plot_perm(ax, perm: np.ndarray, title_extra=""):
    """
    한 개 permutation을 산점도로 그리기.
    x: step(0..L-1), y: token position(perm[step])
    색: token position (viridis)
    """
    L = int(perm.size)
    x = np.arange(L, dtype=np.int32)
    y = perm
    ax.scatter(x, y, c=y, s=6, cmap="viridis", edgecolors="none")
    ax.set_xlim(0, L - 1)
    ax.set_ylim(-1, L)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title(f"{L} steps, {L} length{title_extra}", fontsize=9)
    ax.tick_params(labelsize=8)

def main():
    ap = argparse.ArgumentParser(description="Visualize n×n successful permutations.")
    ap.add_argument("--in_dir", type=str, default="results/raw",
                    help="Directory containing success_perms_*.jsonl")
    ap.add_argument("--out_dir", type=str, default="results/final",
                    help="Output directory for the figure files")
    ap.add_argument("--num_row", type=int, default=4,
                    help="Grid rows/cols (n). Exactly n*n samples will be drawn.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--method", type=str, default=None,
                    help="Filter success permutations by method (e.g., random, margin, conv, halton).")
    ap.add_argument("--semi_ar_block_size", type=int, default=None,
                    help="Specify block size when --method semi_ar to disambiguate.")
    ap.add_argument("--method_file_filter", action="append", default=[],
                    help="Limit permutation JSONLs for specific methods, format method=substring (e.g., confidence=_of04).")
    ap.add_argument("--method_file_exclude", action="append", default=[],
                    help="Exclude permutation JSONLs for specific methods, format method=substring (e.g., confidence=_of08).")
    ap.add_argument("--show_title", action="store_true",
                    help="Display a global title on the figure (default: off).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    method_include = parse_method_filter_args(args.method_file_filter)
    method_exclude = parse_method_filter_args(args.method_file_exclude)
    items = load_success_perms(args.in_dir, method_include, method_exclude)
    total_items = len(items)
    method_filter = normalize_method_label(args.method, args.semi_ar_block_size)
    if args.method and method_filter is None:
        raise ValueError(f"Unable to interpret --method value: {args.method}")
    if method_filter:
        items = [obj for obj in items if method_matches(obj.get("method"), method_filter)]
    print(f"[INFO] loaded {total_items} success entries, {len(items)} after filtering")

    need = args.num_row * args.num_row
    if len(items) < need:
        raise ValueError(f"Not enough successful permutations: need {need}, found {len(items)} under {args.in_dir}")

    random.seed(args.seed)
    picked = random.sample(items, need)

    # 동적 캔버스 사이즈: 셀당 약 3.0×2.3 inch
    fig_w = max(3.0 * args.num_row, 6.0)
    fig_h = max(2.3 * args.num_row, 4.6)
    fig, axes = plt.subplots(args.num_row, args.num_row, figsize=(fig_w, fig_h), dpi=args.dpi)

    # axes를 2D로 보장
    if args.num_row == 1:
        axes = np.array([[axes]])
    axes = np.asarray(axes)

    for i, obj in enumerate(picked):
        r, c = divmod(i, args.num_row)
        ax = axes[r, c]
        perm = parse_perm(obj["permutation"])
        title_extra = f"  (idx {obj.get('idx','?')}, t {obj.get('trial','?')})"
        plot_perm(ax, perm, title_extra=title_extra)
        if r == args.num_row - 1:  # 마지막 행만 x라벨
            ax.set_xlabel("Step", fontsize=9)
        if c == 0:                 # 첫 열만 y라벨
            ax.set_ylabel("Token Position", fontsize=9)

    # 남는 칸은 없음(정확히 n*n 샘플)
    if args.show_title:
        if method_filter:
            plt.suptitle(
                f"Successful permutations [{method_filter}] (random sample, {args.num_row}×{args.num_row})",
                fontsize=12,
            )
        else:
            plt.suptitle(
                f"Successful permutations (random sample, {args.num_row}×{args.num_row})",
                fontsize=12,
            )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        plt.tight_layout()

    method_tag = None
    if method_filter:
        method_tag = method_tag_from_label(method_filter)
    suffix = f"_{method_tag}" if method_tag else ""
    base_name = f"success_perm_grid{suffix}_{args.num_row}x{args.num_row}"
    out_pdf = os.path.join(args.out_dir, f"{base_name}.pdf")
    out_png = os.path.join(args.out_dir, f"{base_name}.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    print(f"[SAVED] {out_pdf}")
    print(f"[SAVED] {out_png}")

if __name__ == "__main__":
    main()
