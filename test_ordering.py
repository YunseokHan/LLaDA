#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import json
import glob
import math
import argparse
import random
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch = None
load_dataset = None
AutoTokenizer = None
AutoModel = None

# ====== Special token ids (LLaDA) ======
MASK_TOKEN_ID = 126336  # LLaDA [MASK]
EOS_TOKEN_ID  = 126081
EOT_TOKEN_ID  = 126348

# =========================
# Utils
# =========================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def simple_instruction(problem: str) -> str:
    return (
        "Solve the following math problem.\n"
        "You may reason in free form. At the very end, write the final numeric result "
        "wrapped exactly as \\box{...}. Put only the number inside the box (e.g., 18 not 18.0).\n\n"
        f"Problem: {problem}\n"
    )

_BOX_RE = re.compile(r"\\box(?:ed)?\s*\{([^{}]+)\}")
def extract_after_answer(text: str) -> str:
    m = _BOX_RE.search(text)
    if m:
        return m.group(1).strip()
    # fallback for legacy "Answer:" style
    key = "Answer:"
    idx = text.find(key)
    if idx != -1:
        tail = text[idx + len(key):]
        tail = tail.splitlines()[0] if "\n" in tail else tail
        return tail.strip()
    return text.strip()

_GSM_RE = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)")
def parse_gsm8k_truth(ans_text: str) -> str:
    m = _GSM_RE.search(ans_text)
    if m:
        return m.group(1).strip()
    if "####" in ans_text:
        tail = ans_text.split("####", 1)[1].strip()
        return tail.split()[0].strip()
    return ans_text.strip()

_NUM_RE = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s*$")
def exact_number_match(pred_answer: str, true_answer: str) -> int:
    p = pred_answer.strip()
    t = true_answer.strip()
    if not _NUM_RE.match(p):
        return 0
    return int(p == t)

def build_chat_prompts(tokenizer, problems: List[str]) -> List[str]:
    messages = [{"role": "user", "content": simple_instruction(p)} for p in problems]
    prompts = [
        tokenizer.apply_chat_template([m], add_generation_prompt=True, tokenize=False)
        for m in messages
    ]
    return prompts

def forward_logits(model, x, attention_mask=None, cfg_scale: float = 0.0):
    if torch is None:
        raise RuntimeError("PyTorch is required to run decoding.")
    with torch.no_grad():
        if cfg_scale <= 0.0:
            return model(x, attention_mask=attention_mask).logits
        prompt_index = (x != MASK_TOKEN_ID)
        un_x = x.clone()
        un_x[prompt_index] = MASK_TOKEN_ID
        x_cat = torch.cat([x, un_x], dim=0)
        attn_cat = torch.cat([attention_mask, attention_mask], dim=0) if attention_mask is not None else None
        logits_cat = model(x_cat, attention_mask=attn_cat).logits
        logits, un_logits = torch.chunk(logits_cat, 2, dim=0)
        return un_logits + (cfg_scale + 1.0) * (logits - un_logits)

def chunk_indices_by_k(perm: np.ndarray, k: int) -> List[np.ndarray]:
    chunks = []
    n = len(perm)
    for i in range(0, n, k):
        chunks.append(perm[i: min(i + k, n)])
    return chunks

def generate_k_per_step_with_perm(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,          # (B, Lp)
    attention_mask: torch.Tensor,      # (B, Lp)
    perm: np.ndarray,                  # (gen_length,)
    gen_length: int,
    num_steps: int,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    block_eos: bool = False,
    device: str = "cuda",
):
    if torch is None:
        raise RuntimeError("PyTorch is required to run decoding.")
    with torch.no_grad():
        B, Lp = prompt_ids.shape
        assert len(perm) == gen_length

        x = torch.full((B, Lp + gen_length), MASK_TOKEN_ID, dtype=torch.long, device=device)
        x[:, :Lp] = prompt_ids
        attn = torch.cat(
            [attention_mask, torch.ones((B, gen_length), dtype=attention_mask.dtype, device=device)],
            dim=-1
        ) if attention_mask is not None else None

        start = Lp
        k = max(1, math.ceil(gen_length / max(1, num_steps)))
        perm_chunks = chunk_indices_by_k(perm, k)

        for rel_chunk in perm_chunks:
            logits = forward_logits(model, x, attn, cfg_scale=cfg_scale)  # (B, L, V)
            if block_eos:
                for rel_pos in rel_chunk:
                    abs_pos = start + int(rel_pos)
                    logits[:, abs_pos, EOS_TOKEN_ID] = -torch.inf
                    logits[:, abs_pos, EOT_TOKEN_ID] = -torch.inf

            for rel_pos in rel_chunk:
                abs_pos = start + int(rel_pos)
                pos_logits = logits[:, abs_pos, :]  # (B, V)
                if temperature and temperature > 0.0:
                    u = torch.clamp(torch.rand_like(pos_logits), 1e-20, 1 - 1e-20)
                    g = -torch.log(-torch.log(u))
                    sampled = torch.argmax((pos_logits / temperature) + g, dim=-1)
                else:
                    sampled = torch.argmax(pos_logits, dim=-1)
                x[torch.arange(B, device=device), torch.full((B,), abs_pos, device=device)] = sampled

        return x

def generate_confidence_decoding(
    model,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_length: int,
    cfg_scale: float = 0.0,
    block_eos: bool = False,
    device: str = "cuda",
):
    """
    Decode tokens by repeatedly selecting the mask position whose top-1 logit is largest.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required to run decoding.")
    with torch.no_grad():
        B, Lp = prompt_ids.shape
        if B != 1:
            raise ValueError("confidence decoding expects batch size 1 per call.")

        x = torch.full((B, Lp + gen_length), MASK_TOKEN_ID, dtype=torch.long, device=device)
        x[:, :Lp] = prompt_ids
        attn = torch.cat(
            [attention_mask, torch.ones((B, gen_length), dtype=attention_mask.dtype, device=device)],
            dim=-1
        ) if attention_mask is not None else None

        start = Lp
        remaining = list(range(gen_length))
        decode_order: List[int] = []

        while remaining:
            logits = forward_logits(model, x, attn, cfg_scale=cfg_scale)
            if block_eos:
                for rel_pos in remaining:
                    abs_pos = start + rel_pos
                    logits[:, abs_pos, EOS_TOKEN_ID] = -torch.inf
                    logits[:, abs_pos, EOT_TOKEN_ID] = -torch.inf

            best_scores = []
            best_tokens = []
            for rel_pos in remaining:
                abs_pos = start + rel_pos
                pos_logits = logits[:, abs_pos, :]
                val, tok = torch.max(pos_logits, dim=-1)
                best_scores.append(val.squeeze(0))
                best_tokens.append(tok.squeeze(0))

            score_tensor = torch.stack(best_scores)  # (num_remaining,)
            best_idx = int(torch.argmax(score_tensor).item())
            chosen_rel = remaining.pop(best_idx)
            chosen_abs = start + chosen_rel
            chosen_token = best_tokens[best_idx]
            x[0, chosen_abs] = chosen_token
            decode_order.append(int(chosen_rel))

        return x, decode_order

def decode_generated_only(tokenizer, full_ids: torch.Tensor, prompt_len: int) -> List[str]:
    return tokenizer.batch_decode(full_ids[:, prompt_len:], skip_special_tokens=True)

def ks_from_trials(trials: int) -> List[int]:
    max_pow = int(math.log2(trials))
    return [1 << i for i in range(max_pow + 1)]

def perm_for_trial(gen_length: int, trial_idx: int, seed_base: int, problem_idx: int) -> np.ndarray:
    """
    Deterministic permutation per (problem_idx, trial_idx) to avoid overlap across shards.
    """
    rng = np.random.default_rng(seed_base * 1000003 + problem_idx * 4099 + trial_idx)
    p = np.arange(gen_length, dtype=np.int32)
    rng.shuffle(p)
    return p

def perm_for_trial_blockwise(gen_length: int, trial_idx: int, seed_base: int, problem_idx: int, block_size: int) -> np.ndarray:
    """
    Shuffle positions within each contiguous block (size block_size) while keeping block order intact.
    """
    rng = np.random.default_rng(seed_base * 1000003 + problem_idx * 4099 + trial_idx)
    p = np.arange(gen_length, dtype=np.int32)
    block = max(1, block_size)
    for start in range(0, gen_length, block):
        end = min(start + block, gen_length)
        rng.shuffle(p[start:end])
    return p

def format_method_label(method: str, semi_ar_block_size: int = None) -> str:
    if method == "semi_ar" and semi_ar_block_size is not None:
        return f"semi_ar ({semi_ar_block_size})"
    return method

def method_file_prefix(method: str, semi_ar_block_size: int = None) -> str:
    if method == "semi_ar" and semi_ar_block_size is not None:
        return f"semi_ar_bs{semi_ar_block_size}"
    return method

_SEMI_AR_BS_RE = re.compile(r"semi_ar_bs(\d+)")
def parse_block_size_from_filename(path: str) -> int:
    m = _SEMI_AR_BS_RE.search(os.path.basename(path))
    if m:
        return int(m.group(1))
    return None

# =========================
# W&B helpers
# =========================
def wandb_init(args, job_type: str, run_name: str, group: str = None, config: Dict[str, Any] = None, tags: List[str] = None):
    """Initialize wandb unless disabled."""
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except Exception as e:
        print(f"[WARN] wandb import failed ({e}); proceeding without logging.")
        return None

    settings = wandb.Settings(start_method="thread")
    run = wandb.init(
        project=args.wandb_project,
        entity=getattr(args, "wandb_entity", None),
        mode=args.wandb_mode,              # online / offline
        group=group,
        job_type=job_type,
        name=run_name,
        settings=settings,
        config=config or {},
        tags=tags or [],
    )
    return run

def wandb_log(run, data: Dict[str, Any], step: int = None):
    if run is None:
        return
    try:
        import wandb
        if step is None:
            wandb.log(data)
        else:
            wandb.log(data, step=step)
    except Exception as e:
        print(f"[WARN] wandb.log failed: {e}")

def wandb_log_artifact(run, path: str, art_type: str, name: str = None):
    if run is None:
        return
    try:
        import wandb
        art = wandb.Artifact(name or os.path.basename(path), type=art_type)
        art.add_file(path)
        run.log_artifact(art)
    except Exception as e:
        print(f"[WARN] artifact upload failed for {path}: {e}")

# =========================
# Worker mode
# =========================
def run_worker(args):
    global torch, load_dataset, AutoTokenizer, AutoModel
    import torch as _torch
    from datasets import load_dataset as _load_dataset
    from transformers import AutoTokenizer as _AutoTokenizer, AutoModel as _AutoModel

    torch = _torch
    load_dataset = _load_dataset
    AutoTokenizer = _AutoTokenizer
    AutoModel = _AutoModel

    seed_everything(args.seed)

    # Load GSM8K
    ds = load_dataset("openai/gsm8k", "main")
    dtest = ds["test"]
    # Pick first n problems (default 10)
    n = min(args.n, len(dtest))
    questions = [dtest[i]["question"] for i in range(n)]
    truths_raw = [dtest[i]["answer"] for i in range(n)]
    truths = [parse_gsm8k_truth(a) for a in truths_raw]
    idxs = list(range(n))

    # Model & tokenizer
    device = args.device
    model = AutoModel.from_pretrained(
        args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    assert tokenizer.pad_token_id != MASK_TOKEN_ID, "pad_token_id must differ from MASK_TOKEN_ID."

    # Prompts (pre-tokenize once)
    chat_prompts = build_chat_prompts(tokenizer, questions)
    enc = tokenizer(chat_prompts, add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids_all = enc["input_ids"].to(device)
    attention_all = enc["attention_mask"].to(device)
    prompt_len = input_ids_all.size(1)

    # Decoding config
    gen_length = args.gen_length
    # For sequential behavior, set num_steps = gen_length (k=1)
    num_steps = args.num_steps if args.num_steps > 0 else gen_length
    if args.force_k1:
        num_steps = gen_length
    method_label = format_method_label(args.method, args.semi_ar_block_size)

    os.makedirs(args.out_dir, exist_ok=True)
    result_rows: List[Dict[str, Any]] = []
    success_lines: List[str] = []
    failed_lines: List[str] = []

    # Trial sharding
    all_trials = list(range(args.trials))
    shard_trials = [t for t in all_trials if (t % args.num_shards) == args.trial_shard]

    print(f"[INFO] mode=worker method={args.method} n={n} trials={args.trials} "
          f"num_shards={args.num_shards} this_shard={args.trial_shard} shard_trials={len(shard_trials)}")

    # ==== W&B init ====
    run_name = args.wandb_run_name or f"{args.method}-sh{args.trial_shard:02d}-of{args.num_shards:02d}"
    group = args.wandb_group or f"passk-{args.model_name.replace('/','_')}"
    run = wandb_init(
        args,
        job_type="worker",
        run_name=run_name,
        group=group,
        config={
            "mode": "worker",
            "method": args.method,
            "semi_ar_block_size": args.semi_ar_block_size,
            "model_name": args.model_name,
            "n": n,
            "trials_total": args.trials,
            "trials_this_shard": len(shard_trials),
            "num_shards": args.num_shards,
            "trial_shard": args.trial_shard,
            "gen_length": gen_length,
            "num_steps": num_steps,
            "force_k1": bool(args.force_k1),
            "batch_size": args.batch_size,
            "cfg_scale": args.cfg_scale,
            "block_eos": bool(args.block_eos),
            "device": args.device,
        },
        tags=[args.method, "worker"]
    )

    # Batch over problems for each trial (keeps GPU busy)
    B = args.batch_size

    for t in tqdm(shard_trials, desc=f"Trials(shard={args.trial_shard})", dynamic_ncols=True):
        # Set per-trial randomness for AR sampling
        if args.method == "ar":
            seed_everything(args.seed + 17 * t + 101)
            perm = np.arange(gen_length, dtype=np.int32)  # left-to-right
            temperature = 0.7
        elif args.method in {"random", "semi_ar"}:
            temperature = 0.0
        elif args.method == "confidence":
            temperature = 0.7
        else:
            raise ValueError("--method must be 'ar', 'random', 'semi_ar', or 'confidence'")

        correct_count_trial = 0

        # Process problems in batches
        for start in range(0, n, B):
            end = min(start + B, n)
            batch_ids = input_ids_all[start:end]
            batch_attn = attention_all[start:end]
            batch_truth = truths[start:end]
            batch_idxs = idxs[start:end]

            if args.method == "random":
                perms = [perm_for_trial(gen_length, t, args.seed, problem_idx=i) for i in batch_idxs]
            elif args.method == "semi_ar":
                perms = [
                    perm_for_trial_blockwise(gen_length, t, args.seed, problem_idx=i, block_size=args.semi_ar_block_size)
                    for i in batch_idxs
                ]
            elif args.method == "ar":
                perms = [perm for _ in range(len(batch_idxs))]
            else:
                perms = [None for _ in range(len(batch_idxs))]

            # Run each problem separately (permutation differs case-by-case)
            for b in range(len(batch_idxs)):
                # Slice single example view (shape (1, L))
                p_ids = batch_ids[b:b+1]
                p_attn = batch_attn[b:b+1]
                perm_b = perms[b]

                if args.method == "confidence":
                    full_ids, conf_order = generate_confidence_decoding(
                        model=model,
                        prompt_ids=p_ids,
                        attention_mask=p_attn,
                        gen_length=gen_length,
                        cfg_scale=args.cfg_scale,
                        block_eos=args.block_eos,
                        device=device,
                    )
                else:
                    if perm_b is None:
                        raise RuntimeError("Permutation expected but not provided.")
                    full_ids = generate_k_per_step_with_perm(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_ids=p_ids,
                        attention_mask=p_attn,
                        perm=perm_b,
                        gen_length=gen_length,
                        num_steps=num_steps,
                        temperature=temperature,
                        cfg_scale=args.cfg_scale,
                        block_eos=args.block_eos,
                        device=device,
                    )
                gen_txt = decode_generated_only(tokenizer, full_ids, prompt_len)[0]
                pred = extract_after_answer(gen_txt)
                corr = exact_number_match(pred, batch_truth[b])

                row = {
                    "idx": batch_idxs[b],
                    "trial": t,
                    "method": method_label,
                    "pred": pred,
                    "truth": batch_truth[b],
                    "correct": corr,
                }
                if args.method == "random":
                    row["permutation"] = " ".join(map(str, perms[b].tolist()))
                elif args.method == "semi_ar":
                    row["permutation"] = " ".join(map(str, perms[b].tolist()))
                elif args.method == "confidence":
                    row["permutation"] = " ".join(map(str, conf_order))
                else:
                    row["permutation"] = ""  # left-to-right implicit
                if (
                    args.save_success_perms
                    and corr == 1
                    and args.method in {"random", "semi_ar", "confidence"}
                ):
                    success_lines.append(json.dumps({
                        "idx": batch_idxs[b],
                        "trial": t,
                        "method": method_label,
                        "permutation": row["permutation"],
                    }, ensure_ascii=False))
                if (
                    args.save_failed_perms
                    and corr == 0
                    and args.method in {"random", "semi_ar", "confidence"}
                ):
                    failed_lines.append(json.dumps({
                        "idx": batch_idxs[b],
                        "trial": t,
                        "method": method_label,
                        "permutation": row["permutation"],
                    }, ensure_ascii=False))
                result_rows.append(row)
                correct_count_trial += int(corr)

        # ---- per-trial wandb log ----
        trial_acc = correct_count_trial / float(n) if n > 0 else 0.0
        wandb_log(run, {
            "trial": t,
            "trial_acc": trial_acc,
            "trial_correct": correct_count_trial,
            "trial_total_problems": n,
        }, step=t)

    # Save shard outputs
    fname_method = method_file_prefix(args.method, args.semi_ar_block_size)
    tag = f"{fname_method}_sh{args.trial_shard:02d}_of{args.num_shards:02d}"
    out_csv = os.path.join(args.out_dir, f"trials_{tag}.csv")
    pd.DataFrame(result_rows).to_csv(out_csv, index=False)
    print(f"[DONE] wrote {len(result_rows)} rows to {out_csv}")
    wandb_log_artifact(run, out_csv, art_type="results", name=f"trials_{tag}")

    if args.method in {"random", "semi_ar", "confidence"} and args.save_success_perms:
        succ_path = os.path.join(args.out_dir, f"success_perms_{tag}.jsonl")
        with open(succ_path, "w", encoding="utf-8") as f:
            for line in success_lines:
                f.write(line + "\n")
        print(f"[DONE] wrote successful permutations to {succ_path}")
        wandb_log_artifact(run, succ_path, art_type="metadata", name=f"success_perms_{tag}")
    if args.method in {"random", "semi_ar", "confidence"} and args.save_failed_perms:
        fail_path = os.path.join(args.out_dir, f"failed_perms_{tag}.jsonl")
        with open(fail_path, "w", encoding="utf-8") as f:
            for line in failed_lines:
                f.write(line + "\n")
        print(f"[DONE] wrote failed permutations to {fail_path}")
        wandb_log_artifact(run, fail_path, art_type="metadata", name=f"failed_perms_{tag}")

    # ---- final shard summary ----
    if len(result_rows) > 0:
        df_shard = pd.DataFrame(result_rows)
        shard_acc = df_shard["correct"].mean()
        wandb_log(run, {
            "shard_acc_overall": float(shard_acc),
            "shard_rows": int(len(df_shard)),
            "shard_trials": int(len(shard_trials)),
        })

    if run is not None:
        try:
            import wandb
            run.finish()
        except Exception:
            pass

# =========================
# Aggregate mode
# =========================
def compute_passk(df: pd.DataFrame, trials: int, ks: List[int]) -> pd.DataFrame:
    """
    df columns: idx, trial, method, correct
    Returns a dataframe: columns: method, k, pass_at_k
    """
    rows = []
    for method, g in df.groupby("method"):
        # Build per-problem success boolean array over trial indices [0..trials-1]
        # If some trials missing (due to shard issues), treat missing as fail.
        pass_by_k = {k: [] for k in ks}
        for idx, gg in g.groupby("idx"):
            success = np.zeros(trials, dtype=np.int8)
            for _, r in gg.iterrows():
                t = int(r["trial"])
                if 0 <= t < trials:
                    success[t] = max(success[t], int(r["correct"]))
            cumsum_any = np.maximum.accumulate(success)  # running "any succeeded"
            for k in ks:
                k_cap = min(k, trials)
                v = 1 if cumsum_any[:k_cap].any() else 0
                pass_by_k[k].append(v)
        for k in ks:
            arr = pass_by_k[k]
            mean = float(np.mean(arr)) if len(arr) > 0 else 0.0
            rows.append({"method": method, "k": k, "pass_at_k": mean})
    return pd.DataFrame(rows)

def plot_passk(df_passk: pd.DataFrame, pdf_path: str, png_path: str = None):
    plt.figure(figsize=(6,4.2))
    for method, g in df_passk.groupby("method"):
        g = g.sort_values("k")
        plt.plot(g["k"].values, g["pass_at_k"].values, marker="o", label=method)
    plt.xscale("log", base=2)
    plt.xlabel("k (log2)")
    plt.ylabel("pass@k")
    plt.title("GSM8K pass@k by decoding method")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pdf_path)
    if png_path:
        plt.savefig(png_path)
    msg = f"[PLOT] saved {pdf_path}" if png_path is None else f"[PLOT] saved {pdf_path} and {png_path}"
    print(msg)

def load_trials_with_method_labels(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "method" not in df.columns:
        return df
    mask = df["method"] == "semi_ar"
    if mask.any():
        block_size = parse_block_size_from_filename(path)
        if block_size is not None:
            df.loc[mask, "method"] = format_method_label("semi_ar", block_size)
    return df

def infer_out_dir_from_in_dir(in_dir: str) -> str:
    norm_in = os.path.normpath(in_dir)
    base = os.path.basename(norm_in)
    root = os.path.dirname(norm_in) if base == "raw" else norm_in
    if root == "":
        root = "."
    return os.path.join(root, "final")

def run_aggregate(args):
    if not args.out_dir:
        args.out_dir = infer_out_dir_from_in_dir(args.in_dir)
        print(f"[INFO] out_dir not provided, defaulting to {args.out_dir}")

    # Collect all trial csvs produced by workers
    pattern = os.path.join(args.in_dir, "trials_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No shard CSVs found under: {pattern}")

    dfs = [load_trials_with_method_labels(p) for p in files]
    df = pd.concat(dfs, ignore_index=True)

    # Keep only first n problems if larger (safety)
    if args.n is not None and args.n > 0:
        df = df[df["idx"] < args.n].reset_index(drop=True)

    ks = ks_from_trials(args.trials)
    passk = compute_passk(df, trials=args.trials, ks=ks)
    os.makedirs(args.out_dir, exist_ok=True)

    out_csv = os.path.join(args.out_dir, "passk_summary.csv")
    passk.to_csv(out_csv, index=False)
    print(f"[DONE] wrote pass@k summary to {out_csv}")

    pdf_path = os.path.join(args.out_dir, "passk_plot.pdf")
    png_path = os.path.join(args.out_dir, "passk_plot.png")
    plot_passk(passk, pdf_path, png_path)

    # ==== W&B aggregate ====
    run_name = args.wandb_run_name or "aggregate"
    group = args.wandb_group or f"passk-{args.model_name.replace('/','_')}" if hasattr(args, "model_name") else None
    run = wandb_init(
        args,
        job_type="aggregate",
        run_name=run_name,
        group=group,
        config={
            "mode": "aggregate",
            "trials": args.trials,
            "n": args.n,
        },
        tags=["aggregate"]
    )

    # Log series as individual metrics (method/k)
    for method, g in passk.groupby("method"):
        g = g.sort_values("k")
        for _, r in g.iterrows():
            wandb_log(run, {f"{method}/pass@{int(r['k'])}": float(r["pass_at_k"])})
    # Upload artifacts and image
    wandb_log_artifact(run, out_csv, art_type="summary", name="passk_summary_csv")
    wandb_log_artifact(run, pdf_path, art_type="plot", name="passk_plot_pdf")
    # also log PNG for inline preview
    try:
        import wandb
        if run is not None:
            run.log({"passk_plot": wandb.Image(png_path)})
    except Exception as e:
        print(f"[WARN] wandb image log failed: {e}")

    if run is not None:
        try:
            import wandb
            run.finish()
        except Exception:
            pass

# =========================
# Main
# =========================
def main():
    p = argparse.ArgumentParser(description="Permutation/AR decoding pass@k on GSM8K")
    sub = p.add_subparsers(dest="mode", required=True)

    # Common W&B args helper
    def add_wandb_args(parser):
        parser.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"], default="online",
                            help="W&B logging mode (default: online)")
        parser.add_argument("--wandb_project", type=str, default="llada-passk",
                            help="W&B project name (default: llada-passk)")
        parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (team) name")
        parser.add_argument("--wandb_group", type=str, default=None, help="W&B run group")
        parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")

    # Worker
    pw = sub.add_parser("worker", help="Run decoding trials on a shard")
    pw.add_argument("--method", type=str, choices=["ar", "random", "semi_ar", "confidence"], required=True)
    pw.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    pw.add_argument("--n", type=int, default=10, help="Number of problems to evaluate (default 10).")
    pw.add_argument("--gen_length", type=int, default=256)
    pw.add_argument("--num_steps", type=int, default=0, help="If 0, use gen_length. If --force_k1, overridden to gen_length.")
    pw.add_argument("--force_k1", action="store_true", help="Force k=1 per step (num_steps=gen_length). Recommended.")
    pw.add_argument("--batch_size", type=int, default=4)
    pw.add_argument("--cfg_scale", type=float, default=0.0)
    pw.add_argument("--block_eos", action="store_true")
    pw.add_argument("--seed", type=int, default=1234)
    pw.add_argument("--device", type=str, default="cuda")
    pw.add_argument("--semi_ar_block_size", type=int, default=16,
                    help="Block size used for semi-ar decoding (default: 16).")

    pw.add_argument("--trials", type=int, default=1024, help="Trials per problem.")
    pw.add_argument("--num_shards", type=int, default=1)
    pw.add_argument("--trial_shard", type=int, default=0)
    pw.add_argument("--out_dir", type=str, default="results/raw")
    pw.add_argument("--save_success_perms", action="store_true", help="Save successful permutations (random/semi_ar/confidence modes).")
    pw.add_argument("--save_failed_perms", action="store_true", help="Save failed permutations (random/semi_ar/confidence modes).")
    add_wandb_args(pw)

    # Aggregate
    pa = sub.add_parser("aggregate", help="Aggregate shards and plot pass@k")
    pa.add_argument("--in_dir", type=str, required=True, help="Directory containing shard CSVs (trials_*.csv).")
    pa.add_argument("--out_dir", type=str, default=None,
                    help="Directory to store summary outputs (default: derived from in_dir)")
    pa.add_argument("--trials", type=int, default=1024)
    pa.add_argument("--n", type=int, default=10, help="Number of problems included (safety).")
    # (optional) model name is only used for wandb group default
    pa.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    add_wandb_args(pa)

    args = p.parse_args()

    if args.mode == "worker":
        run_worker(args)
    else:
        run_aggregate(args)

if __name__ == "__main__":
    main()
