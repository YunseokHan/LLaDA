#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from test_ordering import (
    MASK_TOKEN_ID,
    EOS_TOKEN_ID,
    EOT_TOKEN_ID,
    build_chat_prompts,
    extract_after_answer,
    parse_gsm8k_truth,
    exact_number_match,
)


@dataclass
class ProblemRecord:
    idx: int
    question: str
    truth: str
    prompt_ids: List[int]
    attention_mask: List[int]
    prompt_len: int
    baseline_tokens: List[int]
    baseline_log_probs: List[float]
    baseline_text: str
    baseline_pred: str
    baseline_correct: int
    baseline_avg_log_prob: float
    baseline_perplexity: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def forward_logits(model, x, attention_mask=None, cfg_scale: float = 0.0):
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


def init_generation_state(
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    device = prompt_ids.device
    B, Lp = prompt_ids.shape
    x = torch.full((B, Lp + gen_length), MASK_TOKEN_ID, dtype=torch.long, device=device)
    x[:, :Lp] = prompt_ids
    if attention_mask is not None:
        suffix_mask = torch.ones((B, gen_length), dtype=attention_mask.dtype, device=device)
        attn = torch.cat([attention_mask, suffix_mask], dim=-1)
    else:
        attn = None
    return x, attn, Lp


def decode_confidence_positions(
    model,
    x: torch.Tensor,
    attention_mask: torch.Tensor,
    start: int,
    rel_positions: List[int],
    cfg_scale: float = 0.0,
    block_eos: bool = False,
) -> Dict[int, float]:
    remaining = list(rel_positions)
    log_prob_map: Dict[int, float] = {}
    if not remaining:
        return log_prob_map

    finfo = torch.finfo(torch.float32)
    while remaining:
        logits = forward_logits(model, x, attention_mask, cfg_scale=cfg_scale)
        best_scores = []
        best_tokens = []
        best_log_probs = []
        for rel_pos in remaining:
            abs_pos = start + rel_pos
            pos_logits = logits[:, abs_pos, :].to(torch.float32)
            if block_eos:
                pos_logits[:, EOS_TOKEN_ID] = finfo.min
                pos_logits[:, EOT_TOKEN_ID] = finfo.min
            log_probs = F.log_softmax(pos_logits, dim=-1)
            val, tok = torch.max(pos_logits, dim=-1)
            best_scores.append(val.squeeze(0))
            best_tokens.append(tok.squeeze(0))
            best_log_probs.append(log_probs[0, tok.squeeze(0)])

        score_tensor = torch.stack(best_scores)
        best_idx = int(torch.argmax(score_tensor).item())
        chosen_rel = remaining.pop(best_idx)
        chosen_abs = start + chosen_rel
        chosen_token = best_tokens[best_idx]
        chosen_log_prob = float(best_log_probs[best_idx].item())
        x[0, chosen_abs] = chosen_token
        log_prob_map[chosen_rel] = chosen_log_prob

    return log_prob_map


def decode_block_in_order(
    model,
    x: torch.Tensor,
    attention_mask: torch.Tensor,
    start: int,
    ordered_positions: Sequence[int],
    cfg_scale: float = 0.0,
    block_eos: bool = False,
) -> Dict[int, float]:
    log_prob_map: Dict[int, float] = {}
    if not ordered_positions:
        return log_prob_map
    finfo = torch.finfo(torch.float32)

    for rel_pos in ordered_positions:
        logits = forward_logits(model, x, attention_mask, cfg_scale=cfg_scale)
        abs_pos = start + rel_pos
        pos_logits = logits[:, abs_pos, :].to(torch.float32)
        if block_eos:
            pos_logits[:, EOS_TOKEN_ID] = finfo.min
            pos_logits[:, EOT_TOKEN_ID] = finfo.min
        log_probs = F.log_softmax(pos_logits, dim=-1)
        max_val, tok = torch.max(pos_logits, dim=-1)
        chosen_token = tok.squeeze(0)
        chosen_log_prob = float(log_probs[0, chosen_token].item())
        x[0, abs_pos] = chosen_token
        log_prob_map[rel_pos] = chosen_log_prob

    return log_prob_map


def generate_baseline(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_length: int,
    cfg_scale: float = 0.0,
    block_eos: bool = False,
) -> Tuple[List[int], List[float], str, str, int, float, float]:
    x, attn, start = init_generation_state(prompt_ids, attention_mask, gen_length)
    rel_positions = list(range(gen_length))
    log_prob_map = decode_confidence_positions(
        model,
        x,
        attn,
        start,
        rel_positions,
        cfg_scale=cfg_scale,
        block_eos=block_eos,
    )
    full_tokens = x[0, start:start + gen_length].tolist()
    log_prob_list = [float(log_prob_map[i]) for i in range(gen_length)]
    avg_log_prob = float(sum(log_prob_list) / gen_length)
    perplexity = float(math.exp(-avg_log_prob))
    generated_text = tokenizer.batch_decode(x[:, start:], skip_special_tokens=True)[0]
    pred = extract_after_answer(generated_text)
    return (
        full_tokens,
        log_prob_list,
        generated_text,
        pred,
        start,
        avg_log_prob,
        perplexity,
    )


def chunk_evenly(seq: List[Any], num_chunks: int) -> List[List[Any]]:
    chunks = [[] for _ in range(num_chunks)]
    for idx, item in enumerate(seq):
        chunks[idx % num_chunks].append(item)
    return chunks


def prepare_dataset(num_problems: int) -> Tuple[List[str], List[str], List[int]]:
    ds = load_dataset("openai/gsm8k", "main")
    test_split = ds["test"]
    n = min(num_problems, len(test_split))
    questions = [test_split[i]["question"] for i in range(n)]
    truths_raw = [test_split[i]["answer"] for i in range(n)]
    truths = [parse_gsm8k_truth(ans) for ans in truths_raw]
    idxs = list(range(n))
    return questions, truths, idxs


def determine_target_blocks(
    gen_length: int,
    block_size: int,
    block_indices: List[int],
    block_offset: int,
    block_stride: int,
    max_blocks: int,
) -> List[int]:
    total_blocks = math.ceil(gen_length / block_size)
    if block_indices:
        base = [b for b in block_indices if 0 <= b < total_blocks]
    else:
        stride = max(1, block_stride)
        start_idx = min(max(block_offset, 0), max(total_blocks - 1, 0))
        base = list(range(start_idx, total_blocks, stride))
    if max_blocks > 0:
        base = base[:max_blocks]
    return base


def build_permutation_tasks(
    problems: List[ProblemRecord],
    gen_length: int,
    block_size: int,
    target_blocks: List[int],
) -> List[Dict[str, Any]]:
    perm_cache: Dict[int, List[Tuple[int, ...]]] = {}
    tasks: List[Dict[str, Any]] = []
    for block_index in target_blocks:
        block_start = block_index * block_size
        block_len = min(block_size, gen_length - block_start)
        if block_len <= 0:
            continue
        if block_len not in perm_cache:
            perm_cache[block_len] = list(itertools.permutations(range(block_len)))
        perms = perm_cache[block_len]
        block_positions = [block_start + i for i in range(block_len)]
        for perm_idx, perm in enumerate(perms):
            for prob_idx, record in enumerate(problems):
                tasks.append({
                    "problem_local_idx": prob_idx,
                    "dataset_idx": record.idx,
                    "block_index": block_index,
                    "block_number": block_index + 1,
                    "block_start": block_start,
                    "block_end": block_start + block_len,
                    "block_positions": block_positions,
                    "permutation": list(perm),
                    "permutation_idx": perm_idx,
                })
    return tasks


def worker_run(
    device: str,
    tasks: List[Dict[str, Any]],
    problems: List[ProblemRecord],
    model_name: str,
    gen_length: int,
    cfg_scale: float,
    block_eos: bool,
    torch_dtype,
) -> List[Dict[str, Any]]:
    if not tasks:
        return []
    torch.cuda.set_device(device)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(device).eval()
    results: List[Dict[str, Any]] = []
    for task in tasks:
        record = problems[task["problem_local_idx"]]
        prompt_ids = torch.tensor(record.prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        attn = torch.tensor(record.attention_mask, dtype=torch.long, device=device).unsqueeze(0)
        baseline_tokens = torch.tensor(record.baseline_tokens, dtype=torch.long, device=device)
        block_start = task["block_start"]
        block_positions = task["block_positions"]
        permutation = task["permutation"]
        ordered_positions = [block_positions[i] for i in permutation]

        x, full_attn, start = init_generation_state(prompt_ids, attn, gen_length)
        if block_start > 0:
            prefix_tokens = baseline_tokens[:block_start]
            x[0, start:start + block_start] = prefix_tokens

        block_log_probs = decode_block_in_order(
            model,
            x,
            full_attn,
            start,
            ordered_positions,
            cfg_scale=cfg_scale,
            block_eos=block_eos,
        )
        remaining = [
            rel for rel in range(gen_length)
            if int(x[0, start + rel].item()) == MASK_TOKEN_ID
        ]
        tail_log_probs = decode_confidence_positions(
            model,
            x,
            full_attn,
            start,
            remaining,
            cfg_scale=cfg_scale,
            block_eos=block_eos,
        )

        per_pos_log_probs = [None for _ in range(gen_length)]
        for i in range(block_start):
            per_pos_log_probs[i] = float(record.baseline_log_probs[i])
        for rel_pos, val in block_log_probs.items():
            per_pos_log_probs[rel_pos] = val
        for rel_pos, val in tail_log_probs.items():
            per_pos_log_probs[rel_pos] = val
        if any(lp is None for lp in per_pos_log_probs):
            missing = [i for i, lp in enumerate(per_pos_log_probs) if lp is None]
            raise RuntimeError(f"Missing log probs for positions {missing}")

        total_log_prob = float(sum(per_pos_log_probs))
        avg_log_prob = total_log_prob / gen_length
        perplexity = math.exp(-avg_log_prob)

        generated_tokens = x[0, start:start + gen_length].tolist()
        results.append({
            "problem_idx": record.idx,
            "problem_local_idx": task["problem_local_idx"],
            "block_index": task["block_index"],
            "block_number": task["block_number"],
            "block_start": block_start,
            "block_end": task["block_end"],
            "permutation_idx": task["permutation_idx"],
            "permutation_order": permutation,
            "permutation_positions": ordered_positions,
            "tokens": generated_tokens,
            "log_probs": per_pos_log_probs,
            "avg_log_prob": avg_log_prob,
            "perplexity": perplexity,
        })

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="Block-wise permutation sensitivity experiment.")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--num_problems", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--block_eos", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--device_ids", type=str, default="", help="Comma separated CUDA device indices.")
    parser.add_argument("--output_dir", type=str, default="results/block_permutation")
    parser.add_argument("--block_indices", type=str, default="", help="Comma separated 1-based block indices to probe.")
    parser.add_argument("--block_offset", type=int, default=0)
    parser.add_argument("--block_stride", type=int, default=0, help="Stride between blocks (defaults to block_size).")
    parser.add_argument("--max_target_blocks", type=int, default=0, help="Limit number of target blocks (0=all).")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    seed_everything(args.seed)
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    questions, truths, dataset_indices = prepare_dataset(args.num_problems)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    if tokenizer.pad_token_id == MASK_TOKEN_ID:
        raise ValueError("pad_token_id matches MASK_TOKEN_ID; please choose a tokenizer with different PAD.")

    prompts = build_chat_prompts(tokenizer, questions)
    encoded = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt")
    prompt_ids_all = encoded["input_ids"]
    attention_all = encoded["attention_mask"]
    prompt_len = prompt_ids_all.size(1)

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("CUDA devices are required for this experiment.")

    if args.device_ids:
        requested = [int(x.strip()) for x in args.device_ids.split(",") if x.strip()]
    else:
        requested = list(range(available_gpus))
    if not requested:
        raise ValueError("No CUDA devices specified or detected.")
    unique_requested = []
    for idx in requested:
        if idx < 0 or idx >= available_gpus:
            raise ValueError(f"Invalid device index {idx}; available range is [0, {available_gpus - 1}].")
        if idx not in unique_requested:
            unique_requested.append(idx)
    device_ids = unique_requested[: max(1, min(args.num_gpus, len(unique_requested)))]
    base_device = f"cuda:{device_ids[0]}"

    print(f"[INFO] Loading model on {base_device} for baseline generation...")
    baseline_model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(base_device).eval()

    problems: List[ProblemRecord] = []
    baseline_path = output_dir / "baseline.jsonl"
    with baseline_path.open("w", encoding="utf-8") as baseline_f:
        for local_idx, (dataset_idx, question, truth) in enumerate(zip(dataset_indices, questions, truths)):
            prompt_ids = prompt_ids_all[local_idx:local_idx + 1].to(base_device)
            attn = attention_all[local_idx:local_idx + 1].to(base_device)
            (
                tokens,
                log_probs,
                gen_text,
                pred,
                _,
                avg_log_prob,
                perplexity,
            ) = generate_baseline(
                baseline_model,
                tokenizer,
                prompt_ids,
                attn,
                args.gen_length,
                cfg_scale=args.cfg_scale,
                block_eos=args.block_eos,
            )
            correct = exact_number_match(pred, truth)
            record = ProblemRecord(
                idx=dataset_idx,
                question=question,
                truth=truth,
                prompt_ids=prompt_ids_all[local_idx].tolist(),
                attention_mask=attention_all[local_idx].tolist(),
                prompt_len=prompt_len,
                baseline_tokens=tokens,
                baseline_log_probs=log_probs,
                baseline_text=gen_text,
                baseline_pred=pred,
                baseline_correct=correct,
                baseline_avg_log_prob=avg_log_prob,
                baseline_perplexity=perplexity,
            )
            problems.append(record)
            baseline_line = {
                "problem_idx": dataset_idx,
                "question": question,
                "truth": truth,
                "prediction": pred,
                "correct": correct,
                "avg_log_prob": avg_log_prob,
                "perplexity": perplexity,
            }
            baseline_f.write(json.dumps(baseline_line, ensure_ascii=False) + "\n")
            print(f"[BASELINE] idx={dataset_idx} correct={correct} perplexity={perplexity:.3f}")

    del baseline_model
    torch.cuda.empty_cache()

    block_indices = []
    if args.block_indices:
        for tok in args.block_indices.split(","):
            tok = tok.strip()
            if not tok:
                continue
            val = int(tok)
            block_indices.append(val - 1)
    stride = args.block_stride if args.block_stride > 0 else args.block_size
    target_blocks = determine_target_blocks(
        args.gen_length,
        args.block_size,
        block_indices,
        args.block_offset,
        stride,
        args.max_target_blocks,
    )
    if not target_blocks:
        raise RuntimeError("No target blocks selected. Adjust block parameters.")

    tasks = build_permutation_tasks(problems, args.gen_length, args.block_size, target_blocks)
    print(f"[INFO] Prepared {len(tasks)} permutation tasks "
          f"({len(target_blocks)} blocks, block_size={args.block_size}).")
    if not tasks:
        print("[WARN] No tasks to execute; exiting.")
        return

    max_workers = min(len(device_ids), len(tasks))
    worker_devices = [f"cuda:{idx}" for idx in device_ids[:max_workers]]
    task_buckets = chunk_evenly(tasks, len(worker_devices))

    print(f"[INFO] Launching workers on devices: {worker_devices}")
    worker_results: List[Dict[str, Any]] = []
    if len(worker_devices) == 1:
        worker_results = worker_run(
            worker_devices[0],
            task_buckets[0],
            problems,
            args.model_name,
            args.gen_length,
            args.cfg_scale,
            args.block_eos,
            torch_dtype,
        )
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(worker_devices)) as pool:
            mp_args = [
                (
                    worker_devices[i],
                    task_buckets[i],
                    problems,
                    args.model_name,
                    args.gen_length,
                    args.cfg_scale,
                    args.block_eos,
                    torch_dtype,
                )
                for i in range(len(worker_devices))
            ]
            results = pool.starmap(worker_run, mp_args)
            for chunk in results:
                worker_results.extend(chunk)

    if not worker_results:
        print("[WARN] Workers returned no results.")
        return

    worker_results.sort(
        key=lambda item: (
            item["problem_idx"],
            item["block_index"],
            item["permutation_idx"],
        )
    )

    permutation_path = output_dir / "block_permutation_results.jsonl"
    block_summary: Dict[int, Dict[str, Any]] = {}

    with permutation_path.open("w", encoding="utf-8") as perm_f:
        for entry in worker_results:
            record = problems[entry["problem_local_idx"]]
            gen_text = tokenizer.decode(entry["tokens"], skip_special_tokens=True)
            pred = extract_after_answer(gen_text)
            correct = exact_number_match(pred, record.truth)
            line = dict(entry)
            line.update({
                "problem_idx": record.idx,
                "truth": record.truth,
                "prediction": pred,
                "correct": correct,
                "generated_text": gen_text,
            })
            perm_f.write(json.dumps(line, ensure_ascii=False) + "\n")

            summary = block_summary.setdefault(entry["block_index"], {
                "block_number": entry["block_number"],
                "count": 0,
                "correct": 0,
                "avg_log_prob_sum": 0.0,
                "perplexities": [],
            })
            summary["count"] += 1
            summary["correct"] += int(correct)
            summary["avg_log_prob_sum"] += entry["avg_log_prob"]
            summary["perplexities"].append(entry["perplexity"])

    summary_out = output_dir / "block_summary.json"
    block_summary_formatted = {}
    for block_idx, stats in sorted(block_summary.items()):
        count = max(1, stats["count"])
        block_summary_formatted[block_idx] = {
            "block_number": stats["block_number"],
            "num_samples": stats["count"],
            "accuracy": stats["correct"] / count,
            "avg_log_prob": stats["avg_log_prob_sum"] / count,
            "perplexity_mean": sum(stats["perplexities"]) / count,
            "perplexity_min": min(stats["perplexities"]),
            "perplexity_max": max(stats["perplexities"]),
        }
    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(block_summary_formatted, f, ensure_ascii=False, indent=2)

    config_out = output_dir / "config.json"
    config_dict = {
        "model_name": args.model_name,
        "gen_length": args.gen_length,
        "block_size": args.block_size,
        "num_problems": args.num_problems,
        "cfg_scale": args.cfg_scale,
        "block_eos": args.block_eos,
        "target_blocks": target_blocks,
        "device_ids": device_ids,
        "seed": args.seed,
        "dtype": args.dtype,
    }
    with config_out.open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Baseline saved to {baseline_path}")
    print(f"[DONE] Permutation results saved to {permutation_path}")
    print(f"[DONE] Block summary saved to {summary_out}")


if __name__ == "__main__":
    main()
