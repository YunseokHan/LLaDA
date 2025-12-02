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
F = None
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

def generate_halton_decoding(
    model,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_length: int,
    halton_steps: int,
    cfg_scale: float = 0.0,
    block_eos: bool = False,
    device: str = "cuda",
    temp_min: float = 1.0,
    temp_max: float = 1.0,
    temp_pow: float = 1.0,
    temp_warmup: int = 0,
    top_k: int = -1,
    randomize: bool = False,
    sched_pow: float = 1.0,
    nb_point: int = 10_000,
):
    if torch is None:
        raise RuntimeError("PyTorch is required to run decoding.")
    with torch.no_grad():
        B, Lp = prompt_ids.shape
        steps = max(1, halton_steps)
        x = torch.full((B, Lp + gen_length), MASK_TOKEN_ID, dtype=torch.long, device=device)
        x[:, :Lp] = prompt_ids
        attn = torch.cat(
            [attention_mask, torch.ones((B, gen_length), dtype=attention_mask.dtype, device=device)],
            dim=-1
        ) if attention_mask is not None else None

        base_order = torch.from_numpy(halton_order_for_length(gen_length, nb_point)).to(device=device)
        order = base_order.clone().unsqueeze(0).expand(B, -1).clone()
        if randomize:
            shifts = torch.randint(0, gen_length, (B,), device=device)
            for b in range(B):
                order[b] = torch.roll(order[b], int(shifts[b].item()))

        temperatures = torch.linspace(temp_min, temp_max, steps=steps, dtype=torch.float32, device=device)
        decode_orders: List[List[int]] = [[] for _ in range(B)]
        prev_target = 0
        start = Lp

        for step_idx in range(steps):
            if prev_target >= gen_length:
                break
            target = halton_schedule_target(step_idx, steps, gen_length, sched_pow=sched_pow)
            if step_idx == steps - 1:
                target = gen_length
            target = max(target, prev_target + 1)
            target = min(target, gen_length)
            rel_pos = order[:, prev_target:target]
            chunk = rel_pos.size(1)
            if chunk <= 0:
                continue

            logits = forward_logits(model, x, attn, cfg_scale=cfg_scale).float()
            abs_pos = rel_pos + start
            if block_eos:
                batch_idx = torch.arange(B, device=device).unsqueeze(1)
                logits[batch_idx, abs_pos, EOS_TOKEN_ID] = -torch.inf
                logits[batch_idx, abs_pos, EOT_TOKEN_ID] = -torch.inf

            batch_idx = torch.arange(B, device=device).unsqueeze(1)
            step_logits = logits[batch_idx, abs_pos, :]  # (B, chunk, V)
            temp_scale = float(temperatures[min(step_idx, temperatures.size(0) - 1)].item())
            temp_scale = max(temp_scale, 1e-5)
            temp_scale = temp_scale ** temp_pow
            if step_idx < temp_warmup:
                temp_scale *= 0.5
            scaled_logits = step_logits * temp_scale
            probs = torch.softmax(scaled_logits, dim=-1)

            if top_k > 0 and top_k < probs.size(-1):
                topk_prob, topk_idx = torch.topk(probs, k=top_k, dim=-1)
                topk_prob = topk_prob / torch.clamp(topk_prob.sum(dim=-1, keepdim=True), min=1e-9)
                sampled = torch.distributions.Categorical(topk_prob.view(-1, top_k)).sample()
                pred = topk_idx.view(-1, top_k).gather(1, sampled.unsqueeze(-1))
                pred = pred.view(B, chunk)
            else:
                flat_probs = probs.view(-1, probs.size(-1))
                sampled = torch.distributions.Categorical(flat_probs).sample()
                pred = sampled.view(B, chunk)

            x[batch_idx, abs_pos] = pred
            for b in range(B):
                decode_orders[b].extend([int(v) for v in rel_pos[b].tolist()])
            prev_target = target

        return x, decode_orders
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

def generate_margin_decoding(
    model,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_length: int,
    cfg_scale: float = 0.0,
    block_eos: bool = False,
    device: str = "cuda",
):
    """
    Decode tokens by selecting the mask position with the largest top-probability margin.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required to run decoding.")
    with torch.no_grad():
        B, Lp = prompt_ids.shape
        if B != 1:
            raise ValueError("margin decoding expects batch size 1 per call.")

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

            margins = []
            best_tokens = []
            for rel_pos in remaining:
                abs_pos = start + rel_pos
                pos_logits = logits[:, abs_pos, :]
                probs = torch.softmax(pos_logits, dim=-1)
                k = min(2, probs.size(-1))
                topk_probs, topk_idx = torch.topk(probs, k=k, dim=-1)
                if k == 1:
                    diff = topk_probs[:, 0]
                else:
                    diff = torch.abs(topk_probs[:, 0] - topk_probs[:, 1])
                margins.append(diff.squeeze(0))
                best_tokens.append(topk_idx[:, 0].squeeze(0))

            margin_tensor = torch.stack(margins)  # (num_remaining,)
            best_idx = int(torch.argmax(margin_tensor).item())
            chosen_rel = remaining.pop(best_idx)
            chosen_abs = start + chosen_rel
            chosen_token = best_tokens[best_idx]
            x[0, chosen_abs] = chosen_token
            decode_order.append(int(chosen_rel))

        return x, decode_order

def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        return logits
    eps = 1e-12
    noise = torch.rand_like(logits, dtype=torch.float64)
    noise = torch.clamp(noise, eps, 1.0 - eps)
    gumbel = -torch.log(-torch.log(noise))
    gumbel = gumbel.to(logits.dtype)
    return logits + gumbel * temperature

def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    if torch is None:
        raise RuntimeError("PyTorch is required to run decoding.")
    steps = max(1, int(steps))
    if mask_index.ndim != 2:
        mask_index = mask_index.view(mask_index.size(0), -1)
    mask_counts = mask_index.sum(dim=1, dtype=torch.int64)
    base = mask_counts // steps
    remainder = mask_counts % steps
    num_transfer = base.unsqueeze(1).repeat(1, steps)
    for b in range(mask_counts.size(0)):
        r = int(remainder[b].item())
        if r > 0:
            num_transfer[b, :r] += 1
    return num_transfer

def generate_conv_decoding(
    model,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_length: int,
    conv_steps: int,
    conv_block_length: int,
    conv_temperature: float = 0.0,
    cfg_scale: float = 0.0,
    block_eos: bool = False,
    device: str = "cuda",
    conv_remask: str = "low_confidence",
    conv_mult: float = 1.0,
):
    if torch is None:
        raise RuntimeError("PyTorch is required to run decoding.")
    if F is None:
        raise RuntimeError("torch.nn.functional is required for convolutional decoding.")
    with torch.no_grad():
        B, Lp = prompt_ids.shape
        steps = max(1, conv_steps)
        x = torch.full((B, Lp + gen_length), MASK_TOKEN_ID, dtype=torch.long, device=device)
        x[:, :Lp] = prompt_ids
        attn = torch.cat(
            [attention_mask, torch.ones((B, gen_length), dtype=attention_mask.dtype, device=device)],
            dim=-1
        ) if attention_mask is not None else None

        kernel_size = max(1, conv_block_length)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = torch.ones(1, 1, kernel_size, device=device, dtype=torch.float32)
        padding = (kernel_size - 1) // 2

        block_mask_index = (x[:, Lp:] == MASK_TOKEN_ID)
        transfer_counts = get_num_transfer_tokens(block_mask_index, steps)
        decode_orders: List[List[int]] = [[] for _ in range(B)]
        start = Lp

        for step_idx in range(steps):
            mask_index = (x == MASK_TOKEN_ID)
            if not bool(mask_index[:, start:].any().item()):
                break

            logits = forward_logits(model, x, attn, cfg_scale=cfg_scale).float()
            if block_eos:
                logits[:, start:, EOS_TOKEN_ID] = -torch.inf
                logits[:, start:, EOT_TOKEN_ID] = -torch.inf

            logits_noised = add_gumbel_noise(logits, conv_temperature)
            x0 = torch.argmax(logits_noised, dim=-1)

            if conv_remask == "random":
                x0_p = torch.rand((B, x.size(1)), device=device, dtype=torch.float32)
            else:
                probs = torch.softmax(logits, dim=-1)
                x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.full_like(x0_p, -torch.inf)
            confidence[mask_index] = x0_p[mask_index]
            confidence[:, :start] = -torch.inf

            unmasked = (~mask_index).float().unsqueeze(1)
            conv_scores = F.conv1d(unmasked, kernel, padding=padding).squeeze(1)
            conv_scores = torch.tanh(conv_scores * conv_mult)
            confidence[mask_index] = confidence[mask_index] * conv_scores[mask_index]

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for b in range(B):
                k = int(transfer_counts[b, step_idx].item())
                available = int(mask_index[b, start:].sum().item())
                k = min(k, available)
                if k <= 0:
                    continue
                vals, idx = torch.topk(confidence[b], k=k)
                transfer_index[b, idx] = True

            x[transfer_index] = x0[transfer_index]
            newly_filled = transfer_index[:, start:]
            nz = torch.nonzero(newly_filled, as_tuple=False)
            for row, rel in nz:
                decode_orders[int(row.item())].append(int(rel.item()))

        return x, decode_orders

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

_HALTON_ORDER_CACHE: Dict[Tuple[int, int], np.ndarray] = {}

def _halton_sequence(base: int, n_sample: int) -> np.ndarray:
    """
    Generate a 1D Halton sequence.
    """
    seq = np.zeros(n_sample, dtype=np.float64)
    for i in range(n_sample):
        f = 1.0
        r = 0.0
        idx = i + 1
        while idx > 0:
            f /= base
            r += f * (idx % base)
            idx //= base
        seq[i] = r
    return seq

def build_halton_order_1d(length: int, nb_point: int = 10_000) -> np.ndarray:
    """
    Build a Halton-based ordering over [0, length).
    """
    if length <= 0:
        return np.zeros(0, dtype=np.int64)
    nb_point = max(nb_point, length * 2)
    halton_vals = _halton_sequence(2, nb_point)
    order = np.floor(halton_vals * length).astype(np.int64)
    uniq_index = np.unique(order, return_index=True)[1]
    order = order[np.sort(uniq_index)]
    if order.size < length:
        # Append any missing indices in increasing order for determinism.
        missing = sorted(set(range(length)) - set(order.tolist()))
        order = np.concatenate([order, np.asarray(missing, dtype=np.int64)])
    return order[:length]

def halton_order_for_length(length: int, nb_point: int = 10_000) -> np.ndarray:
    key = (length, nb_point)
    if key not in _HALTON_ORDER_CACHE:
        _HALTON_ORDER_CACHE[key] = build_halton_order_1d(length, nb_point)
    return _HALTON_ORDER_CACHE[key]

def halton_schedule_target(step_idx: int, total_steps: int, total_positions: int, sched_pow: float = 1.0) -> int:
    """
    Determine how many positions should have been decoded after (step_idx + 1) steps.
    """
    if total_steps <= 1:
        return total_positions
    ratio = (step_idx + 1) / float(total_steps)
    ratio = max(min(ratio, 1.0), 1e-6)
    if sched_pow != 1.0:
        ratio = ratio ** sched_pow
    smooth = 1.0 - (math.acos(ratio) / (math.pi * 0.5))
    target = int(smooth * total_positions)
    target = max(step_idx + 1, target)
    return min(total_positions, target)

def format_method_label(
    method: str,
    semi_ar_block_size: int = None,
    halton_desc: str = None,
    conv_desc: str = None,
) -> str:
    if method == "semi_ar" and semi_ar_block_size is not None:
        return f"semi_ar ({semi_ar_block_size})"
    if method == "halton" and halton_desc:
        return f"halton ({halton_desc})"
    if method == "conv" and conv_desc:
        return f"conv ({conv_desc})"
    return method

def method_file_prefix(
    method: str,
    semi_ar_block_size: int = None,
    halton_tag: str = None,
    conv_tag: str = None,
) -> str:
    if method == "semi_ar" and semi_ar_block_size is not None:
        return f"semi_ar_bs{semi_ar_block_size}"
    if method == "halton" and halton_tag:
        return f"halton_{halton_tag}"
    if method == "conv" and conv_tag:
        return f"conv_{conv_tag}"
    return method

def halton_method_strings(args) -> Tuple[str | None, str | None]:
    if getattr(args, "method", None) != "halton":
        return None, None
    label_parts = [f"steps={args.halton_steps}"]
    if args.halton_randomize:
        label_parts.append("rand")
    if args.halton_top_k > 0:
        label_parts.append(f"topk={args.halton_top_k}")
    if args.halton_temp_min != 1.0 or args.halton_temp_max != 1.0:
        label_parts.append(f"T[{args.halton_temp_min:g},{args.halton_temp_max:g}]")
    if args.halton_temp_pow != 1.0:
        label_parts.append(f"pow={args.halton_temp_pow:g}")
    label = ", ".join(label_parts)

    tag_parts = [f"s{args.halton_steps}"]
    if args.halton_randomize:
        tag_parts.append("rand")
    if args.halton_top_k > 0:
        tag_parts.append(f"tk{args.halton_top_k}")
    if args.halton_temp_pow != 1.0:
        tag_parts.append(f"pow{args.halton_temp_pow:g}")
    tag = "_".join(tag_parts)
    return label or None, tag or None

def conv_method_strings(args) -> Tuple[str | None, str | None]:
    if getattr(args, "method", None) != "conv":
        return None, None
    desc_parts = []
    if getattr(args, "conv_block_length", None):
        desc_parts.append(f"blk={args.conv_block_length}")
    desc_parts.append(f"steps={args.conv_steps}")
    if args.conv_temperature > 0:
        desc_parts.append(f"T={args.conv_temperature:g}")
    if args.conv_remask and args.conv_remask != "low_confidence":
        desc_parts.append(args.conv_remask)
    if args.conv_mult != 1.0:
        desc_parts.append(f"m{args.conv_mult:g}")
    label = ", ".join(desc_parts)

    tag_parts = []
    if getattr(args, "conv_block_length", None):
        tag_parts.append(f"blk{args.conv_block_length}")
    tag_parts.append(f"s{args.conv_steps}")
    if args.conv_temperature > 0:
        tag_parts.append(f"t{args.conv_temperature:g}")
    if args.conv_remask and args.conv_remask != "low_confidence":
        tag_parts.append(args.conv_remask)
    if args.conv_mult != 1.0:
        tag_parts.append(f"m{args.conv_mult:g}")
    tag = "_".join(tag_parts) if tag_parts else None
    return label or None, tag or None

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
    global torch, F, load_dataset, AutoTokenizer, AutoModel
    import torch as _torch
    import torch.nn.functional as _F
    from datasets import load_dataset as _load_dataset
    from transformers import AutoTokenizer as _AutoTokenizer, AutoModel as _AutoModel

    torch = _torch
    F = _F
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
    halton_desc, halton_tag = halton_method_strings(args)
    conv_desc, conv_tag = conv_method_strings(args)
    method_label = format_method_label(args.method, args.semi_ar_block_size, halton_desc, conv_desc)

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
            "halton_steps": args.halton_steps,
            "halton_temp_min": args.halton_temp_min,
            "halton_temp_max": args.halton_temp_max,
            "halton_temp_pow": args.halton_temp_pow,
            "halton_temp_warmup": args.halton_temp_warmup,
            "halton_top_k": args.halton_top_k,
            "halton_randomize": bool(args.halton_randomize),
            "halton_sched_pow": args.halton_sched_pow,
            "halton_nb_points": args.halton_nb_points,
            "conv_steps": args.conv_steps,
            "conv_block_length": args.conv_block_length,
            "conv_temperature": args.conv_temperature,
            "conv_remask": args.conv_remask,
            "conv_mult": args.conv_mult,
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
        elif args.method == "margin":
            temperature = 0.7
        elif args.method == "conv":
            temperature = 0.0
        elif args.method == "halton":
            temperature = 0.0
        else:
            raise ValueError("--method must be 'ar', 'random', 'semi_ar', 'confidence', 'margin', 'conv', or 'halton'")

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
                elif args.method == "margin":
                    full_ids, conf_order = generate_margin_decoding(
                        model=model,
                        prompt_ids=p_ids,
                        attention_mask=p_attn,
                        gen_length=gen_length,
                        cfg_scale=args.cfg_scale,
                        block_eos=args.block_eos,
                        device=device,
                    )
                elif args.method == "conv":
                    full_ids, conv_orders = generate_conv_decoding(
                        model=model,
                        prompt_ids=p_ids,
                        attention_mask=p_attn,
                        gen_length=gen_length,
                        conv_steps=args.conv_steps,
                        conv_block_length=args.conv_block_length,
                        conv_temperature=args.conv_temperature,
                        cfg_scale=args.cfg_scale,
                        block_eos=args.block_eos,
                        device=device,
                        conv_remask=args.conv_remask,
                        conv_mult=args.conv_mult,
                    )
                    conf_order = conv_orders[0]
                elif args.method == "halton":
                    full_ids, halton_orders = generate_halton_decoding(
                        model=model,
                        prompt_ids=p_ids,
                        attention_mask=p_attn,
                        gen_length=gen_length,
                        halton_steps=args.halton_steps,
                        cfg_scale=args.cfg_scale,
                        block_eos=args.block_eos,
                        device=device,
                        temp_min=args.halton_temp_min,
                        temp_max=args.halton_temp_max,
                        temp_pow=args.halton_temp_pow,
                        temp_warmup=args.halton_temp_warmup,
                        top_k=args.halton_top_k,
                        randomize=args.halton_randomize,
                        sched_pow=args.halton_sched_pow,
                        nb_point=args.halton_nb_points,
                    )
                    conf_order = halton_orders[0]
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
                elif args.method == "margin":
                    row["permutation"] = " ".join(map(str, conf_order))
                elif args.method == "conv":
                    row["permutation"] = " ".join(map(str, conf_order))
                elif args.method == "halton":
                    row["permutation"] = " ".join(map(str, conf_order))
                else:
                    row["permutation"] = ""  # left-to-right implicit
                if (
                    args.save_success_perms
                    and corr == 1
                    and args.method in {"random", "semi_ar", "confidence", "margin", "conv", "halton"}
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
                    and args.method in {"random", "semi_ar", "confidence", "margin", "conv", "halton"}
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
    fname_method = method_file_prefix(args.method, args.semi_ar_block_size, halton_tag, conv_tag)
    tag = f"{fname_method}_sh{args.trial_shard:02d}_of{args.num_shards:02d}"
    out_csv = os.path.join(args.out_dir, f"trials_{tag}.csv")
    pd.DataFrame(result_rows).to_csv(out_csv, index=False)
    print(f"[DONE] wrote {len(result_rows)} rows to {out_csv}")
    wandb_log_artifact(run, out_csv, art_type="results", name=f"trials_{tag}")

    if args.method in {"random", "semi_ar", "confidence", "margin", "conv", "halton"} and args.save_success_perms:
        succ_path = os.path.join(args.out_dir, f"success_perms_{tag}.jsonl")
        with open(succ_path, "w", encoding="utf-8") as f:
            for line in success_lines:
                f.write(line + "\n")
        print(f"[DONE] wrote successful permutations to {succ_path}")
        wandb_log_artifact(run, succ_path, art_type="metadata", name=f"success_perms_{tag}")
    if args.method in {"random", "semi_ar", "confidence", "margin", "conv", "halton"} and args.save_failed_perms:
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
    pw.add_argument("--method", type=str, choices=["ar", "random", "semi_ar", "confidence", "margin", "conv", "halton"], required=True)
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
    pw.add_argument("--halton_steps", type=int, default=64,
                    help="Number of Halton scheduling steps (default: 64).")
    pw.add_argument("--halton_temp_min", type=float, default=1.0,
                    help="Minimum Halton softmax temperature multiplier (default: 1.0).")
    pw.add_argument("--halton_temp_max", type=float, default=1.0,
                    help="Maximum Halton softmax temperature multiplier (default: 1.0).")
    pw.add_argument("--halton_temp_pow", type=float, default=1.0,
                    help="Exponent applied to Halton temperature schedule (default: 1.0).")
    pw.add_argument("--halton_temp_warmup", type=int, default=0,
                    help="Number of initial Halton steps using half temperature (default: 0).")
    pw.add_argument("--halton_top_k", type=int, default=-1,
                    help="Top-k sampling inside Halton updates (-1 disables).")
    pw.add_argument("--halton_randomize", action="store_true",
                    help="Randomly roll the Halton ordering per example.")
    pw.add_argument("--halton_sched_pow", type=float, default=1.0,
                    help="Power applied to Halton schedule progress (default: 1.0).")
    pw.add_argument("--halton_nb_points", type=int, default=10_000,
                    help="Number of points used to build the Halton ordering (default: 10k).")
    pw.add_argument("--conv_steps", type=int, default=64,
                    help="Number of convolutional decoding steps (default: 64).")
    pw.add_argument("--conv_block_length", type=int, default=64,
                    help="Context window (kernel size) used for convolutional confidence (default: 64).")
    pw.add_argument("--conv_temperature", type=float, default=0.0,
                    help="Gumbel noise temperature for convolutional decoding (default: 0.0).")
    pw.add_argument("--conv_remask", type=str, choices=["low_confidence", "random"], default="low_confidence",
                    help="Confidence estimate used to select tokens per convolutional step.")
    pw.add_argument("--conv_mult", type=float, default=1.0,
                    help="Scaling multiplier applied to convolutional context weights (default: 1.0).")

    pw.add_argument("--trials", type=int, default=1024, help="Trials per problem.")
    pw.add_argument("--num_shards", type=int, default=1)
    pw.add_argument("--trial_shard", type=int, default=0)
    pw.add_argument("--out_dir", type=str, default="results/raw")
    pw.add_argument("--save_success_perms", action="store_true",
                    help="Save successful permutations (random/semi_ar/confidence/margin/conv/halton modes).")
    pw.add_argument("--save_failed_perms", action="store_true",
                    help="Save failed permutations (random/semi_ar/confidence/margin/conv/halton modes).")
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
