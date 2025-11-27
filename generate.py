import json
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
try:
    import matplotlib.font_manager as fm
    from matplotlib.ft2font import FT2Font
except ImportError:
    fm = None
    FT2Font = None

from transformers import AutoTokenizer, AutoModel

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def _ensure_tokenizer(model, tokenizer):
    if tokenizer is not None:
        return tokenizer

    inferred_tokenizer = getattr(model, 'tokenizer', None)
    if inferred_tokenizer is not None:
        return inferred_tokenizer

    config = getattr(model, 'config', None)
    name_or_path = getattr(model, 'name_or_path', None)
    if name_or_path is None and config is not None:
        name_or_path = getattr(config, '_name_or_path', None)

    if name_or_path is None:
        raise ValueError('Failed to infer tokenizer; please pass one explicitly.')

    trust_remote = getattr(config, 'trust_remote_code', False) if config is not None else False
    try:
        return AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=trust_remote)
    except Exception as exc:
        raise ValueError('Failed to infer tokenizer; please pass one explicitly.') from exc


def _export_topk_visualization(logged_steps, logged_topk, logged_states, prompt_length,
                               gen_length, mask_id, tokenizer, figs_dir,
                               fig_name, json_name, display_rows=10):
    if not logged_steps:
        raise RuntimeError('No steps were logged; please check log_interval and steps configuration.')

    figs_dir.mkdir(parents=True, exist_ok=True)

    if json_name is not None:
        json_path = figs_dir / json_name
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(logged_topk, f, ensure_ascii=False, indent=2)

    if fig_name is None:
        return

    generation_indices = np.arange(gen_length)
    width = gen_length * 0.18
    height_per_step = 10.0
    height_ratios = []
    for _ in logged_steps:
        height_ratios.extend([5, 1.5])
    fig = plt.figure(figsize=(width, height_per_step * len(logged_steps)))
    gs = fig.add_gridspec(
        nrows=2 * len(logged_steps),
        ncols=1,
        height_ratios=height_ratios,
        hspace=0.5
    )

    font_size = 6
    table_font_size = 3
    step_results = []
    strings_for_font = set()

    for step_num in logged_steps:
        step_key = str(step_num)
        step_entries = logged_topk.get(step_key, [])
        state_snapshot = logged_states[step_num][0]

        pos_entry_map = {}
        for entry in step_entries:
            rel_pos = entry['position'] - prompt_length
            if 0 <= rel_pos < gen_length:
                pos_entry_map[rel_pos] = entry['top_tokens']

        max_probs = np.zeros(gen_length, dtype=np.float64)
        table_data = [['-' for _ in range(gen_length)] for _ in range(display_rows)]

        def token_to_text(tid):
            text = tokenizer.convert_ids_to_tokens(tid)
            return text

        for rel_pos in range(gen_length):
            abs_pos = prompt_length + rel_pos
            token_id = int(state_snapshot[abs_pos].item())
            if token_id != mask_id:
                token_str = token_to_text(token_id)
                table_data[0][rel_pos] = token_str
                for row in range(1, display_rows):
                    table_data[row][rel_pos] = '-'
                max_probs[rel_pos] = np.nan
                if token_str and token_str != '-':
                    strings_for_font.add(token_str)
                continue

            entry_tokens = pos_entry_map.get(rel_pos)
            if entry_tokens:
                logits_vals = torch.tensor([tok['logit'] for tok in entry_tokens], dtype=torch.float64)
                probs = torch.softmax(logits_vals, dim=0).cpu().numpy()
                max_probs[rel_pos] = float(probs[0])
                for row in range(display_rows):
                    if row < len(entry_tokens):
                        tid = int(entry_tokens[row]['token_id'])
                        token_str = token_to_text(tid)
                        table_data[row][rel_pos] = token_str
                        if token_str and token_str != '-':
                            strings_for_font.add(token_str)
                    else:
                        table_data[row][rel_pos] = '-'
            else:
                max_probs[rel_pos] = 0.0
                for row in range(display_rows):
                    table_data[row][rel_pos] = '-'

        step_results.append((step_num, max_probs, table_data))

    font_prop = _resolve_font_properties(strings_for_font)

    for idx, (step_num, max_probs, table_data) in enumerate(step_results):
        ax_plot = fig.add_subplot(gs[2 * idx])
        mask = ~np.isnan(max_probs)
        if np.any(mask):
            valid_indices = np.where(mask)[0]
            breaks = np.where(np.diff(valid_indices) > 1)[0]
            segments = np.split(valid_indices, breaks + 1)
            for segment in segments:
                if segment.size == 0:
                    continue
                ax_plot.plot(
                    generation_indices[segment],
                    max_probs[segment],
                    marker='o',
                    markersize=2,
                    linewidth=1.0
                )
        ax_plot.set_ylim(0.0, 1.05)
        ax_plot.set_xlim(-0.5, gen_length - 0.5)
        tick_step = max(1, gen_length // 16)
        ax_plot.set_xticks(np.arange(0, gen_length, tick_step))
        ax_plot.set_ylabel('Max Prob')
        if font_prop is not None:
            ax_plot.set_title(f'Step {step_num}', fontproperties=font_prop)
        else:
            ax_plot.set_title(f'Step {step_num}')
        if idx == len(logged_steps) - 1:
            ax_plot.set_xlabel('Generation Position')
        else:
            ax_plot.set_xlabel('')
            ax_plot.set_xticklabels([])

        ax_table = fig.add_subplot(gs[2 * idx + 1])
        ax_table.set_xlim(-0.5, gen_length - 0.5)
        ax_table.set_ylim(display_rows - 0.5, -0.5)
        ax_table.set_xticks(np.arange(0, gen_length, tick_step))
        ax_table.set_xticklabels(np.arange(0, gen_length, tick_step))
        ax_table.set_yticks(np.arange(display_rows))
        ax_table.set_yticklabels([f'Top-{row + 1}' for row in range(display_rows)])
        ax_table.tick_params(axis='x', labelrotation=90, labelsize=font_size)
        ax_table.tick_params(axis='y', labelsize=font_size)
        for spine in ax_table.spines.values():
            spine.set_visible(False)

        for col in range(gen_length + 1):
            ax_table.axvline(col - 0.5, color='lightgray', linewidth=0.4)
        for row in range(display_rows + 1):
            ax_table.axhline(row - 0.5, color='lightgray', linewidth=0.4)

        for row in range(display_rows):
            for col in range(gen_length):
                text = table_data[row][col]
                text_kwargs = {'ha': 'center', 'va': 'center', 'fontsize': table_font_size}
                if font_prop is not None:
                    text_kwargs['fontproperties'] = font_prop
                ax_table.text(col, row, text, **text_kwargs)

        if font_prop is not None:
            for label in ax_table.get_xticklabels():
                label.set_fontproperties(font_prop)
            for label in ax_table.get_yticklabels():
                label.set_fontproperties(font_prop)

    fig.tight_layout()
    fig_path = figs_dir / fig_name
    fig.savefig(fig_path, dpi=1000)
    plt.close(fig)


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, intervened_output=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336. <|mdm_mask|>
        intervened_output: Optional tensor of shape (B, gen_length) used to pre-fill the generation segment.
    '''
    batch_size = prompt.shape[0]
    if intervened_output is not None:
        if intervened_output.shape != (batch_size, gen_length):
            raise ValueError(f'intervened_output must have shape {(batch_size, gen_length)}, '
                             f'got {tuple(intervened_output.shape)}.')
        gen_segment = intervened_output.to(model.device, non_blocking=True).long().clone()
    else:
        gen_segment = torch.full((batch_size, gen_length), mask_id, dtype=torch.long, device=model.device)
    x = torch.cat([prompt.clone().to(model.device), gen_segment], dim=1)

    prompt_index = (x != mask_id)

    total_steps = steps
    num_blocks, steps_per_block, block_offsets, tokens_per_step = _prepare_block_schedule(
        gen_segment, mask_id, gen_length, block_length, total_steps
    )

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = block_start + block_length
        for i in range(block_offsets[num_block], steps_per_block):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            tokens_this_step = tokens_per_step[:, i]
            for j in range(confidence.shape[0]):
                k = int(tokens_this_step[j].item())
                if k <= 0:
                    continue
                block_mask = mask_index[j, block_start:block_end]
                available_in_block = int(block_mask.sum().item())
                if available_in_block == 0:
                    continue
                k = min(k, available_in_block)
                block_confidence = confidence[j, block_start:block_end]
                _, select_relative = torch.topk(block_confidence, k=k)
                select_index = select_relative + block_start
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


@ torch.no_grad()
def generate_w_fig(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                   cfg_scale=0., remasking='low_confidence', mask_id=126336,
                   fig_name='generation_order.pdf', order_json_name='generation_order.json',
                   likelihood_json_name='step_log_likelihood.json', tokenizer=None,
                   intervened_output=None, include_confidence_fig=True,
                   conf_log_interval=20, conf_top_k=10,
                   conf_json_name='step_top_logits.json', conf_fig_name='step_top_logits.pdf'):
    '''
    Combined diagnostic visualisation for the iterative sampling process.

    The figure overlays the generation-order heatmap (left y-axis) with the
    cumulative log-likelihood trace (right y-axis). Optional JSON exports mirror
    the legacy helpers: ``order_json_name`` stores decoded tokens per step
    (requires a tokenizer) and ``likelihood_json_name`` stores per-step
    likelihood statistics. When ``include_confidence_fig`` is enabled, the
    routine reuses the logits computed during sampling to export the
    top-k confidence table/plot and JSON previously produced by
    :func:`generate_w_conf`.
    '''
    batch_size = prompt.shape[0]
    if batch_size != 1:
        raise ValueError('generate_w_fig currently supports batch size 1 for visualization.')

    device = model.device

    capture_conf = (
        include_confidence_fig
        and conf_log_interval is not None
        and conf_top_k is not None
        and ((conf_json_name is not None) or (conf_fig_name is not None))
    )
    if capture_conf:
        if conf_log_interval < 1:
            raise ValueError('conf_log_interval must be a positive integer.')
        if conf_top_k < 1:
            raise ValueError('conf_top_k must be a positive integer.')
        if conf_top_k < 10:
            raise ValueError('conf_top_k must be at least 10 to build the confidence table.')

    if intervened_output is not None:
        if intervened_output.shape != (batch_size, gen_length):
            raise ValueError(f'intervened_output must have shape {(batch_size, gen_length)}, '
                             f'got {tuple(intervened_output.shape)}.')
        gen_segment = intervened_output.to(device, non_blocking=True).long().clone()
    else:
        gen_segment = torch.full((batch_size, gen_length), mask_id, dtype=torch.long, device=device)

    prompt_length = prompt.shape[1]
    total_steps = steps
    num_blocks, steps_per_block, block_offsets, tokens_per_step = _prepare_block_schedule(
        gen_segment, mask_id, gen_length, block_length, total_steps
    )

    collect_generation_log = order_json_name is not None
    tokenizer_required = collect_generation_log or capture_conf
    if tokenizer_required:
        tokenizer = _ensure_tokenizer(model, tokenizer)

    cumulative_log_likelihood = torch.zeros(batch_size, dtype=torch.float64, device=device)
    step_axes = [[] for _ in range(batch_size)]
    loglik_values = [[] for _ in range(batch_size)]
    step_history = []
    prefill_history = []
    order_grid = torch.full((gen_length, total_steps), float('nan'))
    generation_log = [] if collect_generation_log else None
    if capture_conf:
        conf_logged_topk = {}
        conf_logged_states = {}
        conf_logged_steps = []

    block_prefill_positions = []
    block_prefill_tokens = []
    block_prefill_cursors = []
    for block_idx in range(num_blocks):
        block_slice = slice(block_idx * block_length, (block_idx + 1) * block_length)
        block_segment = gen_segment[:, block_slice]
        positions_per_sample = []
        tokens_per_sample = []
        cursors_per_sample = []
        for sample_idx in range(batch_size):
            non_mask = torch.nonzero(block_segment[sample_idx] != mask_id, as_tuple=False).squeeze(-1)
            if non_mask.numel() > 0:
                global_positions = (non_mask + prompt_length + block_idx * block_length).tolist()
                positions_per_sample.append(global_positions)
                tokens_per_sample.append(block_segment[sample_idx, non_mask].tolist())
            else:
                positions_per_sample.append([])
                tokens_per_sample.append([])
            cursors_per_sample.append(0)
        block_prefill_positions.append(positions_per_sample)
        block_prefill_tokens.append(tokens_per_sample)
        block_prefill_cursors.append(cursors_per_sample)

    x = torch.cat(
        [prompt.clone().to(device), torch.full((batch_size, gen_length), mask_id, dtype=torch.long, device=device)],
        dim=1
    )

    prefill_steps = 0
    if any(len(pos_list[sample_idx]) > 0
           for pos_list in block_prefill_positions
           for sample_idx in range(batch_size)):
        for num_block in range(num_blocks):
            block_start = prompt_length + num_block * block_length
            block_end = block_start + block_length
            for i in range(block_offsets[num_block]):
                if cfg_scale > 0.:
                    prompt_index_prefill = (x != mask_id)
                    un_x = x.clone()
                    un_x[prompt_index_prefill] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits

                p = F.softmax(logits, dim=-1)
                tokens_this_step = tokens_per_step[:, i]
                delta_log_likelihood = torch.zeros(batch_size, dtype=torch.float64, device=device)
                step_positions = [[] for _ in range(batch_size)]
                step_tokens = [[] for _ in range(batch_size)]
                step_log_probs = [[] for _ in range(batch_size)]

                for sample_idx in range(batch_size):
                    k = int(tokens_this_step[sample_idx].item())
                    if k <= 0:
                        continue
                    positions_queue = block_prefill_positions[num_block][sample_idx]
                    tokens_queue = block_prefill_tokens[num_block][sample_idx]
                    cursor = block_prefill_cursors[num_block][sample_idx]
                    available = len(positions_queue) - cursor
                    if k > available:
                        raise ValueError('Intervened output is inconsistent with scheduling metadata.')
                    selected_positions = positions_queue[cursor:cursor + k]
                    selected_tokens = tokens_queue[cursor:cursor + k]
                    block_prefill_cursors[num_block][sample_idx] += k

                    pos_tensor = torch.tensor(selected_positions, dtype=torch.long, device=device)
                    tok_tensor = torch.tensor(selected_tokens, dtype=torch.long, device=device)
                    prob_tensor = torch.gather(
                        p[sample_idx, pos_tensor],
                        dim=-1,
                        index=tok_tensor.unsqueeze(-1)
                    ).squeeze(-1).clamp_min(1e-12)
                    log_probs = prob_tensor.log().to(torch.float64)
                    delta_log_likelihood[sample_idx] += log_probs.sum()

                    x[sample_idx, pos_tensor] = tok_tensor
                    step_positions[sample_idx] = [int(pos - prompt_length) for pos in selected_positions]
                    step_tokens[sample_idx] = [int(token) for token in selected_tokens]
                    step_log_probs[sample_idx] = log_probs.detach().cpu().tolist()

                cumulative_log_likelihood += delta_log_likelihood
                global_step = num_block * steps_per_block + i
                prefill_history.append({
                    'step': int(global_step + 1),
                    'delta_log_likelihood': delta_log_likelihood.detach().cpu().tolist(),
                    'cumulative_log_likelihood': cumulative_log_likelihood.detach().cpu().tolist(),
                    'positions': step_positions,
                    'tokens': step_tokens,
                    'token_log_probs': step_log_probs
                })
                prefill_steps = global_step + 1

    for block_idx in range(num_blocks):
        for sample_idx in range(batch_size):
            cursor = block_prefill_cursors[block_idx][sample_idx]
            total = len(block_prefill_positions[block_idx][sample_idx])
            if cursor != total:
                raise ValueError('Failed to reconcile intervened_output with scheduling constraints.')

    x[:, prompt_length:] = torch.where(gen_segment != mask_id, gen_segment, x[:, prompt_length:])
    prompt_index = (x != mask_id)
    prefill_cumulative_snapshot = cumulative_log_likelihood.detach().cpu().tolist()

    if prefill_steps > 0 or torch.any(cumulative_log_likelihood != 0):
        for sample_idx in range(batch_size):
            step_axes[sample_idx].append(prefill_steps)
            loglik_values[sample_idx].append(float(cumulative_log_likelihood[sample_idx].item()))

    for num_block in range(num_blocks):
        block_start = prompt_length + num_block * block_length
        block_end = block_start + block_length
        for i in range(block_offsets[num_block], steps_per_block):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            current_step = num_block * steps_per_block + i
            if capture_conf and current_step % conf_log_interval == 0:
                step_key = current_step + 1
                if step_key not in conf_logged_topk:
                    mask_positions = torch.nonzero(mask_index, as_tuple=False)
                    if mask_positions.numel() > 0:
                        step_entries = []
                        vocab_top = min(conf_top_k, logits.size(-1))
                        for pos in mask_positions:
                            batch_idx = int(pos[0].item())
                            token_idx = int(pos[1].item())
                            top_values, top_indices = torch.topk(logits[batch_idx, token_idx], k=vocab_top)
                            top_tokens = [
                                {
                                    'token_id': int(token_id),
                                    'logit': float(logit_val)
                                }
                                for token_id, logit_val in zip(
                                    top_indices.detach().cpu().tolist(),
                                    top_values.detach().to(torch.float32).cpu().tolist()
                                )
                            ]
                            step_entries.append({
                                'batch': batch_idx,
                                'position': token_idx,
                                'top_tokens': top_tokens
                            })
                        conf_logged_topk[str(step_key)] = step_entries
                    else:
                        conf_logged_topk[str(step_key)] = []
                    conf_logged_steps.append(step_key)
                    conf_logged_states[step_key] = x.detach().clone().to('cpu')

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            p = F.softmax(logits, dim=-1)

            if remasking == 'low_confidence':
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking == 'high_confidence':
                x0_p = -torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            tokens_this_step = tokens_per_step[:, i]
            delta_log_likelihood = torch.zeros(batch_size, dtype=torch.float64, device=device)
            step_positions = [[] for _ in range(batch_size)]
            step_tokens = [[] for _ in range(batch_size)]
            step_log_probs = [[] for _ in range(batch_size)]
            selected_positions_batch = [None] * batch_size

            for j in range(confidence.shape[0]):
                k = int(tokens_this_step[j].item())
                if k <= 0:
                    continue
                block_mask = mask_index[j, block_start:block_end]
                available_in_block = int(block_mask.sum().item())
                if available_in_block == 0:
                    continue
                k = min(k, available_in_block)
                block_confidence = confidence[j, block_start:block_end]
                _, select_relative = torch.topk(block_confidence, k=k)
                select_index = select_relative + block_start
                transfer_index[j, select_index] = True
                selected_positions_batch[j] = select_index

                selected_tokens = x0[j, select_index]
                prob_tensor = torch.gather(
                    p[j, select_index],
                    dim=-1,
                    index=selected_tokens.unsqueeze(-1)
                ).squeeze(-1).clamp_min(1e-12)
                log_probs = prob_tensor.log().to(torch.float64)
                delta_log_likelihood[j] += log_probs.sum()

                rel_positions = (select_index - prompt_length).tolist()
                step_positions[j] = [int(pos) for pos in rel_positions]
                step_tokens[j] = [int(token) for token in selected_tokens.tolist()]
                step_log_probs[j] = log_probs.detach().cpu().tolist()

            x[transfer_index] = x0[transfer_index]

            cumulative_log_likelihood += delta_log_likelihood
            global_step = num_block * steps_per_block + i
            step_entry = {
                'step': int(global_step + 1),
                'delta_log_likelihood': delta_log_likelihood.detach().cpu().tolist(),
                'cumulative_log_likelihood': cumulative_log_likelihood.detach().cpu().tolist(),
                'positions': step_positions,
                'tokens': step_tokens,
                'token_log_probs': step_log_probs
            }
            step_history.append(step_entry)

            generated_indices = torch.nonzero(transfer_index[0], as_tuple=False).squeeze(-1)
            if generated_indices.numel() > 0:
                generated_indices = generated_indices[generated_indices >= prompt_length]
            if generated_indices.numel() > 0:
                rel_indices = (generated_indices - prompt_length).long().cpu()
                valid = rel_indices < gen_length
                rel_indices = rel_indices[valid]
                order_grid[rel_indices, global_step] = rel_indices.to(order_grid.dtype) + 1

            if collect_generation_log:
                selected_positions = selected_positions_batch[0]
                if selected_positions is not None and selected_positions.numel() > 0:
                    selected_positions = selected_positions[selected_positions >= prompt_length]
                    if selected_positions.numel() > 0:
                        rel_pos = (selected_positions - prompt_length).long()
                        valid = rel_pos < gen_length
                        if valid.any():
                            selected_positions = selected_positions[valid]
                            rel_pos = rel_pos[valid]
                            order_idx = torch.argsort(rel_pos)
                            rel_pos = rel_pos[order_idx]
                            selected_positions = selected_positions[order_idx]
                            token_ids = x[0, selected_positions].detach().cpu().tolist()
                            rel_pos_list = rel_pos.detach().cpu().tolist()
                            generation_log.append((global_step, rel_pos_list, token_ids))

            step_number = float(global_step + 1)
            for sample_idx in range(batch_size):
                if step_positions[sample_idx]:
                    step_axes[sample_idx].append(step_number)
                    loglik_values[sample_idx].append(float(cumulative_log_likelihood[sample_idx].item()))

    figs_dir = Path(__file__).resolve().parent / 'figs'
    figs_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figs_dir / fig_name

    order_grid_np = order_grid.numpy()
    masked_grid = np.ma.masked_invalid(order_grid_np)
    cmap = plt.colormaps.get_cmap('viridis').copy()
    cmap.set_bad(color='white')
    font_props = _resolve_font_properties(None)

    fig = plt.figure(figsize=(9, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[24, 1], wspace=0.4)
    ax = fig.add_subplot(gs[0, 0])
    extent = [0.5, total_steps + 0.5, -0.5, gen_length - 0.5]
    im = ax.imshow(masked_grid, aspect='auto', origin='lower', cmap=cmap,
                   norm=mcolors.Normalize(vmin=1, vmax=gen_length, clip=False), extent=extent)

    if intervened_output is not None:
        prefilled = torch.nonzero(gen_segment[0] != mask_id, as_tuple=False).squeeze(-1)
        if prefilled.numel() > 0:
            prefilled_np = prefilled.detach().cpu().numpy()
            splits = np.where(np.diff(prefilled_np) != 1)[0] + 1
            segments = np.split(prefilled_np, splits)
            for segment in segments:
                if segment.size == 0:
                    continue
                start = float(segment[0])
                end = float(segment[-1])
                ax.axhspan(start - 0.5, end + 0.5, facecolor='red', alpha=0.15, linewidth=0, zorder=1.5)

    if prefill_steps > 0:
        ax.axvline(prefill_steps, color='gray', linestyle='--', linewidth=1)

    ax.set_xlabel('Step', fontproperties=font_props)
    ax.set_ylabel('Token index', fontproperties=font_props)
    ax.set_xlim(0.5, total_steps + 0.5)
    ax.set_ylim(-0.5, gen_length - 0.5)
    ax.set_title('Generation Order & Log-Likelihood', fontproperties=font_props)
    cax = fig.add_subplot(gs[0, 1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Token position (1-based)', fontproperties=font_props)

    ax2 = ax.twinx()
    plotted_any = False
    for sample_idx in range(batch_size):
        if not step_axes[sample_idx]:
            continue
        steps_arr = np.array(step_axes[sample_idx], dtype=np.float64)
        loglik_arr = np.array(loglik_values[sample_idx], dtype=np.float64)
        ax2.plot(
            steps_arr,
            loglik_arr,
            marker='o',
            markersize=3,
            linestyle='-',
            linewidth=1.0,
            color='tab:orange',
            label='Log-Likelihood'
        )
        plotted_any = True
    ax2.set_xlim(ax.get_xlim())
    ax2.set_ylabel('Cumulative Log-Likelihood', fontproperties=font_props)
    if plotted_any:
        ax2.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    if likelihood_json_name is not None:
        likelihood_json_path = figs_dir / likelihood_json_name
        json_dump = {
            'prefill_steps': int(prefill_steps),
            'prefill_cumulative_log_likelihood': prefill_cumulative_snapshot if prefill_steps > 0
            else [0.0] * batch_size,
            'prefill_history': prefill_history,
            'step_history': step_history
        }
        with open(likelihood_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_dump, f, ensure_ascii=False, indent=2)

    if collect_generation_log:
        order_json_path = figs_dir / order_json_name
        step_token_map = {}
        for step, _rel_positions, token_ids in (generation_log or []):
            key = str(step + 1)
            text = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            step_token_map.setdefault(key, []).append(text)
        with open(order_json_path, 'w', encoding='utf-8') as f:
            json.dump(step_token_map, f, ensure_ascii=False, indent=2)

    if capture_conf:
        _export_topk_visualization(
            conf_logged_steps,
            conf_logged_topk,
            conf_logged_states,
            prompt_length,
            gen_length,
            mask_id,
            tokenizer,
            figs_dir,
            conf_fig_name,
            conf_json_name
        )

    return x


@ torch.no_grad()
def generate_w_order(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                     cfg_scale=0., remasking='low_confidence', mask_id=126336,
                     fig_name='generation_order.jpg', json_name='generation_order.json',
                     tokenizer=None, intervened_output=None):
    '''Convenience wrapper that preserves the original generate_w_order signature.'''
    return generate_w_fig(model, prompt, steps=steps, gen_length=gen_length,
                          block_length=block_length, temperature=temperature,
                          cfg_scale=cfg_scale, remasking=remasking, mask_id=mask_id,
                          fig_name=fig_name, order_json_name=json_name,
                          likelihood_json_name=None, tokenizer=tokenizer,
                          intervened_output=intervened_output)


@ torch.no_grad()
def generate_w_likelihood(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                          cfg_scale=0., remasking='low_confidence', mask_id=126336,
                          fig_name='step_log_likelihood.pdf', json_name='step_log_likelihood.json',
                          intervened_output=None):
    '''Convenience wrapper matching the legacy generate_w_likelihood signature.'''
    return generate_w_fig(model, prompt, steps=steps, gen_length=gen_length,
                          block_length=block_length, temperature=temperature,
                          cfg_scale=cfg_scale, remasking=remasking, mask_id=mask_id,
                          fig_name=fig_name, order_json_name=None,
                          likelihood_json_name=json_name, tokenizer=None,
                          intervened_output=intervened_output)



def _font_supports_chars(font_path, chars):
    if FT2Font is None:
        return False
    try:
        font = FT2Font(font_path)
    except Exception:
        return False
    for ch in chars:
        if font.get_char_index(ord(ch)) == 0:
            return False
    return True


def _resolve_font_properties(strings):
    if not strings or fm is None or FT2Font is None:
        return None
    chars = set()
    for text in strings:
        if not text:
            continue
        for ch in text:
            if ch.strip():
                chars.add(ch)
    if not chars:
        return None

    candidate_families = [
        'Noto Sans CJK JP',
        'Noto Sans CJK SC',
        'Noto Sans CJK KR',
        'Noto Sans JP',
        'Noto Sans SC',
        'Noto Sans KR',
        'Noto Sans TC',
        'Noto Sans',
        'SimHei',
        'WenQuanYi Zen Hei',
        'Arial Unicode MS'
    ]
    for family in candidate_families:
        try:
            font_path = fm.findfont(family, fallback_to_default=False)
        except Exception:
            continue
        if _font_supports_chars(font_path, chars):
            return fm.FontProperties(fname=font_path)

    for font in fm.fontManager.ttflist if fm is not None else []:
        if _font_supports_chars(font.fname, chars):
            return fm.FontProperties(fname=font.fname)

    return None


def build_intervened_output(tokenizer, gen_length, mask_id, interventions, *, device=None, batch_size=1):
    '''
    Creates a generation segment initialized with [MASK] tokens and applies user interventions.

    Args:
        tokenizer: Tokenizer used to convert textual interventions into token ids.
        gen_length: Target generation length.
        mask_id: Token id that represents the mask token.
        interventions: Either a dict or a list of dicts, each containing:
            - 'text': The substring to insert.
            - 'position': Zero-based index relative to the generation segment where the substring starts.
            - Optional 'batch_index': The batch index to modify (default: 0).
        device: Optional torch device for the returned tensor.
        batch_size: Number of sequences to prepare.

    Returns:
        Tensor of shape (batch_size, gen_length) containing the intervened output segment.
    '''
    if tokenizer is None:
        raise ValueError('tokenizer must be provided to build_intervened_output.')
    if device is None:
        device = torch.device('cpu')
    else:
        device = torch.device(device)

    if interventions is None:
        interventions = []
    elif isinstance(interventions, dict):
        interventions = [interventions]

    output = torch.full((batch_size, gen_length), mask_id, dtype=torch.long, device=device)

    for item in interventions:
        if not isinstance(item, dict):
            raise TypeError('Each intervention must be a dict with keys "text" and "position".')

        if 'position' not in item:
            raise ValueError('Intervention dict must include a "position" key.')
        if 'text' not in item:
            raise ValueError('Intervention dict must include a "text" key.')

        position = int(item['position'])
        if position < 0:
            raise ValueError(f'Intervention position must be non-negative, got {position}.')

        batch_index = int(item.get('batch_index', 0))
        if not (0 <= batch_index < batch_size):
            raise ValueError(f'Intervention batch_index {batch_index} is out of range for batch_size {batch_size}.')

        text = item['text']
        tokenized = tokenizer(text, add_special_tokens=False)
        token_ids = tokenized.get('input_ids', [])
        if len(token_ids) == 0:
            continue

        if position >= gen_length:
            continue

        end_position = min(gen_length, position + len(token_ids))
        slice_length = end_position - position
        if slice_length <= 0:
            continue

        replacement = torch.tensor(token_ids[:slice_length], dtype=torch.long, device=device)
        output[batch_index, position:end_position] = replacement

    return output


def _prepare_block_schedule(gen_segment, mask_id, gen_length, block_length, steps):
    '''
    Computes per-block scheduling metadata for linear mask scheduling with possible interventions.

    Returns:
        num_blocks: Number of generation blocks.
        steps_per_block: Steps allocated to each block.
        block_offsets: List of starting step indices per block accounting for pre-filled tokens.
        base_transfer: Tensor (batch_size, steps_per_block) with token counts per step under full masking.
    '''
    if gen_length % block_length != 0:
        raise ValueError('gen_length must be divisible by block_length.')

    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError('steps must be divisible by gen_length // block_length under linear schedule.')
    steps_per_block = steps // num_blocks

    batch_size = gen_segment.size(0)
    device = gen_segment.device

    if steps_per_block == 0:
        return num_blocks, steps_per_block, [0] * num_blocks, torch.zeros((batch_size, 0), dtype=torch.int64, device=device)

    base_mask = torch.ones((batch_size, block_length), dtype=torch.bool, device=device)
    base_transfer = get_num_transfer_tokens(base_mask, steps_per_block) # shape: (batch_size, steps_per_block)
    cumsum_transfer = torch.cumsum(base_transfer, dim=1)

    block_offsets = []
    for block_idx in range(num_blocks):
        block_slice = slice(block_idx * block_length, (block_idx + 1) * block_length)
        block_segment = gen_segment[:, block_slice]
        block_size = block_segment.size(1)
        mask_remaining = (block_segment == mask_id).sum(dim=1)
        tokens_unmasked = block_size - mask_remaining

        offsets = []
        for sample_idx in range(batch_size):
            tokens = int(tokens_unmasked[sample_idx].item())
            tokens = max(0, min(tokens, block_size))
            if tokens == 0:
                offset = 0
            elif tokens >= block_size:
                offset = steps_per_block
            else:
                cs = cumsum_transfer[sample_idx]
                target = torch.tensor(tokens, dtype=cs.dtype, device=cs.device)
                idx = int(torch.searchsorted(cs, target, right=False).item())
                offset = idx + 1
            offsets.append(offset)

        offsets_tensor = torch.tensor(offsets, device=device)
        if torch.any(offsets_tensor != offsets_tensor[0]):
            raise ValueError('Interventions must align across the batch for linear scheduling.')
        block_offsets.append(int(offsets_tensor[0].item()))

    return num_blocks, steps_per_block, block_offsets, base_transfer

@ torch.no_grad()
def generate_w_kappa(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                    cfg_scale=0., remasking='low_confidence', mask_id=126336,
                    fig_name='kappa_over_steps.jpg', json_name='kappa_values.json',
                    intervened_output=None):
    '''
    Same sampling procedure as `generate`, but tracks the dependency correction term, kappa,
    between consecutive sampling steps. For every step (except the first), it compares the
    probabilities assigned to the tokens unmasked at that step under two distributions:
        1. The distribution before the previous step committed its tokens (p(x2 | x3)).
        2. The distribution after conditioning on the previous step's committed tokens (p(x2 | x1, x3)).

    The ratio between the two (kappa) indicates how much the conditional probabilities shift
    once the newly revealed tokens are taken into account. We record kappa for each step and
    export both a plot and a JSON summary for downstream analysis.
    Optionally accepts `intervened_output` to pre-fill part of the generation segment.
    '''
    batch_size = prompt.shape[0]
    if intervened_output is not None:
        if intervened_output.shape != (batch_size, gen_length):
            raise ValueError(f'intervened_output must have shape {(batch_size, gen_length)}, '
                             f'got {tuple(intervened_output.shape)}.')
        gen_segment = intervened_output.to(model.device, non_blocking=True).long().clone()
    else:
        gen_segment = torch.full((batch_size, gen_length), mask_id, dtype=torch.long, device=model.device)
    x = torch.cat([prompt.clone().to(model.device), gen_segment], dim=1)

    if x.size(0) != 1:
        raise ValueError('generate_w_topk currently supports batch size 1 for analysis.')

    prompt_index = (x != mask_id)
    prompt_length = prompt.shape[1]
    config = getattr(model, 'config', None)
    fallback_vocab = getattr(config, 'vocab_size', None)
    if fallback_vocab is None:
        fallback_vocab = top_k

    total_steps = steps
    num_blocks, steps_per_block, block_offsets, tokens_per_step = _prepare_block_schedule(
        gen_segment, mask_id, gen_length, block_length, total_steps
    )

    figs_dir = Path(__file__).resolve().parent / 'figs'
    figs_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figs_dir / fig_name
    json_path = figs_dir / json_name

    EPS = 1e-20
    kappa_steps = []
    kappa_values = []
    log_kappa_values = []
    tokens_per_step_record = []
    kappa_details = []

    processed_any_step = False
    for num_block in range(num_blocks):
        block_start = prompt_length + num_block * block_length
        block_end = prompt_length + (num_block + 1) * block_length

        prev_pending = {}

        for i in range(block_offsets[num_block], steps_per_block):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            selected_positions = None
            tokens_this_step = tokens_per_step[:, i]
            for j in range(confidence.shape[0]):
                k = int(tokens_this_step[j].item())
                if k <= 0:
                    continue
                block_mask = mask_index[j, block_start:block_end]
                available_in_block = int(block_mask.sum().item())
                if available_in_block == 0:
                    continue
                k = min(k, available_in_block)
                block_confidence = confidence[j, block_start:block_end]
                _, select_relative = torch.topk(block_confidence, k=k)
                select_index = select_relative + block_start
                transfer_index[j, select_index] = True
                if j == 0:
                    selected_positions = select_index
            x[transfer_index] = x0[transfer_index]

            current_selected_info = []
            if selected_positions is not None and selected_positions.numel() > 0:
                selected_positions = selected_positions[selected_positions >= prompt_length]
                valid = (selected_positions >= block_start) & (selected_positions < block_end)
                selected_positions = selected_positions[valid]
                if selected_positions.numel() > 0:
                    pos_list = selected_positions.detach().cpu().tolist()
                    prob_list = x0_p[0, selected_positions].detach().cpu().tolist()
                    token_list = x0[0, selected_positions].detach().cpu().tolist()
                    for pos, prob, token in zip(pos_list, prob_list, token_list):
                        current_selected_info.append((pos, prob, token))

            global_step = num_block * steps_per_block + i

            if processed_any_step:
                cond_logs = []
                prior_logs = []
                missing_reason = None

                if current_selected_info:
                    for pos, prob, token in current_selected_info:
                        prev_data = prev_pending.get(pos)
                        if prev_data is None:
                            missing_reason = 'missing_prior'
                            break
                        prev_token, prev_log_prob = prev_data
                        if prev_token != token:
                            missing_reason = 'token_mismatch'
                            break
                        if not np.isfinite(prob) or prob <= 0.:
                            cond_logs.append(-np.inf)
                        else:
                            cond_logs.append(float(np.log(max(prob, EPS))))
                        prior_logs.append(prev_log_prob)
                else:
                    missing_reason = 'no_tokens'

                if missing_reason is None:
                    log_cond_total = float(np.sum(cond_logs))
                    log_prior_total = float(np.sum(prior_logs))
                    log_kappa_val = log_cond_total - log_prior_total
                    kappa_val = float(np.exp(log_kappa_val))
                else:
                    log_kappa_val = float('nan')
                    kappa_val = float('nan')

                kappa_steps.append(global_step)
                kappa_values.append(kappa_val)
                log_kappa_values.append(log_kappa_val)
                tokens_per_step_record.append(len(current_selected_info))

                rel_positions = [pos - prompt_length for pos, _, _ in current_selected_info]
                kappa_details.append({
                    'step': int(global_step),
                    'positions': rel_positions,
                    'tokens': [token for _, _, token in current_selected_info],
                    'num_tokens': len(current_selected_info),
                    'kappa': None if not np.isfinite(kappa_val) else kappa_val,
                    'log_kappa': None if not np.isfinite(log_kappa_val) else log_kappa_val,
                    'status': 'ok' if missing_reason is None else missing_reason
                })

            remaining_mask_index = (x == mask_id)
            remaining_positions = torch.nonzero(remaining_mask_index[0], as_tuple=False).squeeze(-1)
            new_pending = {}
            if remaining_positions.numel() > 0:
                remaining_positions = remaining_positions[
                    (remaining_positions >= block_start) & (remaining_positions < block_end)
                ]
                if remaining_positions.numel() > 0:
                    pos_list = remaining_positions.detach().cpu().tolist()
                    prob_list = x0_p[0, remaining_positions].detach().cpu().tolist()
                    token_list = x0[0, remaining_positions].detach().cpu().tolist()
                    for pos, prob, token in zip(pos_list, prob_list, token_list):
                        if not np.isfinite(prob) or prob <= 0.:
                            continue
                        new_pending[pos] = (token, float(np.log(max(prob, EPS))))
            prev_pending = new_pending
            processed_any_step = True

    if kappa_steps:
        steps_arr = np.array(kappa_steps)
        kappa_arr = np.array(kappa_values, dtype=np.float64)
    else:
        steps_arr = np.array([])
        kappa_arr = np.array([])

    fig, ax = plt.subplots(figsize=(8, 4))
    if steps_arr.size > 0:
        ax.plot(steps_arr, kappa_arr, marker='o', linestyle='-')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Kappa')
    ax.set_title('Dependency Correction Term Across Steps')
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    json_dump = {
        'steps': kappa_steps,
        'kappa': [None if not np.isfinite(k) else k for k in kappa_values],
        'log_kappa': [None if not np.isfinite(k) else k for k in log_kappa_values],
        'tokens_per_step': tokens_per_step_record,
        'details': kappa_details
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_dump, f, ensure_ascii=False, indent=2)

    return x

@ torch.no_grad()
def generate_w_topk(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                    cfg_scale=0., remasking='low_confidence', mask_id=126336,
                    top_k=5, fig_name='next_step_rank.jpg', json_name='next_step_rank.json',
                    intervened_output=None):
    '''
    Same sampling procedure as `generate`, but tracks whether the tokens revealed at the current
    step already appeared in the previous step's top-k logits. The resulting scatter plot marks,
    for each step (except the first), the prior-step rank of the selected token, with an extra row
    indicating cases that were absent from the prior top-k set.
    Optionally accepts `intervened_output` to pre-fill part of the generation segment.
    '''
    if top_k < 1:
        raise ValueError('top_k must be a positive integer.')

    batch_size = prompt.shape[0]
    if intervened_output is not None:
        if intervened_output.shape != (batch_size, gen_length):
            raise ValueError(f'intervened_output must have shape {(batch_size, gen_length)}, '
                             f'got {tuple(intervened_output.shape)}.')
        gen_segment = intervened_output.to(model.device, non_blocking=True).long().clone()
    else:
        gen_segment = torch.full((batch_size, gen_length), mask_id, dtype=torch.long, device=model.device)
    x = torch.cat([prompt.clone().to(model.device), gen_segment], dim=1)

    if x.size(0) != 1:
        raise ValueError('generate_w_topk currently supports batch size 1 for analysis.')

    prompt_index = (x != mask_id)
    prompt_length = prompt.shape[1]
    config = getattr(model, 'config', None)
    fallback_vocab = getattr(config, 'vocab_size', None)
    if fallback_vocab is None:
        fallback_vocab = top_k

    total_steps = steps
    num_blocks, steps_per_block, block_offsets, tokens_per_step_schedule = _prepare_block_schedule(
        gen_segment, mask_id, gen_length, block_length, total_steps
    )

    figs_dir = Path(__file__).resolve().parent / 'figs'
    figs_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figs_dir / fig_name
    json_path = figs_dir / json_name if json_name is not None else None

    prev_topk_indices = None
    rank_records = []
    k_used = None

    for num_block in range(num_blocks):
        block_start = prompt_length + num_block * block_length
        block_end = prompt_length + (num_block + 1) * block_length

        for i in range(block_offsets[num_block], steps_per_block):
            current_step = num_block * steps_per_block + i
            mask_index = (x == mask_id)

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            vocab_size = logits.shape[-1]
            if k_used is None:
                k_used = min(top_k, vocab_size)
            current_topk_indices = torch.topk(logits, k=k_used, dim=-1).indices
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            selected_positions = None
            tokens_this_step = tokens_per_step_schedule[:, i]
            for j in range(confidence.shape[0]):
                k_tokens = int(tokens_this_step[j].item())
                if k_tokens <= 0:
                    continue
                block_mask = mask_index[j, block_start:block_end]
                available_in_block = int(block_mask.sum().item())
                if available_in_block == 0:
                    continue
                k_tokens = min(k_tokens, available_in_block)
                block_confidence = confidence[j, block_start:block_end]
                _, select_relative = torch.topk(block_confidence, k=k_tokens)
                select_index = select_relative + block_start
                transfer_index[j, select_index] = True
                if j == 0:
                    selected_positions = select_index
            x[transfer_index] = x0[transfer_index]

            if prev_topk_indices is not None and selected_positions is not None and selected_positions.numel() > 0:
                for pos_tensor in selected_positions:
                    pos = int(pos_tensor.item())
                    if pos < prompt_length or pos >= block_end:
                        continue
                    token_id = int(x0[0, pos].item())
                    prev_tokens = prev_topk_indices[0, pos]
                    match = torch.nonzero(prev_tokens == token_id, as_tuple=False)
                    rank = int(match[0].item()) + 1 if match.numel() > 0 else None
                    rank_records.append({
                        'step': current_step + 1,
                        'position': pos - prompt_length,
                        'token_id': token_id,
                        'rank': rank
                    })

            prev_topk_indices = current_topk_indices.detach().cpu()

    plot_steps = []
    plot_ranks = []
    rank_flags = []
    for record in rank_records:
        plot_steps.append(record['step'])
        if record['rank'] is None:
            plot_ranks.append(k_used + 1)
            rank_flags.append(False)
        else:
            plot_ranks.append(record['rank'])
            rank_flags.append(True)

    fig, ax = plt.subplots(figsize=(8, 4))
    if plot_steps:
        colors = []
        base_cmap = plt.cm.get_cmap('tab10')
        for has_match, value in zip(rank_flags, plot_ranks):
            if has_match:
                colors.append(base_cmap((value - 1) % base_cmap.N))
            else:
                colors.append('tab:red')
        ax.scatter(plot_steps, plot_ranks, c=colors, s=30, edgecolors='black', linewidths=0.5)

    if k_used is None:
        k_used = min(top_k, fallback_vocab)
    ax.set_xlim(1, max(total_steps, 1))
    ax.set_ylim(0.5, k_used + 1.5)
    y_ticks = list(range(1, k_used + 1))
    y_tick_labels = [f'Top-{idx}' for idx in y_ticks]
    y_ticks.append(k_used + 1)
    y_tick_labels.append(f'Not in Top-{k_used}')
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)

    xticks = np.arange(0, total_steps + 1, 50)
    if total_steps not in xticks:
        xticks = np.append(xticks, total_steps)
    ax.set_xticks(np.unique(xticks))

    ax.set_xlabel('Step')
    ax.set_ylabel('Previous Step Rank')
    ax.set_title(f'Next-Step Rank (Top-{k_used})')
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    if json_path is not None:
        json_dump = {
            'top_k': k_used,
            'total_steps': total_steps,
            'records': rank_records
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_dump, f, ensure_ascii=False, indent=2)

    return x

@ torch.no_grad()
def generate_w_conf(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                    cfg_scale=0., remasking='low_confidence', mask_id=126336,
                    log_interval=20, top_k=10, json_name='step_top_logits.json',
                    fig_name='step_top_logits.pdf', tokenizer=None, intervened_output=None):
    '''
    Same as `generate`, but records the top-k logits for every masked position at a fixed interval.
    Additionally, for each logged step it plots a two-row subplot: (i) the max probability per
    generation position derived from the recorded logits and (ii) a table of the top-k decoded
    tokens per position. The resulting figure stacks the per-step subplots vertically and is saved
    as a PDF.
    Optionally accepts `intervened_output` to pre-fill part of the generation segment.
    '''
    if log_interval < 1:
        raise ValueError('log_interval must be a positive integer.')
    if top_k < 1:
        raise ValueError('top_k must be a positive integer.')
    if top_k < 10:
        raise ValueError('top_k must be at least 10 to build the requested table.')

    batch_size = prompt.shape[0]
    if intervened_output is not None:
        if intervened_output.shape != (batch_size, gen_length):
            raise ValueError(f'intervened_output must have shape {(batch_size, gen_length)}, '
                             f'got {tuple(intervened_output.shape)}.')
        gen_segment = intervened_output.to(model.device, non_blocking=True).long().clone()
    else:
        gen_segment = torch.full((batch_size, gen_length), mask_id, dtype=torch.long, device=model.device)
    x = torch.cat([prompt.clone().to(model.device), gen_segment], dim=1)

    tokenizer = _ensure_tokenizer(model, tokenizer)

    if x.size(0) != 1:
        raise ValueError('generate_w_prob currently supports batch size 1 for visualization.')

    prompt_index = (x != mask_id)
    prompt_length = prompt.shape[1]

    total_steps = steps
    num_blocks, steps_per_block, block_offsets, tokens_per_step_schedule = _prepare_block_schedule(
        gen_segment, mask_id, gen_length, block_length, total_steps
    )

    figs_dir = Path(__file__).resolve().parent / 'figs'

    logged_topk = {}
    logged_states = {}
    logged_steps = []
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length

        for i in range(block_offsets[num_block], steps_per_block):
            current_step = num_block * steps_per_block + i
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            if current_step % log_interval == 0:
                mask_positions = torch.nonzero(mask_index, as_tuple=False)
                if mask_positions.numel() > 0:
                    step_entries = []
                    vocab_top = min(top_k, logits.size(-1))
                    for pos in mask_positions:
                        batch_idx = int(pos[0].item())
                        token_idx = int(pos[1].item())
                        top_values, top_indices = torch.topk(logits[batch_idx, token_idx], k=vocab_top)
                        top_tokens = [
                            {
                                'token_id': int(token_id),
                                'logit': float(logit_val)
                            }
                            for token_id, logit_val in zip(
                                top_indices.detach().cpu().tolist(),
                                top_values.detach().to(torch.float32).cpu().tolist()
                            )
                        ]
                        step_entries.append({
                            'batch': batch_idx,
                            'position': token_idx,
                            'top_tokens': top_tokens
                        })
                    step_key = current_step + 1
                    logged_topk[str(step_key)] = step_entries
                    logged_steps.append(step_key)
                    logged_states[step_key] = x.detach().clone().to('cpu')
                else:
                    step_key = current_step + 1
                    logged_topk[str(step_key)] = []
                    logged_steps.append(step_key)
                    logged_states[step_key] = x.detach().clone().to('cpu')

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            tokens_this_step = tokens_per_step_schedule[:, i]
            for j in range(confidence.shape[0]):
                k_tokens = int(tokens_this_step[j].item())
                if k_tokens <= 0:
                    continue
                block_mask = mask_index[j, block_start:block_end]
                available_in_block = int(block_mask.sum().item())
                if available_in_block == 0:
                    continue
                k_tokens = min(k_tokens, available_in_block)
                block_confidence = confidence[j, block_start:block_end]
                _, select_relative = torch.topk(block_confidence, k=k_tokens)
                select_index = select_relative + block_start
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    _export_topk_visualization(
        logged_steps,
        logged_topk,
        logged_states,
        prompt_length,
        gen_length,
        mask_id,
        tokenizer,
        figs_dir,
        fig_name,
        json_name
    )

    return x

def main():
    device = 'cuda:0'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours? Answer the question with detailed explanation."
    # prompt = "Let $x,y$ and $z$ be positive real numbers that satisfy the following system of equations:\n\n\\[\\log_2\\left({x \\over yz}\\right) = {1 \\over 2}\]\n\n\\[\\log_2\left({y \\over xz}\\right) = {1 \\over 3}\]\n\n\\[\\log_2\\left({z \\over xy}\\right) = {1 \\over 4}\]\n\nThen the value of $\\left|\\log_2(x^4y^3z^2)\\right|$ is $\\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$. Answer the question with detailed explanation."
    # prompt = "Please recommand a movies."

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    print(prompt)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    steps = 128; gen_length = 128; block_length = 128

    intervened = build_intervened_output(
        tokenizer,
        gen_length,
        mask_id=126336,
        interventions=[
            {'text': '12 * 4 = 48', 'position': 30},
            # {'text': '6 * 4 = 24', 'position': 60},
        ],
        device=model.device,
        batch_size=1,
    )

    # out = generate(model, input_ids, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., cfg_scale=0., remasking='low_confidence')
    # out = generate_w_order(model, input_ids, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., cfg_scale=0., remasking='low_confidence')
    # out = generate_w_likelihood(model, input_ids, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., cfg_scale=0., remasking='low_confidence')
    out = generate_w_fig(model, input_ids, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., cfg_scale=0., remasking='low_confidence')
    
    '''
    Intervened generation example:
    '''
    # out = generate_w_order(model, input_ids, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., cfg_scale=0., remasking='low_confidence', intervened_output=intervened)
    # out = generate_w_conf(model, input_ids, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., cfg_scale=0., remasking='low_confidence', intervened_output=intervened)
    # out = generate_w_likelihood(model, input_ids, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., cfg_scale=0., remasking='low_confidence', intervened_output=intervened)
    
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
