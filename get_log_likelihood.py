import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

def restore_substring_tokens(tokenizer, original_ids: torch.Tensor, masked_ids: torch.Tensor,
                             substring: str, mask_id: int = 126336):
    """
    original_ids: (L,)  원본 전체 시퀀스 토큰 (special tokens 포함 가능)
    masked_ids:   (L,)  현재 마스크된(혹은 부분 복구된) 시퀀스 토큰
    substring:           복구하고 싶은 원문 문자열 (등장하는 모든 구간을 복구)
    mask_id:             [MASK] id (기본: 126336)

    반환: updated_ids, restored_count
          - updated_ids: 복구가 반영된 masked_ids
          - restored_count: 이번 호출로 mask_id -> 원래 토큰으로 바뀐 개수
    """
    assert original_ids.dim() == 1 and masked_ids.dim() == 1, "1D torch.Tensor를 넣어주세요."
    assert original_ids.shape == masked_ids.shape, "original_ids와 masked_ids 길이가 같아야 합니다."

    device = masked_ids.device
    
    sub_ids = tokenizer(substring, add_special_tokens=False)["input_ids"]

    # 간단한 부분수열 매칭 함수
    def find_all_occurrences(haystack, needle):
        pos = []
        Lh, Ln = len(haystack), len(needle)
        if Ln == 0 or Ln > Lh:
            return pos
        for i in range(Lh - Ln + 1):
            if haystack[i:i+Ln] == needle:
                pos.append(i)
        return pos

    orig_list = original_ids.tolist()

    total_restored = 0
    updated = masked_ids.clone()

    hits = find_all_occurrences(orig_list, sub_ids)
    for start in hits:
        end = start + len(sub_ids)
        # mask_id -> 원래 토큰으로 바뀌는 개수만 카운트
        was_mask = (updated[start:end] == mask_id)
        total_restored += was_mask.sum().item()
        # 원래 토큰으로 복구
        updated[start:end] = torch.as_tensor(orig_list[start:end], dtype=updated.dtype, device=device)

    print(f"[restore_substring_tokens] Restored tokens: {total_restored}")
    return updated, total_restored

def forward_process(batch, prompt_index, mask_id):
    b, l = batch.shape

    target_len = (l - prompt_index.sum()).item()
    k = torch.randint(1, target_len + 1, (), device=batch.device)

    x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
    x = ((x - 1) % target_len) + 1
    assert x.min() >= 1 and x.max() <= target_len

    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len)]

    is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
    noisy_batch = torch.where(is_mask, mask_id, batch)

    # Return the masked batch and the mask ratio
    return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)


def get_logits(model, batch, prompt_index, cfg_scale, mask_id):
    if cfg_scale > 0.:
        assert len(prompt_index) == batch.shape[1]
        prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
        un_batch = batch.clone()
        un_batch[prompt_index] = mask_id
        batch = torch.cat([batch, un_batch])

    input = batch
    logits = model(input).logits

    if cfg_scale > 0.:
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    return logits


@ torch.no_grad()
def get_log_likelihood(model, prompt, answer, mc_num=128, batch_size=16, cfg_scale=0., mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (l1).
        answer: A tensor of shape (l2).
        mc_num: Monte Carlo estimation times.
                As detailed in Appendix B.5. Since MMLU, CMMLU, and C-EVAL only require the likelihood of a single token, a
                single Monte Carlo estimate is sufficient for these benchmarks. For all other benchmarks, we find that 128
                Monte Carlo samples are adequate to produce stable results.
        batch_size: Mini batch size.
        cfg_scale: Unsupervised classifier-free guidance scale.
        mask_id: The toke id of [MASK] is 126336.
    '''
    seq = torch.concatenate([prompt, answer])[None, :]
    seq = seq.repeat((batch_size, 1)).to(model.device)
    prompt_index = torch.arange(seq.shape[1], device=model.device) < len(prompt)

    loss_ = []
    for _ in range(mc_num // batch_size):
        perturbed_seq, p_mask = forward_process(seq, prompt_index, mask_id)
        mask_index = perturbed_seq == mask_id

        logits = get_logits(model, perturbed_seq, prompt_index, cfg_scale, mask_id)

        loss = F.cross_entropy(logits[mask_index], seq[mask_index], reduction='none') / p_mask[mask_index]
        loss = loss.sum() / batch_size

        loss_.append(loss.item())

    return - sum(loss_) / len(loss_)

@ torch.no_grad()
def get_fixed_log_likelihood(model, prompt, answer, cfg_scale=0., mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (l1).
        answer: A tensor of shape (l2). Here the noise is fixed.
        cfg_scale: Unsupervised classifier-free guidance scale.
        mask_id: The toke id of [MASK] is 126336.
    '''
    seq = torch.concatenate([prompt, answer])[None, :]
    seq = seq.repeat((1, 1)).to(model.device)
    prompt_index = torch.arange(seq.shape[1], device=model.device) < len(prompt)

    target_len = seq.shape[1] - len(prompt)
    mask_ratio = (seq[:, len(prompt):] == mask_id).float().sum(dim=1, keepdim=True) / max(1, target_len)
    p_mask = mask_ratio.repeat(1, seq.shape[1]).clamp_min(1e-8)
    mask_index = seq == mask_id

    logits = get_logits(model, seq, prompt_index, cfg_scale, mask_id)

    loss = F.cross_entropy(logits[mask_index], seq[mask_index], reduction='none') / p_mask[mask_index]
    loss = loss.sum()

    return - loss.item()

def main():
    device = 'cuda'

    # model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    # tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    # this prompt and answer is from Hellaswag dataset
    # prompt = 'Roof shingle removal: A man is sitting on a roof. He'
    # answer = ' is using wrap to wrap a pair of skis.'

    prompt = '<|startoftext|><|start_header_id|>user<|end_header_id|>\n\nLily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours? Answer the question with detailed explanation.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    answer = 'First, calculate the distance Lily runs in the first 4 hours. Since she runs 12 kilometers per hour for 4 hours, the distance is 12 * 4 = 48 kilometers.\n\nNext, calculate the distance Lily runs in the next 4 hours. Since she runs 6 kilometers per hour for 4 hours, the distance is 6 * 4 = 24 kilometers.\n\nNow, add the distances from both periods: 48 + 24 = 72 kilometers.\n\nTherefore, Lily can run 72 kilometers in 8 hours.<|eot_id|><|endoftext|>'

    prompt = torch.tensor(tokenizer(prompt)['input_ids']).to(device)
    answer = torch.tensor(tokenizer(answer)['input_ids']).to(device)
    # print(get_log_likelihood(model, prompt, answer, mc_num=128))
    masked = torch.full_like(answer, 126336)

    substrings_semantic = [
        "12 * 4 = 48 kilometers.",
        "6 * 4 = 24 kilometers.",
        "48 + 24 = 72 kilometers."
    ]

    substrings_orig = [
        " run 72 kilometers in 8 hours.<|eot_id|><|endoftext|>",
        "72 kilometers.\n\nTherefore, Lily can",
        " both periods: 48 + 24 = "
    ]
    
    for substring in substrings_semantic:
    # for substring in substrings_orig:
        masked, n1 = restore_substring_tokens(tokenizer, answer, masked, substring, mask_id=126336)
        print("Unmask ->", substring)
        print(tokenizer.decode(masked))
        print(get_fixed_log_likelihood(model, prompt, masked))


if __name__ == '__main__':
    main()