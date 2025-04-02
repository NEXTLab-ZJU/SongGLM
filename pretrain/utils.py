from typing import Dict

import torch
import torch.nn.functional as F

from pretrain.tokenizer import OctupleLikeTokenizer

def get_sampling_strategy(field_index: str) -> Dict[str, float]:
    field_name = OctupleLikeTokenizer().field_names[field_index]
    
    if field_name == "bar":
        return {"temperature": 1.0, "top_p": 0.9}
    elif field_name == "position":
        return {"temperature": 1.2, "top_p": 1.0}
    
    elif field_name == "duration":
        return {"temperature": 2.0, "top_p": 0.9}
    elif field_name == "pitch":
        return {"temperature": 1.0, "top_p": 0.9}
    elif field_name == "tempo":
        return {"temperature": 1.2, "top_p": 0.9}
    elif field_name == "velocity":
        return {"temperature": 1.2, "top_p": 0.9}
    
    elif field_name == "lyrics":
        return {"temperature": 1.2, "top_p": 0.9}
    
    elif field_name == "group":
        return {"temperature": 1.0, "top_p": 0.9}
    elif field_name == "phrase":
        return {"temperature": 1.0, "top_p": 0.9}
    
    elif field_name == "token_type":
        return {"temperature": 1.0, "top_p": 0.9}
    elif field_name == "task_id":
        return {"temperature": 1.0, "top_p": 0.9}
    
    else:
        raise ValueError(f"Unknown field name: {field_name}")


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def top_k_top_p_sample(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    temperature: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Sample from a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    """
    assert temperature > 0, "t has to be strictly positive"
    if temperature != 1.0:
        logits = logits / temperature
    logits = top_k_top_p_filtering(
        logits, top_k=top_k, top_p=top_p, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep
    )
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def gumbel_sample(logits: torch.Tensor, t: float = 1.0) -> torch.Tensor:
    noise = torch.zeros_like(logits, device=logits.device).uniform_(0, 1)
    noise = -torch.log(-torch.log(noise))
    return torch.argmax((logits / t) + noise, dim=-1)
