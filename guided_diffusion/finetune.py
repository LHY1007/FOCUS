"""Freeze encoders and add zero-initialized low-rank updates to task weights."""
import math
import torch
from torch import nn
from torch.nn.utils import parametrize


class LowRankUpdate(nn.Module):
    def __init__(self, weight, rank):
        super().__init__()
        self.shape = weight.shape
        rows, cols = weight.shape[0], weight.numel() // weight.shape[0]
        rank = min(int(rank), rows, cols)
        if rank <= 0:
            raise ValueError('LoRA rank must be positive')
        self.A = nn.Parameter(weight.new_zeros((rows, rank)))
        self.B = nn.Parameter(weight.new_empty((rank, cols)))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def forward(self, weight):
        return weight + (self.A @ self.B).view(self.shape)


def configure_finetuning(model, rank=8):
    if not model.manuscript_mode:
        raise ValueError('Fine-tuning requires manuscript_mode')
    # Decoder and conditional matching/fusion remain trainable. Encoder and
    # time embedding preserve the foundation representation.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith((
            'input_blocks.', 'input_blocks_lr.', 'input_blocks_WSI5120.',
            'input_blocks_WSI320.', 'time_embed.',
        )))
    roots = [model.manuscript.alignment, model.manuscript.coordinator,
             model.manuscript.cell_fusion, model.manuscript.cell_query,
             model.manuscript.cell_key, model.manuscript.cell_value,
             model.MemoryNetwork, model.pre, model.post]
    count = 0
    for root in roots:
        for module in list(root.modules()):
            for name, weight in list(module.named_parameters(recurse=False)):
                if weight.ndim < 2:
                    continue
                weight.requires_grad_(False)
                parametrize.register_parametrization(module, name, LowRankUpdate(weight, rank))
                count += 1
    return count


def merged_state_dict(model):
    """Export ordinary effective weights for inference without LoRA wrappers."""
    import copy
    merged = copy.deepcopy(model)
    for module in list(merged.modules()):
        if parametrize.is_parametrized(module):
            for name in list(module.parametrizations):
                parametrize.remove_parametrizations(module, name, leave_parametrized=True)
    return merged.state_dict()
