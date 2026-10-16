"""GPU integration check with explicitly synthetic inputs; no accuracy claims."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import torch
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Warm CUDA before Scanpy-linked native libraries on the validated CB4 stack.
if torch.cuda.is_available():
    torch.cuda.init()
from guided_diffusion.script_util import sr_create_model_and_diffusion
from guided_diffusion.finetune import configure_finetuning, merged_state_dict


def main():
    torch.manual_seed(11)
    torch.set_num_threads(4)
    config = yaml.safe_load(Path('config/config_train.yaml').read_text())
    args = SimpleNamespace(**config)
    model, diffusion = sr_create_model_and_diffusion(args)
    model = model.cuda().train()
    batch, genes, side = 2, 5, 256
    x = torch.rand(batch, genes, side, side, device='cuda')
    he = torch.rand(batch, 3, side, side, device='cuda') * 255
    tiles = he.unfold(2, 16, 16).unfold(3, 16, 16).permute(0, 2, 3, 1, 4, 5).reshape(batch, 256, 3, 16, 16)
    masks = torch.zeros(batch, 12, side, side, device='cuda')
    for i in range(12):
        r, c = 2 + (i // 4) * 5, 2 + (i % 4) * 3
        masks[:, i, r:r + 2, c:c + 2] = 1
    conditions = dict(low_res=torch.rand(batch, genes, 26, 26, device='cuda'),
        WSI_5120=he, WSI_320=tiles, gene_class=torch.arange(genes, device='cuda').repeat(batch, 1),
        Gene_index_map=torch.zeros_like(x), metadata_feature=torch.randn(batch, 768, device='cuda'),
        scale_gt=torch.ones(batch, device='cuda'), co_expression=torch.eye(50, device='cuda').repeat(batch, 1, 1),
        WSI_mask=masks, sc=torch.rand(batch, 8, genes, device='cuda'),
        scgpt=torch.randn(batch, 512, 26, 26, device='cuda'), pre_he=torch.randn(batch, 257, 1536, device='cuda'),
        complexity=torch.rand(batch, 16, 16, device='cuda'))
    # Force task activation for this gradient check; separate tests exercise disabled tasks.
    with torch.no_grad():
        for head in model.manuscript.activation:
            head.weight.zero_(); head.bias.fill_(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    warmup = diffusion.training_losses(model, x, torch.tensor([10, 200], device='cuda'), .3, model_kwargs=conditions)
    warmup['loss'].mean().backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    del warmup
    terms = diffusion.training_losses(model, x, torch.tensor([10, 200], device='cuda'), .3, model_kwargs=conditions)
    loss = terms['loss'].mean()
    loss.backward()
    assert torch.isfinite(loss)
    names = ['meta_processor.fc.weight', 'manuscript.alignment.flow.2.weight',
             'manuscript.activation.0.weight', 'manuscript.activation.1.weight',
             'manuscript.activation.2.weight', 'manuscript.coordinator.tokens.0.weight']
    gradients = {}
    parameters = dict(model.named_parameters())
    for name in names:
        grad = parameters[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
        gradients[name] = float(grad.abs().sum())
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    del optimizer, terms
    # Freeze-update behavior and an actual LoRA optimizer step.
    count = configure_finetuning(model, rank=4)
    frozen = ('input_blocks.', 'input_blocks_lr.', 'input_blocks_WSI5120.',
              'input_blocks_WSI320.', 'time_embed.')
    assert all(not p.requires_grad for n, p in model.named_parameters() if n.startswith(frozen))
    terms = diffusion.training_losses(model, x, torch.tensor([20, 300], device='cuda'), .7, model_kwargs=conditions)
    terms['loss'].mean().backward()
    adapted = [p.grad for n, p in model.named_parameters() if '.parametrizations.' in n and n.endswith('.A')]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in adapted)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    state = merged_state_dict(model)
    assert not any('parametrizations' in k for k in state)
    restored, _ = sr_create_model_and_diffusion(args)
    restored.load_state_dict(state, strict=True)
    restored.cuda().eval()
    # Complete reverse chain with two respaced steps, explicitly for execution validation.
    config['timestep_respacing'] = '2'
    _, fast_diffusion = sr_create_model_and_diffusion(SimpleNamespace(**config))
    with torch.inference_mode():
        conditions['last_HQST'] = torch.zeros_like(x)
        prediction = fast_diffusion.ddim_sample_loop(restored, x.shape, model_kwargs=conditions)
    assert prediction.shape == (2, 5, 256, 256) and torch.isfinite(prediction).all()
    # Disabled experts must not execute; the shared five-channel backbone still runs.
    with torch.no_grad():
        for head in restored.manuscript.activation:
            head.weight.zero_(); head.bias.fill_(-1)
    one = {key: value[:1] for key, value in conditions.items()}
    def forbidden(*args, **kwargs):
        raise AssertionError('inactive expert executed')
    with patch.object(restored.MemoryNetwork, 'forward', side_effect=forbidden), \
         patch.object(restored.manuscript.alignment, 'forward', side_effect=forbidden), \
         patch.object(restored.input_blocks_WSI320[0], 'forward', side_effect=forbidden):
        with torch.inference_mode():
            inactive = restored(torch.cat((torch.zeros_like(x[:1]), x[:1]), 1),
                                torch.tensor([5], device='cuda'), **one)
    assert inactive[-1].shape == (1, 5, 256, 256)
    assert not restored.task_activation_mask.any() and restored.resolution_activation
    result = dict(input_kind='synthetic integration fixture', loss=float(loss.detach()),
                  gradients=gradients, lora_matrices=count, encoder_frozen=True,
                  merged_checkpoint_strict_load=True, prediction_shape=list(prediction.shape),
                  finite_prediction=True, sampling_steps=2, inactive_experts_skipped=True)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
