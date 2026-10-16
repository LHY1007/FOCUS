"""New implementations of the supplied manuscript equations.

Architecture choices absent from the paper are documented in docs/reproduction.md.
These are not recovered historical training code or historical checkpoints.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .morphology import complexity_grid, boundary_weights


def reference_projection(reference, spatial, reg=10.0, iterations=100):
    """Differentiable log-domain Sinkhorn with uniform marginals (Methods 1.3)."""
    b, genes, h, w = spatial.shape
    if reference.ndim != 3 or reference.shape[0] != b or reference.shape[2] != genes:
        raise ValueError('Reference must be [batch, cells, selected genes] in ST gene order')
    y = spatial.flatten(2).transpose(1, 2)
    cost = torch.cdist(reference, y).square()
    log_kernel = -cost / reg
    log_a = spatial.new_full((b, reference.shape[1]), -__import__('math').log(reference.shape[1]))
    log_b = spatial.new_full((b, h * w), -__import__('math').log(h * w))
    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(iterations):
        u = log_a - torch.logsumexp(log_kernel + v[:, None], 2)
        v = log_b - torch.logsumexp(log_kernel + u[:, :, None], 1)
    plan = (log_kernel + u[:, :, None] + v[:, None]).exp()
    return (plan.transpose(1, 2) @ reference).transpose(1, 2).reshape(b, genes, h, w)


def pearson_cells(embeddings):
    centered = embeddings - embeddings.mean(-1, keepdim=True)
    normalized = F.normalize(centered, dim=-1, eps=1e-8)
    return normalized @ normalized.T


def cell_consistency(he_features, st_features, masks, scores, patch_size=16):
    """PCC across embedding coordinates, then vectorized cosine loss (Methods 1.5, p. 35)."""
    size = masks.shape[-2:]
    he_features = F.interpolate(he_features, size=size, mode='bilinear', align_corners=False)
    st_features = F.interpolate(st_features, size=size, mode='bilinear', align_corners=False)
    losses = []
    for he, st, stack, score in zip(he_features, st_features, masks, scores):
        row, col = divmod(int(score.argmax()), score.shape[1])
        r, c = row * patch_size, col * patch_size
        crop = stack[:, r:r + patch_size, c:c + patch_size]
        crop = crop[crop.sum((1, 2)) > 0]
        if len(crop) < 2:
            losses.append((he.sum() + st.sum()) * 0)
            continue
        weights = crop / crop.sum((1, 2), keepdim=True).clamp_min(1e-8)
        e_he = torch.einsum('nhw,chw->nc', weights, he[:, r:r + patch_size, c:c + patch_size])
        e_st = torch.einsum('nhw,chw->nc', weights, st[:, r:r + patch_size, c:c + patch_size])
        corr_he, corr_st = pearson_cells(e_he), pearson_cells(e_st)
        cosine = F.cosine_similarity(corr_he.flatten()[None], corr_st.flatten()[None], dim=1)[0]
        losses.append((1 - cosine) / 2)
    return torch.stack(losses)


class MorphologyAlignment(nn.Module):
    def __init__(self, channels, noise_std=0.01, c=1e-8):
        super().__init__()
        self.c, self.noise_std = c, noise_std
        self.local_kernels = nn.ModuleList([nn.Conv2d(2 * channels, 9, 3, padding=1) for _ in range(2)])
        self.shared = nn.Conv2d(channels, channels // 2, 3, padding=1)
        self.unique = nn.ModuleList([nn.Conv2d(channels, channels // 2, 3, padding=1) for _ in range(2)])
        self.structure = nn.Conv2d(channels // 2, 1, 3, padding=1)
        self.flow = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.SiLU(), nn.Conv2d(16, 2, 3, padding=1))
        nn.init.zeros_(self.flow[-1].weight)
        nn.init.zeros_(self.flow[-1].bias)
        self.fuse = nn.Conv2d(2 * channels, channels, 1)
        sobel = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
        self.register_buffer('sobel', torch.stack([sobel, sobel.T])[:, None])

    def gradient(self, x):
        return F.conv2d(x, self.sobel.to(x), padding=1)

    def forward(self, st, he, masks):
        st, he = F.normalize(st, dim=1), F.normalize(he, dim=1)
        pair = torch.cat([st, he], 1)
        updated = []
        for x, kernel in zip((st, he), self.local_kernels):
            weights = kernel(pair).softmax(1)
            patches = F.unfold(x, 3, padding=1).view(x.shape[0], x.shape[1], 9, *x.shape[-2:])
            updated.append((patches * weights[:, None]).sum(2))
        shared_st, shared_he = (self.shared(x) for x in updated)
        unique_st, unique_he = (m(x) for m, x in zip(self.unique, updated))
        perturbed = F.normalize(shared_st, dim=1)
        if self.training:
            perturbed = F.normalize(perturbed + torch.randn_like(perturbed) * self.noise_std, dim=1)
        s_st, s_he = self.structure(perturbed), self.structure(shared_he)
        union = masks.amax(1, keepdim=True).clamp(0, 1)
        union = F.interpolate(union, size=s_he.shape[-2:], mode='nearest')
        s_he = s_he * union
        flow = self.flow(torch.cat([s_st, s_he], 1)).tanh() * 0.1
        b, _, h, w = flow.shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=flow.device),
                                torch.linspace(-1, 1, w, device=flow.device), indexing='ij')
        grid = torch.stack([xx, yy], -1)[None].expand(b, -1, -1, -1) + flow.permute(0, 2, 3, 1)
        warped = F.grid_sample(s_st, grid, mode='bilinear', padding_mode='border', align_corners=True)
        a, bgrad = self.gradient(warped), self.gradient(s_he)
        ma = (a.square().sum(1, keepdim=True) + self.c).sqrt()
        mb = (bgrad.square().sum(1, keepdim=True) + self.c).sqrt()
        ngf = 1 - ((a * bgrad).sum(1, keepdim=True) + self.c) / (ma * mb + self.c)
        gmc = (ma - mb).square()
        gms = (2 * ma * mb + self.c) / (ma.square() + mb.square() + self.c)
        gmsc = gms.flatten(1).std(1, unbiased=False)
        weight = boundary_weights(masks, (h, w)).to(st)
        structural = (weight * (ngf + gmc + gmsc[:, None, None, None])).mean((1, 2, 3))
        self.last_components = {'ngf': ngf.detach().mean(), 'gmc': gmc.detach().mean(),
                                'gmsc': gmsc.detach().mean(), 'flow': flow.detach()}
        features = self.fuse(torch.cat([shared_st, shared_he, unique_st, unique_he], 1))
        return features, structural, (shared_he, shared_st, unique_he, unique_st)


class ActiveTaskCoordinator(nn.Module):
    """Three experts; inactive tokens never enter attention, including mixed batches."""
    def __init__(self, channels):
        super().__init__()
        self.tokens = nn.ModuleList([nn.Linear(channels, channels) for _ in range(3)])
        self.attention = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.film = nn.ModuleList([nn.Linear(channels, 2 * channels) for _ in range(3)])
        self.gates = nn.Parameter(torch.full((3,), -2.0))
        self.router = nn.Linear(768, 3)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    def forward(self, features, active, prompt, scores):
        out = []
        route = self.router(prompt).sigmoid()
        for b in range(active.shape[0]):
            selected = torch.where(active[b])[0].tolist()
            residual = torch.zeros_like(features[0][b:b + 1])
            if selected:
                tokens = torch.stack([self.tokens[i](features[i][b].mean((1, 2))) for i in selected])[None]
                context, _ = self.attention(tokens, tokens, tokens, need_weights=False)
                for j, i in enumerate(selected):
                    alpha, beta = self.film[i](context[0, j]).chunk(2)
                    feature = features[i][b:b + 1]
                    modulated = (1 + alpha[None, :, None, None]) * feature + beta[None, :, None, None]
                    # Straight-through gate trains the score head; its forward value is exactly 1.
                    gate = 1 + scores[b, i] - scores[b, i].detach()
                    residual = residual + gate * route[b, i] * self.gates[i].sigmoid() * modulated
            out.append(residual)
        return torch.cat(out)


class ManuscriptConditioning(nn.Module):
    def __init__(self, channels, genes, he_feature_dim=1536, st_feature_dim=512):
        super().__init__()
        self.activation = nn.ModuleList([nn.Linear(2 * genes, 1), nn.Linear(genes + 3, 1),
                                         nn.Linear(genes + 3, 1)])
        for head in self.activation:
            nn.init.normal_(head.weight, std=0.01)
            nn.init.constant_(head.bias, 0.1)
        self.he_match = nn.Linear(he_feature_dim, channels)
        self.st_match = nn.Conv2d(st_feature_dim, channels, 1)
        self.alignment = MorphologyAlignment(channels)
        self.coordinator = ActiveTaskCoordinator(channels)
        self.cell_fusion = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.cell_query = nn.Linear(channels, channels)
        self.cell_key = nn.Linear(channels, channels)
        self.cell_value = nn.Linear(channels, channels)
        self.fuse = nn.Conv2d(channels * 2 + 20 + genes, channels, 1)
        self.allowed_tasks = (True, True, True)
        self.progress = 0.0

    def task_scores(self, low_res, he, reference, masks):
        mean_st = low_res.mean((2, 3))
        ref = reference.mean(1)
        cellular = (he * masks.amax(1, keepdim=True).clamp(0, 1)).mean((2, 3))
        inputs = [torch.cat([mean_st, ref], 1), torch.cat([mean_st, he.mean((2, 3))], 1),
                  torch.cat([mean_st, cellular], 1)]
        scores = torch.cat([head(x) for head, x in zip(self.activation, inputs)], 1).sigmoid()
        allowed = torch.tensor(self.allowed_tasks, device=scores.device, dtype=torch.bool)
        return scores, (scores > 0.5) & allowed[None]

    def cellular_fusion(self, tissue, cellular, scores):
        # Grow the set from low to high complexity. Methods 1.5 also says
        # "above an increasing threshold", which conflicts with this order.
        fused = []
        for b in range(tissue.shape[0]):
            values = scores[b].flatten()
            eligible = torch.where(values > 0)[0]
            if len(eligible) == 0:
                eligible = torch.arange(len(values), device=values.device)
            # Curriculum applies only while training; inference uses every
            # nonempty cellular tile after the full training schedule.
            fraction = (0.1 + 0.9 * min(max(float(self.progress), 0.0), 1.0)) if self.training else 1.0
            count = max(1, min(len(eligible), __import__('math').ceil(fraction * len(eligible))))
            selected = eligible[torch.argsort(values[eligible], stable=True)[:count]]
            tokens = cellular[b, selected].mean((-1, -2))[None]
            queries = tissue[b:b + 1].flatten(2).transpose(1, 2)
            context, _ = self.cell_fusion(self.cell_query(queries), self.cell_key(tokens),
                                         self.cell_value(tokens), need_weights=False)
            fused.append(context.transpose(1, 2).reshape_as(tissue[b:b + 1]))
        return torch.cat(fused)
