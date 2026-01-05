import math
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    b, n, d = x.shape
    x = x.view(b, n, heads, -1)
    x = x.transpose(1, 2)
    x = x.reshape(b, heads, n, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.norm_x = nn.LayerNorm(dim)
        self.norm_latent = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm_x(x)
        latents = self.norm_latent(latents)

        b, n_latent, _ = latents.shape

        q = self.to_q(latents)
        kv = torch.cat((x, latents), dim=1)
        k, v = self.to_kv(kv).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        attn = (q * scale) @ (k * scale).transpose(-2, -1)
        attn = torch.softmax(attn.float(), dim=-1).type(attn.dtype)

        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(b, n_latent, -1)

        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        max_seq_len=257,
        apply_pos_emb=False,
        num_latents_mean_pooled=0,
    ):
        super().__init__()

        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.mean_latents = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        if self.pos_emb is not None:
            n = x.shape[1]
            pos = self.pos_emb(torch.arange(n, device=x.device))
            x = x + pos

        latents = self.latents.expand(x.size(0), -1, -1)
        x = self.proj_in(x)

        if self.mean_latents is not None:
            pooled = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            latents = torch.cat((self.mean_latents(pooled), latents), dim=1)

        for attn, ff in self.layers:
            latents = latents + attn(x, latents)
            latents = latents + ff(latents)

        return self.norm_out(self.proj_out(latents))


def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    t = t.masked_fill(~mask, 0.0)

    return t.sum(dim=dim) / denom.clamp(min=1e-5)
