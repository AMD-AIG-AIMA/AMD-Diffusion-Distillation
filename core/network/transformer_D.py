from typing import Callable
from diffusers import PixArtTransformer2DModel
import torch.nn as nn
import torch
from torch.nn.utils.spectral_norm import SpectralNorm
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x) + x) / np.sqrt(2)


class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 8, eps: float = 1e-5):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()

        # Reshape batch into groups.
        G = np.ceil(x.size(0)/self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))

        # Calculate stats.
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(shape)


def make_block(channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        SpectralConv1d(
            channels,
            channels,
            kernel_size = kernel_size,
            padding = kernel_size//2,
            padding_mode = 'circular',
        ),
        BatchNormLocal(channels),
        nn.LeakyReLU(0.2, True),
    )


# Adapted from https://github.com/autonomousvision/stylegan-t/blob/main/networks/discriminator.py
class DiscHead(nn.Module):
    def __init__(self, channels: int, c_dim: int, cmap_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.c_dim = c_dim
        self.cmap_dim = cmap_dim

        self.main = nn.Sequential(
            make_block(channels, kernel_size=1),
            ResidualBlock(make_block(channels, kernel_size=9))
        )

        if self.c_dim > 0:
            self.cmapper = nn.Linear(self.c_dim, cmap_dim)
            self.cls = SpectralConv1d(channels, cmap_dim, kernel_size=1, padding=0)
        else:
            self.cls = SpectralConv1d(channels, 1, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.main(x)
        out = self.cls(h)

        if self.c_dim > 0:
            cmap = self.cmapper(c).unsqueeze(-1)
            out = (out * cmap).sum(1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return out
    

# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/pixart_transformer_2d.py
# Modified to return intermediate features
def transformer_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        added_cond_kwargs,
        block_hooks,
    ):
        encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch_size = hidden_states.shape[0]
        hidden_states = self.pos_embed(hidden_states)
        embedded_timestep = None

        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        # 2. Blocks
        feat_list = []
        for blk_ind, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
            )
            if blk_ind in block_hooks:
                feat_list.append(hidden_states)

        return feat_list


class Transformer2DDiscriminator(nn.Module):
    def __init__(self, pretrained_path, is_multiscale=False):
        super().__init__()
        self.transformer = PixArtTransformer2DModel.from_pretrained(pretrained_path, subfolder="transformer")
        self.transformer.forward = transformer_forward
        self.block_hooks = set([2,8,14,20,27]) if is_multiscale else set([len(self.transformer.transformer_blocks) - 1])

        # use only a part of the DiT as the discriminator backbone
        self.transformer.norm_out = None
        self.transformer.proj_out = None
        self.transformer.scale_shift_table = None

        heads = []
        for i in range(len(self.block_hooks)):
            heads.append(DiscHead(self.transformer.inner_dim, 0, 0))
        self.heads = nn.ModuleList(heads)

    @property
    def model(self):
        return self.transformer
        
    def forward(self, latent, timesteps, encoder_hidden_states, encoder_attention_mask, added_cond_kwargs):
        feat_list = self.transformer.forward(
            self.transformer, 
            latent, 
            timestep=timesteps, 
            encoder_hidden_states=encoder_hidden_states, 
            encoder_attention_mask=encoder_attention_mask, 
            added_cond_kwargs=added_cond_kwargs, 
            block_hooks=self.block_hooks
        )

        res_list = []
        for feat, head in zip(feat_list, self.heads):
            res_list.append(head(feat.transpose(1,2), None).reshape(feat.shape[0], -1))
        
        concat_res = torch.cat(res_list, dim=1)
        
        return concat_res

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)