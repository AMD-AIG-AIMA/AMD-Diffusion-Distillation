# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from diffusers import UNet2DConditionModel
import torch.nn as nn
import torch
from typing import Union, Optional, Dict

#based_on diffusers 0.29.2
def unet_forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        is_multiscale = False,
    ):        

    # 0. center input if necessary
    if self.config.center_input_sample:
        sample = 2 * sample - 1.0

    # 1. time
    t_emb = self.get_time_embed(sample=sample, timestep=timestep)
    emb = self.time_embedding(t_emb, None)
    aug_emb = None

    aug_emb = self.get_aug_embed(
        emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )

    emb = emb + aug_emb if aug_emb is not None else emb

    if self.time_embed_act is not None:
        emb = self.time_embed_act(emb)

    encoder_hidden_states = self.process_encoder_hidden_states(
        encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )

    # 2. pre-process
    sample = self.conv_in(sample)

    feat_list = []

    down_block_res_samples = (sample,)
    for blk_ind, downsample_block in enumerate(self.down_blocks):
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            # For t2i-adapter CrossAttnDownBlock2D
            additional_residuals = {}

            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=None,
                cross_attention_kwargs=None,
                encoder_attention_mask=None,
                **additional_residuals,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
        down_block_res_samples += res_samples
        if is_multiscale and blk_ind <= 1:
            feat_list.append(sample)

    # 4. mid
    if self.mid_block is not None:
        if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=None,
                cross_attention_kwargs=None,
                encoder_attention_mask=None,
            )
        else:
            sample = self.mid_block(sample, emb)


    feat_list.append(sample)
    return feat_list



class Discriminator(nn.Module):
    def __init__(self, pretrained_path, is_multiscale=False):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_path, subfolder="unet")
        self.unet.forward = unet_forward
        self.unet.up_blocks = None
        self.unet.conv_out = None
        self.unet.conv_norm_out = None
        self.is_multiscale = is_multiscale

        self.heads = []
        if is_multiscale:
            channel_list = [320, 640, 1280]
        else:
            channel_list = [1280]

        for feat_c in channel_list:
            self.heads.append(nn.Sequential(nn.GroupNorm(32, feat_c, eps=1e-05, affine=True),
                                        nn.Conv2d(feat_c, feat_c//4, 4, 2, 2),
                                        nn.SiLU(),
                                        nn.Conv2d(feat_c//4,1,1,1,0)
                                            ))
        self.heads = nn.ModuleList(self.heads)

        
    def forward(self, latent, timesteps, encoder_hidden_states,
                added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None):
        feat_list = self.unet.forward(self.unet, latent, timesteps, encoder_hidden_states, 
                                        is_multiscale=self.is_multiscale,
                                        added_cond_kwargs=added_cond_kwargs)

        res_list = []
        for cur_feat, cur_head in zip(feat_list, self.heads):
            cur_out = cur_head(cur_feat)
            res_list.append(cur_out.reshape(cur_out.shape[0], -1))
        
        concat_res = torch.cat(res_list, dim=1)
        
        return concat_res

    def enable_gradient_checkpointing(self):
        self.unet.enable_gradient_checkpointing()

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)