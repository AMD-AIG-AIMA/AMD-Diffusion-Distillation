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

import os

def build_disc(basemodel, multiscale_D=False):
    if basemodel in ['stabilityai/stable-diffusion-2-1-base',
                    'stabilityai/stable-diffusion-xl-base-1.0']:
        from .unet_D import Discriminator
        return Discriminator(basemodel, multiscale_D)
    
    elif basemodel in ['PixArt-alpha/PixArt-Sigma-XL-2-1024-MS']:
        from .transformer_D import Transformer2DDiscriminator
        return Transformer2DDiscriminator(basemodel, multiscale_D)
    
    else:
        raise Exception('undefined base model:', basemodel)

def build_target_model(basemodel, ckpt_path=None):
    if basemodel in ['stabilityai/stable-diffusion-2-1-base',
                    'stabilityai/stable-diffusion-xl-base-1.0']:
        from diffusers import UNet2DConditionModel
        model_tag = basemodel
        if ckpt_path is not None:
            model_tag = os.path.join(ckpt_path, 'unet')
        model = UNet2DConditionModel.from_pretrained(
                model_tag, subfolder='unet')
        return model
    
    elif basemodel in ['PixArt-alpha/PixArt-Sigma-XL-2-1024-MS']:
        from diffusers import PixArtTransformer2DModel
        model_tag = basemodel
        if ckpt_path is not None:
            model_tag = os.path.join(ckpt_path, 'transformer')
        model = PixArtTransformer2DModel.from_pretrained(model_tag, subfolder='transformer')
        return model
    
    else:
        raise Exception('undefined base model:', basemodel)

def build_pipeline(basemodel, model_state_dict=None, scheduler=None):
    if basemodel in ['stabilityai/stable-diffusion-2-1-base',
                    'stabilityai/stable-diffusion-xl-base-1.0']:
        from diffusers import DiffusionPipeline
        kwargs = {'requires_safety_checker': False,
                'safety_checker': None}
        if scheduler is not None:
            kwargs['scheduler'] = scheduler
        pipe = DiffusionPipeline.from_pretrained(basemodel,
                                    **kwargs)
        if model_state_dict is not None:
            pipe.unet.load_state_dict(model_state_dict)
        return pipe
    
    elif basemodel in ['PixArt-alpha/PixArt-Sigma-XL-2-1024-MS']:
        from diffusers import PixArtSigmaPipeline
        kwargs = {}
        pipe = PixArtSigmaPipeline.from_pretrained(basemodel, **kwargs)
        if model_state_dict is not None:
            pipe.transformer.load_state_dict(model_state_dict)
        return pipe
    
    else:
        raise Exception('undefined base model:', basemodel)