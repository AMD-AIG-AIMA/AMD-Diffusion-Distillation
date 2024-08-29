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


import argparse, os
import torch
from diffusers import (DDPMScheduler,
                       DiffusionPipeline,
                       PixArtSigmaPipeline)
import torch.nn as nn

parser = argparse.ArgumentParser()


parser.add_argument("--model",
    type=str,
    required=True
)

parser.add_argument("--ckpt_path",
    type=str,
    required=True
)

parser.add_argument("--out_folder",
    type=str,
    required=True
)

parser.add_argument("--prompt_path",
    type=str,
    required=True
)

parser.add_argument("--seed",
    type=int,
    default=0,
)

args = parser.parse_args()

if args.model == 'sd21base':
    scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler")
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base",
                                            scheduler=scheduler,
                                            requires_safety_checker=False,
                                            safety_checker=None)
    unet_state_dict = torch.load(args.ckpt_path)
    pipe.unet.load_state_dict(unet_state_dict)
    pipe_kwargs = {'num_inference_steps': 1,
                    'guidance_scale': 0,
                    'timesteps': [999]}
elif args.model == 'pixart-sigma':
    pipe = PixArtSigmaPipeline.from_pretrained('PixArt-alpha/PixArt-Sigma-XL-2-1024-MS')
    transformer_state_dict = torch.load(args.ckpt_path)
    pipe.transformer.load_state_dict(transformer_state_dict)
    pipe_kwargs = {'num_inference_steps': 1,
                'guidance_scale': 0,
                'timesteps': [400]}

else:
    raise Exception('wrong model, ', args.model)

pipe = pipe.to("cuda")

os.makedirs(args.out_folder, exist_ok=True)
with open(args.prompt_path, 'r') as f:
    lines = f.readlines()
    prompt_list = []
    for l in lines:
        prompt_list.append(l.strip().split('----')[1])

generator = torch.manual_seed(args.seed)
print('seed: ', args.seed)

with torch.no_grad():
    num_samples = len(prompt_list)
    for id_ind, cur_caption in enumerate(prompt_list):
        outpath = os.path.join(args.out_folder, '%d.png'%id_ind)
        image = pipe(prompt=cur_caption, generator=generator,
                        **pipe_kwargs).images[0]
        image.save(outpath)