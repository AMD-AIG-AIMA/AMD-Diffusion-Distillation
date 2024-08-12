

import argparse, os
import torch
import random
from datetime import datetime
from random import shuffle
import numpy as np
import time
from diffusers import (UNet2DConditionModel, 
                       DDPMScheduler,
                       DiffusionPipeline)
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

parser.add_argument("--latent_code",
    type=str,
    required=True
)

parser.add_argument("--prompt_path",
    type=str,
    required=True
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
else:
    raise Exception('wrong model, ', args.model)

pipe = pipe.to("cuda")

os.makedirs(args.out_folder, exist_ok=True)
with open(args.prompt_path, 'r') as f:
    lines = f.readlines()
    prompt_list = []
    for l in lines:
        prompt_list.append(l.strip().split('----')[1])

start_code = torch.load(args.latent_code).cuda()

with torch.no_grad():
    num_samples = len(prompt_list)
    for id_ind, cur_caption in enumerate(prompt_list):
        outpath = os.path.join(args.out_folder, '%d.png'%id_ind)
        image = pipe(prompt=cur_caption,
                        latents=start_code,
                        **pipe_kwargs).images[0]
        image.save(outpath)