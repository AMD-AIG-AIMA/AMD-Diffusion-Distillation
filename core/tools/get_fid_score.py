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
import argparse
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
import numpy as np
from tqdm import tqdm
import glob
from pycocotools.coco import COCO

parser = argparse.ArgumentParser()
parser.add_argument(
        "--cocoroot",
        type=str,
        required=True,
    )
parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
    )

parser.add_argument(
        "--imgfolder",
        type=str,
        required=True,
    )
opt = parser.parse_args()

cocoann_path = os.path.join(opt.cocoroot, 'annotations/captions_val2017.json')
coco = COCO(cocoann_path)

with open(opt.inputfile, 'r') as f:
    lines = f.readlines()
    id_list = []
    for l in lines:
        caption_id, caption = l.strip().split('----')
        id_list.append((int(caption_id), caption))

fid = FrechetInceptionDistance(feature=2048)
fid.cuda()

num_imgs = len(glob.glob(os.path.join(opt.imgfolder, '*.png')))
print(f'found {num_imgs} images')
count = 0
for img_path in tqdm(glob.glob(os.path.join(opt.fakefolder, '*.png'))):
    try:
        img = Image.open(img_path)
        img = img.resize((299, 299), Image.BICUBIC)
        img = torch.tensor(np.array(img)).cuda()
        img = img.permute((2, 0, 1)).unsqueeze(0)
        fid.update(img, real=False)
    except Exception as e:
        print(f'got error processing {img_path}')
        print(e)

count = 0
for cur_item in tqdm(id_list):
    cur_id = cur_item[0]
    img_id = coco.loadAnns(cur_id)[0]['image_id']
    img_fname = coco.loadImgs(img_id)[0]['file_name']
    img_path = os.path.join(opt.cocoroot, 'val2017', img_fname)
    img = Image.open(img_path).convert('RGB')
    img = img.resize((299, 299), Image.BICUBIC)
    img = torch.tensor(np.array(img)).cuda()
    img = img.permute((2, 0, 1)).unsqueeze(0)
    fid.update(img, real=True)



print('fid score: ', fid.compute())
