import os
import argparse
import torch
from PIL import Image
import numpy as np
import open_clip
from tqdm import tqdm
import glob


parser = argparse.ArgumentParser()
parser.add_argument(
        "--imgfolder",
        type=str,
        required=True,
    )
parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
    )

opt = parser.parse_args()


model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
tokenizer = open_clip.get_tokenizer('ViT-g-14')

model.cuda()

with open(opt.inputfile, 'r') as f:
    lines = f.readlines()
    id_list = []
    for l in lines:
        caption_id, caption = l.strip().split('----')
        id_list.append(caption)

score_sum = 0
count = 0

for img_path in tqdm(glob.glob(os.path.join(opt.imgfolder, '*.png'))):
    img_id = int(os.path.basename(img_path)[:-4])
    prompt = id_list[img_id]
    text = tokenizer([prompt]).cuda()
    try:
        image = preprocess(Image.open(img_path)).unsqueeze(0).cuda()
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            score = image_features @ text_features.T
        score_sum += score.cpu().numpy()
        count += 1
    except Exception as e:
        print(img_path)
        print(e)

print('got %d valid images:'%count)
print('clip score: %f' % (score_sum/count))
    