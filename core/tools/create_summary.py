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

import pickle
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='args for creating summary')

parser.add_argument(
        '--root_folder',
        type=str,
        default='../generated_data',
        help=(
            'root folder for generated data'
        ),
    )

parser.add_argument(
        '--summary_file_name',
        type=str,
        default='summary.pkl',
        help=(
            'summary file name'
        ),
    )

parser.add_argument(
        '--purge_missing_data',
        action='store_true',
        help=(
            'whether purge the missing pairs'
        ),
    )

args = parser.parse_args()


latent_folder = 'latents'
txt_emb_folder = 'txt_embs'
noise_folder = 'noises'

res_list = []
for fname in tqdm(os.listdir(os.path.join(args.root_folder, latent_folder))):
    latent_rel_path = os.path.join(latent_folder, fname)
    txt_emb_rel_path = os.path.join(txt_emb_folder, fname)
    noise_rel_path = os.path.join(noise_folder, fname)

    if os.path.exists(os.path.join(args.root_folder, latent_rel_path)) and \
        os.path.exists(os.path.join(args.root_folder, txt_emb_rel_path)) and \
        os.path.exists(os.path.join(args.root_folder, noise_rel_path)):
        res_list.append((latent_rel_path, 
                        noise_rel_path, txt_emb_rel_path))
    else:
        print('skipped for missing data for ', fname)
        if args.purge_missing_data:
            if os.path.exists(os.path.join(args.root_folder, latent_rel_path)):
                os.remove(os.path.join(args.root_folder, latent_rel_path))
            if os.path.exists(os.path.join(args.root_folder, txt_emb_rel_path)):
                os.remove(os.path.join(args.root_folder, txt_emb_rel_path))
            if os.path.exists(os.path.join(args.root_folder, noise_rel_path)):
                os.remove(os.path.join(args.root_folder, noise_rel_path))
            print('removed pairs for ', fname)

with open(os.path.join(args.root_folder, args.summary_file_name), 'wb') as f:
    pickle.dump(res_list, f)

print(f'done. found {len(res_list)} pairs')
