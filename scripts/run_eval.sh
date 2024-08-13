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


MODEL='sd21base'
CKPT_PATH='ckpts/model.bin'
OUT_FODLER='gen_images_sd21base'
PROMPT_PATH='data/coco_prompt_list.txt'
COCO_ROOT='../../data/coco'


python core/tools/gen_samples.py --model ${MODEL} \
                                 --ckpt_path ${CKPT_PATH} \
                                 --out_folder ${OUT_FODLER} \
                                 --prompt_path ${PROMPT_PATH}

python core/tools/get_fid_score.py --cocoroot ${COCO_ROOT} \
                                --inputfile ${PROMPT_PATH} \
                                --imgfolder ${OUT_FODLER}


python core/tools/get_clip_score.py --imgfolder ${OUT_FODLER} \
                                            --inputfile ${PROMPT_PATH}
