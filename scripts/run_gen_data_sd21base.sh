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

PROMPT_PATH='data/sample_prompts.txt'
OUT_FOLDER='generated_data_sdv21'

available_gpus=(0 1 2 3 4 5 6 7)
for gpu in "${available_gpus[@]}"; do
CUDA_VISIBLE_DEVICES=${gpu} python core/tools/gen_synthetic_data.py --prompt_path $PROMPT_PATH --root_folder $OUT_FOLDER &
done
wait

# after generating data, run this to create a summary file
python core/tools/create_summary.py --root_folder $OUT_FOLDER