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

import torch
import transformers

def build_opt(opt_type, opt_params={}):
    if opt_type in ['adamw', 'adam']:
        opt_class = torch.optim.AdamW
        opt_kwargs = {'betas': (0.9, 0.999),
                    'weight_decay': 1e-2,
                    'eps': 1e-08}
        opt_kwargs.update(opt_params)
    elif opt_type == 'adafactor':
        opt_class = transformers.optimization.Adafactor
        opt_kwargs = {'scale_parameter': False,
                    'relative_step': False,
                    'warmup_init': False}

    return opt_class, opt_kwargs       
     