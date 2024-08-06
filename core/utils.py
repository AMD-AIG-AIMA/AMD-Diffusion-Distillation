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

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def predicted_origin(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "v_prediction":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0

def change_device(src_dict, device):
    """
    input, a dict of tensors, or dict, change the device recursively.
    """
    assert isinstance(src_dict, dict)
    for k in src_dict:
        if isinstance(src_dict[k], torch.Tensor):
            src_dict[k] = src_dict[k].to(device)
        elif isinstance(src_dict[k], dict):
            change_device(src_dict[k], device)


def concat_dict(src_dict):
    """
        concat a dict of tensors, or dict by dim=0 recursively
    """
    assert isinstance(src_dict, dict)
    res_dict = {}
    for k in src_dict:
        if isinstance(src_dict[k], torch.Tensor):
            res_dict[k] = torch.concat([src_dict[k], src_dict[k]], dim=0)
        elif isinstance(src_dict[k], dict):
            res_dict[k] = concat_dict(src_dict[k])
    return res_dict
    

