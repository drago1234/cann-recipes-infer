# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

from executor.utils import update_settings
from executor.utils.common_utils import check_common_parallel_settings


def check_vars(world_size, runner_settings):
    check_common_parallel_settings(world_size, runner_settings)
    attn_tp_size = runner_settings.get("parallel_config").get("attn_tp_size", 1)
    moe_tp_size = runner_settings.get("parallel_config").get("moe_tp_size", 1)
    lmhead_tp_size = runner_settings.get("parallel_config").get("lmhead_tp_size", 1)
    if not world_size == attn_tp_size == moe_tp_size == lmhead_tp_size:
        raise ValueError("The values of world_size, attn_tp_size, moe_tp_size and lmhead_tp_size must be equal.")


def update_vars(world_size, runner_settings):
    input_max_len = runner_settings.get("data_config").get("input_max_len", 32)
    max_new_tokens = runner_settings.get("data_config").get("max_new_tokens", 32)
    max_len_bound = max_new_tokens + input_max_len
    runner_settings = update_settings(runner_settings, "data_config", "max_position_embeddings", max_len_bound)