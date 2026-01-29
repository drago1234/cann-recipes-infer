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

import os
import time
import argparse
import logging
from runner_gpt_oss import GptOssRunner
from executor.utils import read_yaml
from executor.utils.data_utils import generate_prompt
from executor.utils.common_utils import check_common_parallel_settings
from models.model_setting import update_vars

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(description="llm run parameters")
    parser.add_argument('--yaml_file_path', type=str, help="inference configurations")
    parser_args = parser.parse_args()
    return parser_args


def run_gpt_oss(runner_settings):
    preset_prompts, _ = generate_prompt(runner_settings)
    model_runner = GptOssRunner(runner_settings)
    model_runner.init_model()
    # generate perf data
    model_runner.model_generate(preset_prompts)


if __name__ == "__main__":
    args = parse_args()
    yaml_file_path = args.yaml_file_path
    runner_settings = read_yaml(yaml_file_path)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    check_common_parallel_settings(world_size, runner_settings)
    update_vars(world_size, runner_settings)
    logging.info(f"runner_settings is: {runner_settings}")
    run_gpt_oss(runner_settings)
    logging.info("model run success")
