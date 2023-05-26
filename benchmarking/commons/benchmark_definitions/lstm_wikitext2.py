# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from typing import Dict, Any
from pathlib import Path

from benchmarking.commons.benchmark_definitions.common import RealBenchmarkDefinition
from benchmarking.training_scripts.lstm_wikitext2.lstm_wikitext2 import (
    _config_space,
    METRIC_NAME,
    RESOURCE_ATTR,
)
from syne_tune.remote.estimators import (
    DEFAULT_GPU_INSTANCE_1GPU,
    DEFAULT_GPU_INSTANCE_4GPU,
)


def lstm_wikitext2_default_params(sagemaker_backend: bool) -> Dict[str, Any]:
    if sagemaker_backend:
        instance_type = DEFAULT_GPU_INSTANCE_1GPU
        num_workers = 8
    else:
        # For local backend, GPU cores serve different workers, so we
        # need more memory
        instance_type = DEFAULT_GPU_INSTANCE_4GPU
        num_workers = 4
    return {
        "max_resource_level": 81,
        "instance_type": instance_type,
        "num_workers": num_workers,
        "report_current_best": "False",
        "dataset_path": "./",
    }


def lstm_wikitext2_benchmark(sagemaker_backend: bool = False, **kwargs):
    params = lstm_wikitext2_default_params(sagemaker_backend)
    config_space = dict(
        _config_space,
        dataset_path=params["dataset_path"],
        epochs=params["max_resource_level"],
        report_current_best=params["report_current_best"],
    )
    _kwargs = dict(
        script=Path(__file__).parent.parent.parent
        / "training_scripts"
        / "lstm_wikitext2"
        / "lstm_wikitext2.py",
        config_space=config_space,
        max_wallclock_time=7 * 3600,
        n_workers=params["num_workers"],
        instance_type=params["instance_type"],
        metric=METRIC_NAME,
        mode="max",
        max_resource_attr="epochs",
        resource_attr=RESOURCE_ATTR,
        framework="PyTorch",
    )
    _kwargs.update(kwargs)
    return RealBenchmarkDefinition(**_kwargs)


# Support for cost models:
#
# from benchmarking.utils import get_cost_model_for_batch_size
# from benchmarking.training_scripts.lstm_wikitext2.lstm_wikitext2 import (
#     BATCH_SIZE_LOWER,
#     BATCH_SIZE_UPPER,
#     BATCH_SIZE_KEY,
# )
# cost_model = get_cost_model_for_batch_size(
#     cost_model_type="quadratic_spline",
#     batch_size_key = BATCH_SIZE_KEY,
#     batch_size_range = (BATCH_SIZE_LOWER, BATCH_SIZE_UPPER),
# )
