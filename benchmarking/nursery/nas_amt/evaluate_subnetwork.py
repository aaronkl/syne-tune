#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE."""
import json

# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
import os
import time
import logging
import sys

from dataclasses import dataclass, field

import numpy as np
import torch
import datasets

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    HfArgumentParser,
    AutoConfig,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from evaluate import load

from estimate_efficency import compute_parameters
from task_data import TASKINFO
from mask import mask_bert, mask_gpt
from hf_args import DataTrainingArguments, ModelArguments
from load_glue_datasets import load_glue_datasets


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class SearchArguments:
    """
    Arguments to define the search
    """

    num_layers: int = field()
    num_heads: int = field()
    num_units: int = field()
    checkpoint_dir_model: str = field(
        metadata={"help": ""}, default=os.environ["SM_CHANNEL_MODEL"]
    )


def main():
    start_time = time.time()
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, SearchArguments)
    )

    (
        model_args,
        data_args,
        training_args,
        search_args,
    ) = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)

    # Set seed before initializing model.

    if int(training_args.seed) == -1:
        training_args.seed = np.random.randint(2**32 - 1)
    print(training_args.seed)
    set_seed(training_args.seed)

    model_type = model_args.model_name_or_path

    # Labels
    is_regression = data_args.task_name == "stsb"

    metric = load("glue", data_args.task_name)

    _, eval_dataloader, test_dataloader = load_glue_datasets(
        training_args=training_args, model_args=model_args, data_args=data_args
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        search_args.checkpoint_dir_model
    )

    config = model.config
    if model_type.startswith("bert"):
        attention_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        attention_head_size = int(attention_size / num_attention_heads)
        num_layers = config.num_hidden_layers
        intermediate_size = config.intermediate_size
        mask = mask_bert

        n_params_emb = sum(
            p.numel() for p in model.bert.embeddings.parameters() if p.requires_grad
        )
        n_params_pooler = sum(
            p.numel() for p in model.bert.pooler.parameters() if p.requires_grad
        )
        n_params_classifier = sum(
            p.numel() for p in model.classifier.parameters() if p.requires_grad
        )
        n_params_classifier += n_params_pooler

    elif model_type.startswith("gpt2"):
        model.config.pad_token_id = model.config.eos_token_id
        mask = mask_gpt

        num_attention_heads = config.n_head
        attention_size = config.hidden_size
        attention_head_size = int(attention_size / num_attention_heads)
        num_layers = config.n_layer
        intermediate_size = (
            config.n_inner if config.n_inner is not None else 4 * config.hidden_size
        )
        wte = sum(
            p.numel() for p in model.transformer.wte.parameters() if p.requires_grad
        )
        wpe = sum(
            p.numel() for p in model.transformer.wpe.parameters() if p.requires_grad
        )
        n_params_emb = wte + wpe
        n_params_classifier = sum(
            p.numel() for p in model.score.parameters() if p.requires_grad
        )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    metric_name = TASKINFO[data_args.task_name]["metric"]

    head_mask = torch.ones((num_layers, num_attention_heads))
    neuron_mask = torch.ones((num_layers, intermediate_size))

    head_mask[search_args.num_layers :, :] = 0
    head_mask[: search_args.num_layers :, search_args.num_heads :] = 0
    neuron_mask[search_args.num_layers :, :] = 0
    neuron_mask[: search_args.num_layers :, search_args.num_units :] = 0

    head_mask = head_mask.to(device)
    neuron_mask = neuron_mask.to(device)
    n_params_model = compute_parameters(
        dmodel=attention_size,
        dhead=attention_head_size,
        num_heads_per_layer=head_mask.sum(dim=1),
        num_neurons_per_layer=neuron_mask.sum(dim=1),
    )
    n_params = n_params_emb + n_params_model + n_params_classifier

    handles = mask(model, neuron_mask, head_mask)

    model.eval()

    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(head_mask=head_mask, **batch)

        logits = outputs.logits
        predictions = (
            torch.squeeze(logits) if is_regression else torch.argmax(logits, dim=-1)
        )

        metric.add_batch(predictions=predictions, references=batch["labels"])

    eval_metric = metric.compute()

    print(f"number of parameters: {n_params}")
    print(f"validation score: {eval_metric[metric_name]}")

    logger.info("validation score: {:.4f}\n".format(eval_metric[metric_name]))

    logger.info("number of parameters: {:.4f}\n".format(n_params))


if __name__ == "__main__":
    main()
