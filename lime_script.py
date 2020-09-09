import dataclasses
import datetime
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import uuid
from functools import partial

import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    GlueDataset,
    AutoModel,
    AutoModelForTokenClassification,
    BertForTokenClassification,
)
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from lime.lime_text import LimeTextExplainer
from utils.tsv_dataset import (
    convert_examples_to_features,
    InputExample,
    TSVClassificationDataset,
)
import uuid

import torch
from torch.nn import CrossEntropyLoss, MSELoss

from utils.tsv_dataset import TSVClassificationDataset, Split, get_labels
from utils.arguments import (
    datasets,
    DataTrainingArguments,
    ModelArguments,
    parse_config,
)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def classify_sentence(input_words_str_lst, model, train_dataset, batch_size=64):
    input_words_lst = [
        input_words_str.split() for input_words_str in input_words_str_lst
    ]
    inp_feats_lst = convert_examples_to_features(
        [
            InputExample(
                guid=uuid.uuid4().hex,
                words=input_words,
                labels=[model.config.id2label[0]]
                * len(input_words),  # fill with dummy label
            )
            for input_words in input_words_lst
        ],
        **train_dataset.convert_features_dict
    )

    input_ids = [inp_feats.input_ids for inp_feats in inp_feats_lst]
    sm = torch.nn.Softmax(dim=1)
    final_res = None
    for batch_idx in range(0, len(input_ids), batch_size):
        curr_input_ids = input_ids[batch_idx : batch_idx + batch_size]
        data = {}
        data["input_ids"] = torch.tensor(curr_input_ids).to(device)

        res = model(**data)
        res_sm = sm(res[0])

        numpy_res = res_sm.detach().cpu().numpy()

        if final_res is not None:
            final_res = np.append(final_res, numpy_res, axis=0)
        else:
            final_res = numpy_res
        # cleanup memory before next batch
        del data["input_ids"]
        del res
        del res_sm
        torch.cuda.empty_cache()

    return final_res


if __name__ == "__main__":
    if len(sys.argv) != 3:
        logger.error("Required args: [config_path] [gpu_ids]")
        exit()

    config_dict = parse_config(sys.argv[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[2])

    output_path = config_dict["output_file"].format(
        model_name=config_dict["model_name"],
        dataset_name=config_dict["dataset"],
        experiment_name=config_dict["experiment_name"],
        datetime=datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
    )

    set_seed(config_dict["seed"])

    path = config_dict["model_path"]

    tokenizer = AutoTokenizer.from_pretrained(config_dict["model_name"],)

    model = AutoModelForSequenceClassification.from_pretrained(path)
    labels = [model.config.id2label[0], model.config.id2label[1]]

    model_args = ModelArguments(model_name_or_path=config_dict["model_name"])

    data_args = datasets[config_dict["dataset"]]

    data_config = dict(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        labels=labels,
        model_type=model.config.model_type,
        max_seq_length=data_args.max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
        file_name=data_args.file_name,
        make_all_labels_equal_max=config_dict["make_all_labels_equal_max"],
        default_label=config_dict["test_label_dummy"],
        is_seq_class=config_dict["is_seq_class"],
        lowercase=config_dict["lowercase"],
    )
    train_dataset = TSVClassificationDataset(mode=Split.train, **data_config)
    if config_dict["dataset_split"] == "train":
        dataset = train_dataset
    elif config_dict["dataset_split"] == "dev":
        dataset = TSVClassificationDataset(mode=Split.dev, **data_config)
    elif config_dict["dataset_split"] == "test":
        dataset = TSVClassificationDataset(mode=Split.test, **data_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    explainer = LimeTextExplainer(
        class_names=(0, 1),
        bow=False,  # try with True as well: False causes masking to be done, True means removing words
        mask_string=tokenizer.mask_token,
        feature_selection="none",  # use all features
        split_expression=r"\s",
    )
    classify_sentence_partial = partial(
        classify_sentence,
        model=model,
        train_dataset=train_dataset,
        batch_size=config_dict["per_device_eval_batch_size"],
    )
    res_list = []
    for i in range(0, len(dataset)):
        if i % 50 == 0:
            logger.info(i)
        exp = explainer.explain_instance(
            " ".join(dataset.examples[i].words),
            classify_sentence_partial,
            labels=(1,),
            num_samples=config_dict["lime_num_samples"],
        )
        lst = exp.as_map()[1]

        lst.sort(key=(lambda x: x[0]))
        dataset.examples[i].predictions = list(map(lambda x: x[1], lst))
    dataset.write_preds_to_file(output_path, dataset.examples)
