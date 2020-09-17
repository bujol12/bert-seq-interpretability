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
    DefaultDataCollator,
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
from utils.model import SeqClassModel

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def batch_predict(
    input_words_str_lst=None,
    model=None,
    dataset=None,
    batch_size=64,
    method="lime",
    data_collator=None,
    layer_id=None,
    head_id=None,
):
    input_cnt = len(dataset)
    input_ids = None
    if input_words_str_lst is not None:
        # input prep should be replaced with collator
        input_words_lst = [
            input_words_str.split() for input_words_str in input_words_str_lst
        ]
        # convert sentence str to lst of words

        # convert list of words to model features
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
            **dataset.convert_features_dict
        )

        # extract input ids
        input_ids = [inp_feats.input_ids for inp_feats in inp_feats_lst]
        input_cnt = len(input_ids)

    final_res = None
    model.eval()

    for batch_idx in range(0, input_cnt, batch_size):
        # get next batch
        data = {}
        if (batch_idx // 64) % 10 == 0:
            logger.info("batch_idx: " + str(batch_idx))
        if input_ids is not None:
            curr_input_ids = input_ids[batch_idx : batch_idx + batch_size]
            data["input_ids"] = torch.tensor(curr_input_ids).to(device)
        else:
            batch = data_collator.collate_batch(
                dataset[batch_idx : batch_idx + batch_size]
            )
            del batch["labels"]
            for key, val in batch.items():
                data[key] = val.to(device)

            keys_to_del = list(batch.keys())
            for k in keys_to_del:
                del batch[k]

        if method == "lime":
            numpy_res = classify_sentence(model, data)
        elif method == "model_attention":
            numpy_res = classify_sentence_get_attention(model, data, layer_id, head_id)
        elif method == "soft_attention":
            numpy_res = classify_sentence_get_soft_attention(model, data)
        else:
            numpy_res = classify_sentence(model, data)

        if final_res is not None:
            final_res = np.append(final_res, numpy_res, axis=0)
        else:
            final_res = numpy_res
        # cleanup memory before next batch
        keys_to_del = list(data.keys())
        for k in keys_to_del:
            del data[k]

        torch.cuda.empty_cache()

    return final_res


def classify_sentence_get_attention(model, data, layer_id, head_id):
    with torch.no_grad():
        res = model(**data)[-1][layer_id][:][
            head_id
        ]  # [attn layer], [layer_id], [all batch], [head_id], [all tokens]
        res_np = res.detach().cpu().numpy()
        del res
    return res_np


def classify_sentence_get_soft_attention(model, data):
    with torch.no_grad():
        res = model(**data)
        res_np = res[-1].detach().cpu().numpy()
        del res
    return res_np


def classify_sentence(model, data):
    with torch.no_grad():
        sm = torch.nn.Softmax(dim=1)
        res = model(**data)
        res_sm = sm(res[0])

        numpy_res = res_sm.detach().cpu().numpy()
        # numpy_res = res[0].detach().cpu().numpy()
        del res
        del res_sm
    return numpy_res


def classify_lime(model, dataset, train_dataset, config_dict):
    explainer = LimeTextExplainer(
        class_names=(0, 1),
        bow=False,  # try with True as well: False causes masking to be done, True means removing words
        mask_string=tokenizer.mask_token
        if not config_dict.get("lime_mask_string_use_pad", False)
        else tokenizer.pad_token,
        feature_selection="none",  # use all features
        split_expression=r"\s",
    )
    classify_sentence_partial = partial(
        batch_predict,
        model=model,
        dataset=train_dataset,
        batch_size=config_dict["per_device_eval_batch_size"],
        method="lime",
    )

    res_list = []
    for i in range(0, len(dataset)):
        if i % 50 == 0:
            logger.info("lime_sample_idx:" + str(i) + "/" + str(len(dataset)))
        exp = explainer.explain_instance(
            " ".join(dataset.examples[i].words),
            classify_sentence_partial,
            labels=(1,),
            num_samples=config_dict["lime_num_samples"],
        )
        lst = exp.as_map()[1]

        lst.sort(key=(lambda x: x[0]))
        dataset.examples[i].predictions = list(map(lambda x: x[1], lst))
    return dataset


def classify_soft_attention(model, dataset, config_dict, collator):
    preds = batch_predict(
        input_words_str_lst=None,
        model=model,
        dataset=dataset,
        method="soft_attention",
        data_collator=collator,
    )
    preds = convert_token_scores_to_words(preds, dataset)
    for i in range(len(dataset)):
        dataset.examples[i].predictions = preds[i]
    return dataset


def classify_model_attention(model, dataset, config_dict, collator):
    preds = batch_predict(
        input_words_str_lst=None,
        model=model,
        dataset=dataset,
        method="model_attention",
        data_collator=collator,
        layer_id=config_dict["attn_layer_id"],
        head_id=config_dict["attn_head_id"],
    )  # [attentions]

    preds = convert_token_scores_to_words(preds, dataset)

    for i in range(len(dataset)):
        dataset.examples[i].predictions = preds[i]
    return dataset


def convert_token_scores_to_words(result, dataset):
    res = []
    for i in range(0, len(dataset)):
        data = dataset[i]
        word_scores = np.zeros(len(dataset.examples[i].words), dtype=np.float)
        assert len(result[i]) == len(data.tokens_to_words_map)
        for j in range(0, len(result[i])):
            if data.tokens_to_words_map[j] == -1:
                continue
            word_scores[data.tokens_to_words_map[j]] += result[i][j]
        res.append(word_scores)
    return res


if __name__ == "__main__":
    if len(sys.argv) != 3:
        logger.error("Required args: [config_path] [gpu_ids]")
        exit()

    config_dict = parse_config(sys.argv[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[2])

    set_seed(config_dict["seed"])

    path = config_dict["model_path"]
    method = config_dict["method"]

    output_path = config_dict["output_file"].format(
        model_name=config_dict["model_name"],
        dataset_name=config_dict["dataset"],
        experiment_name=config_dict["experiment_name"],
        datetime=datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        method=method,
    )

    tokenizer = AutoTokenizer.from_pretrained(config_dict["model_name"],)
    config = AutoConfig.from_pretrained(path)
    model = SeqClassModel.from_pretrained(path, config=config, params_dict=config_dict)

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
        file_name=data_args.file_name
        if config_dict["is_seq_class"]
        else data_args.file_name_token,
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
    data_collator = DefaultDataCollator()

    if method == "lime":
        res = classify_lime(
            model=model,
            dataset=dataset,
            train_dataset=train_dataset,
            config_dict=config_dict,
        )
    elif method == "soft_attention":
        res = classify_soft_attention(model, dataset, config_dict, data_collator)
    elif method == "model_attention":
        res = classify_model_attention(model, dataset, config_dict, data_collator)

    if output_path is not None:
        res.write_preds_to_file(output_path, res.examples)
