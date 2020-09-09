import dataclasses
import collections
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import uuid
import json

try:
    import ConfigParser as configparser
except:
    import configparser


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    use_fast: bool = field(
        default=False, metadata={"help": "Set this flag to use fast tokenization."}
    )
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    name: str = field(metadata={"help": "The name of the dataset used"})

    data_dir: str = field(
        metadata={
            "help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."
        }
    )
    positive_label: str = field(metadata={"help": "Positive label - labelled as 1."})
    labels: Optional[str] = field(
        metadata={"help": "Path to a training file from which to fetch the labels."}
    )
    file_name: str = field(metadata={"help": "Filename to be used to read data in."})
    file_name_token: str = field(metadata={"help": "filename of the token-level files"})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )


datasets = dict(
    fce_v1=DataTrainingArguments(
        name="fce_v1",
        data_dir="data/fce_v1_tsv",
        labels="data/fce_v1_tsv/fce-public.train.original.tsv_sentencelevel",
        file_name="fce-public.{mode}.original.tsv_sentencelevel",
        file_name_token="fce-public.{mode}.original.tsv_sentencelevel",
        positive_label="i",
    ),
    conll10=DataTrainingArguments(
        name="conll10",
        data_dir="data/conll10",
        labels="data/conll10/conll10_task2_rev2.cue.train.tsv_sentencelevel",
        file_name="conll10_task2_rev2.cue.{mode}.tsv_sentencelevel",
        file_name_token="conll10_task2_rev2.cue.{mode}.tsv",
        positive_label="C",
    ),
    sst2_pos=DataTrainingArguments(
        name="sst2_pos",
        data_dir="data/SST_labelling",
        labels="data/SST_labelling/stanford_sentiment.train.sentences.positive.tsv",
        file_name="stanford_sentiment.{mode}.sentences.positive.tsv",
        file_name_token="stanford_sentiment.{mode}.tokens.positive.tsv",
        positive_label="P",
    ),
    sst2_neg=DataTrainingArguments(
        name="sst2_neg",
        data_dir="data/SST_labelling",
        labels="data/SST_labelling/stanford_sentiment.train.sentences.negative.tsv",
        file_name="stanford_sentiment.{mode}.sentences.negative.tsv",
        file_name_token="stanford_sentiment.{mode}.tokens.negative.tsv",
        positive_label="N",
    ),
    semeval_2013_twitter_pos=DataTrainingArguments(
        name="semeval_2013_twitter_pos",
        data_dir="data/semeval15t10/semeval_2013_twitter",
        labels="data/semeval15t10/semeval_2013_twitter/semeval15t10.sentences.positive.train2013.tsv",
        file_name="semeval15t10.sentences.positive.{mode}2013.tsv",
        file_name_token="semeval15t10.tokens.positive.{mode}2013.tsv",
        positive_label="P",
    ),
    semeval_2013_twitter_neg=DataTrainingArguments(
        name="semeval_2013_twitter_neg",
        data_dir="data/semeval15t10/semeval_2013_twitter",
        labels="data/semeval15t10/semeval_2013_twitter/semeval15t10.sentences.negative.train2013.tsv",
        file_name="semeval15t10.sentences.negative.{mode}2013.tsv",
        file_name_token="semeval15t10.tokens.negative.{mode}2013.tsv",
        positive_label="N",
    ),
    # sst2_pos_neg=DataTrainingArguments(
    #    name="sst2_pos_neg",
    #    data_dir="data/SST-2/",
    #    labels="data/SST-2/train_token.tsv_sentencelevel",
    #    file_name="{mode}_token.tsv_sentencelevel",
    # ),
)


def is_float(value):
    """
    Check in value is of type float()
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def parse_config(config_path, config_section="config"):
    config_parser = configparser.SafeConfigParser(allow_no_value=True)
    config_parser.read(config_path)
    config = collections.OrderedDict()

    for key, value in config_parser.items(config_section):
        if value is None or len(value.strip()) == 0:
            config[key] = None
        elif value.lower() in ["true", "false"]:
            config[key] = config_parser.getboolean(config_section, key)
        elif value.isdigit():
            config[key] = config_parser.getint(config_section, key)
        elif is_float(value):
            config[key] = config_parser.getfloat(config_section, key)
        else:
            config[key] = config_parser.get(config_section, key)
    return config
