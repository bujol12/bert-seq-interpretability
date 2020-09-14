import dataclasses
import logging
import os
import sys
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import uuid
import json

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

from torch.nn import CrossEntropyLoss, MSELoss

from utils.tsv_dataset import (
    TSVClassificationDataset,
    Split,
    get_labels,
    compute_seq_classification_metrics,
    MaskedDataCollator,
)
from utils.arguments import (
    datasets,
    DataTrainingArguments,
    ModelArguments,
    parse_config,
)
from utils.model import SeqClassModel

logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        logger.error("Required args: [config_path] [gpu_ids]")
        exit()

    config_dict = parse_config(sys.argv[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[2])

    model_args = ModelArguments(model_name_or_path=config_dict["model_name"])

    data_args = datasets[config_dict["dataset"]]

    output_dir = config_dict["output_dir"].format(
        model_name=model_args.model_name_or_path,
        dataset_name=data_args.name,
        experiment_name=config_dict["experiment_name"],
        datetime=datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
    )

    set_seed(config_dict["seed"])

    labels = get_labels(data_args.labels)
    # ensure positive label has index == 1
    idx_pos = min(
        [i for i, val in enumerate(labels) if val == data_args.positive_label]
    )
    labels[idx_pos], labels[1] = labels[1], labels[idx_pos]

    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}

    config = AutoConfig.from_pretrained(
        config_dict["model_name"],
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        output_hidden_states=True,
        output_attentions=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config_dict["model_name"],)
    model_raw = SeqClassModel(params_dict=config_dict, model_config=config)

    data_config = dict(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        labels=labels,
        model_type=config.model_type,
        max_seq_length=config_dict["max_seq_length"],
        overwrite_cache=data_args.overwrite_cache,
        file_name=data_args.file_name,
        make_all_labels_equal_max=config_dict["make_all_labels_equal_max"],
        default_label=config_dict["test_label_dummy"],
        is_seq_class=config_dict["is_seq_class"],
        lowercase=config_dict["lowercase"],
    )
    # Get datasets
    train_dataset = TSVClassificationDataset(mode=Split.train, **data_config)
    dev_dataset = TSVClassificationDataset(mode=Split.dev, **data_config)
    test_dataset = TSVClassificationDataset(mode=Split.test, **data_config)

    token_input_filepath = config_dict.get("token_input_filepath", None)
    if token_input_filepath is not None:
        data_config_token = dict(
            data_dir=token_input_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=config_dict["max_seq_length"],
            overwrite_cache=data_args.overwrite_cache,
            make_all_labels_equal_max=False,
            default_label=config_dict["test_label_dummy"],
            is_seq_class=False,
            lowercase=config_dict["lowercase"],
        )
        train_dataset_token_labels = TSVClassificationDataset(
            mode=Split.train,
            file_name=token_input_filepath.format(mode="train"),
            **data_config_token
        )
        # dev_dataset_token_labels = TSVClassificationDataset(
        #    mode=Split.dev,
        #    file_name=token_input_filepath.format(mode="dev"),
        #    **data_config_token
        # )
        # test_dataset_token_labels = TSVClassificationDataset(
        #    mode=Split.test,
        #    file_name=token_input_filepath.format(mode="test"),
        #    **data_config_token
        # )

        train_dataset.set_token_scores_from_other_dataset(train_dataset_token_labels)
        # dev_dataset.set_token_scores_from_other_dataset(dev_dataset_token_labels)
        # test_dataset.set_token_scores_from_other_dataset(test_dataset_token_labels)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=config_dict["per_device_train_batch_size"],
        per_device_eval_batch_size=config_dict["per_device_eval_batch_size"],
        num_train_epochs=config_dict["num_train_epochs"],
        warmup_steps=int(
            config_dict["warmup_ratio"]
            * (
                len(train_dataset)
                // config_dict["gradient_accumulation_steps"]
                * config_dict["num_train_epochs"]
            )
        ),
        gradient_accumulation_steps=config_dict["gradient_accumulation_steps"],
        learning_rate=config_dict["learning_rate"],  # as in roberta paper
        weight_decay=config_dict["weight_decay"],  ## as in roberta paper
        seed=config_dict["seed"],
        adam_epsilon=config_dict["adam_epsilon"],
        logging_steps=config_dict["logging_steps"],
        logging_first_step=True,
        logging_dir=output_dir + "/log",
        save_steps=config_dict["logging_steps"],
        evaluate_during_training=True,
    )

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    logger.info("Training/evaluation parameters %s", training_args)

    model = model_raw
    collator = MaskedDataCollator(
        tokenizer=tokenizer,
        do_mask=config_dict["do_mask_words"],
        mask_prob=config_dict["mask_prob"],
    )
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_seq_classification_metrics,
        data_collator=collator,
    )

    # Training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(output_dir)

    # Evaluate Each Checkpoints
    dev_results = {}
    test_results = {}
    checkpoints_list = trainer._sorted_checkpoints()

    logger.info("Saved Checkpoints:")
    logger.info(str(checkpoints_list))

    cnt = 0
    max_dev_f1 = -0.1
    max_f1_checkpoint_name = None
    CNT_LIMIT = 5

    for checkpoint_name in checkpoints_list:
        path = (
            checkpoint_name  # os.path.join(training_args.output_dir, checkpoint_name)
        )
        model_new = SeqClassModel.from_pretrained(
            path, params_dict=config_dict, config=config,
        )
        new_trainer = Trainer(
            model=model_new,
            args=training_args,
            eval_dataset=dev_dataset,
            compute_metrics=compute_seq_classification_metrics,
        )
        dev_results[checkpoint_name] = new_trainer.evaluate()
        curr_dev_f1 = dev_results[checkpoint_name]["eval_f1"]

        if curr_dev_f1 > max_dev_f1:
            max_f1_checkpoint_name = checkpoint_name
            max_dev_f1 = curr_dev_f1
            cnt = 0
        else:
            cnt += 1
            if cnt > CNT_LIMIT:
                break

        eval_trainer = Trainer(
            model=model_new,
            args=training_args,
            eval_dataset=test_dataset,
            compute_metrics=compute_seq_classification_metrics,
        )
        test_results[checkpoint_name] = eval_trainer.evaluate()

    logger.info("dev results:")
    logger.info(str(dev_results))

    eval_results_path = os.path.join(output_dir, "eval_results.txt")

    with open(eval_results_path, "w") as writer:
        writer.write("[dev]\n")
        for key, values in dev_results.items():
            writer.write("%s = %s\n" % (key, str(values)))

        writer.write("[test]\n")
        for key, values in test_results.items():
            writer.write("%s = %s\n" % (key, str(values)))

    model_final_path = os.path.join(output_dir, "final_model/")
    model_final_results_path = os.path.join(model_final_path, "eval_results.txt")

    logger.info("final_checkpoint name: " + max_f1_checkpoint_name)
    model_final = SeqClassModel.from_pretrained(
        max_f1_checkpoint_name, params_dict=config_dict, config=config,
    )
    logger.info(model_final_path)
    try:
        os.makedirs(model_final_path, exist_ok=True)
    except OSError as e:
        logger.info("Model final dir already exists")
    model_final.save_pretrained(model_final_path)
    with open(model_final_results_path, "w") as writer:
        writer.write("[dev]\n")
        writer.write("%s\n" % (str(dev_results[max_f1_checkpoint_name])))

        writer.write("[test]\n")
        writer.write("%s\n" % (str(test_results[max_f1_checkpoint_name])))
