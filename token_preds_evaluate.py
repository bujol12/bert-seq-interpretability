import sys
import os

from utils.tsv_dataset import (
    TSVClassificationDataset,
    Split,
    get_labels,
    compute_seq_classification_metrics,
)
from utils.arguments import datasets, DataTrainingArguments, ModelArguments
from sklearn.metrics import average_precision_score

import logging
from math import sqrt

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


def choose_top_and_threshold(
    examples, importance_threshold, top_count, pos_label, default_label
):
    y_pred = []
    # print(importance_threshold)
    for example in examples:
        predictions = []
        scores = [
            (idx, float(example.labels[idx])) for idx in range(0, len(example.labels))
        ]
        scores.sort(key=lambda x: x[1])

        labels = list(map(lambda x: x[0], scores[-top_count:]))
        count = 0
        for idx in range(0, len(example.labels)):
            label = example.labels[idx]
            # print(label, importance_threshold, idx)
            # print(float(label) >= importance_threshold, idx in labels)
            if float(label) >= importance_threshold and idx in labels:
                predictions.append(pos_label)
                count += 1
            else:
                predictions.append(default_label)
        example.predictions = predictions
        y_pred.append(predictions)
    return y_pred


def pred_stats(y_true, y_pred, label):
    predicted_cnt = 0
    correct_cnt = 0
    total_cnt = 0
    for i in range(0, len(y_true)):
        # print(i, len(y_true[i]), len(y_pred[i]))
        for j in range(0, len(y_true[i])):
            if y_pred[i][j] == label:
                predicted_cnt += 1
            if y_pred[i][j] == label and y_pred[i][j] == y_true[i][j]:
                correct_cnt += 1
            if y_true[i][j] == label:
                total_cnt += 1
    return {
        "predicted_cnt": predicted_cnt,
        "correct_cnt": correct_cnt,
        "total_cnt": total_cnt,
    }


def get_pred_scores(y_true, y_pred, label):
    pred_stats_res = pred_stats(y_true, y_pred, label)
    print(pred_stats_res)
    res = {}
    res["precision"] = (
        pred_stats_res["correct_cnt"] / pred_stats_res["predicted_cnt"]
        if pred_stats_res["predicted_cnt"] > 0
        else 0.0
    )
    res["recall"] = (
        pred_stats_res["correct_cnt"] / pred_stats_res["total_cnt"]
        if pred_stats_res["total_cnt"] > 0
        else 0.0
    )
    res["f1"] = (
        (2.0 * res["precision"] * res["recall"] / (res["precision"] + res["recall"]))
        if (res["precision"] + res["recall"]) > 0
        else 0.0
    )
    res["f0.5"] = (
        ((1 + 0.5 * 0.5) * res["precision"] * res["recall"])
        / (0.5 * 0.5 * res["precision"] + res["recall"])
        if (0.5 * 0.5 * res["precision"] + res["recall"]) > 0
        else 0.0
    )
    return res


def get_corr(y_target, y_pred):
    sum_pred = 0.0
    sum_target = 0.0
    count = 0.0
    assert len(y_pred) == len(y_target)
    for i in range(0, len(y_target)):
        assert len(y_pred[i]) == len(y_target[i])
        for j in range(0, len(y_target[i])):
            if y_target[i][j] >= 0:
                # print(y_target[i][j], y_pred[i][j])
                count += 1.0
                sum_pred += y_pred[i][j]
                sum_target += y_target[i][j]

    sq_diff_pred = 0.0
    sq_diff_target = 0.0
    diff_sum = 0.0
    mean_pred = sum_pred / count
    mean_target = sum_target / count
    for i in range(0, len(y_target)):
        assert len(y_pred[i]) == len(y_target[i])
        for j in range(0, len(y_target[i])):
            if y_target[i][j] >= 0:
                sq_diff_pred += (y_pred[i][j] - mean_pred) ** 2
                sq_diff_target += (y_target[i][j] - mean_target) ** 2
                diff_sum += (y_pred[i][j] - mean_pred) * (y_target[i][j] - mean_target)

    # print (diff_sum, sq_diff_pred, sq_diff_target, mean_pred, mean_target)
    corr = (
        diff_sum / (sqrt(sq_diff_pred) * sqrt(sq_diff_target))
        if sq_diff_pred != 0 and sq_diff_target != 0
        else 0.0
    )
    return corr


def get_map(y_true, y_pred, label):
    sum_val = 0.0
    assert len(y_true) == len(y_pred)
    cnt = 0
    for i in range(len(y_true)):
        if (
            max(y_true[i]) > 0.0
        ):  # only calculate MAP over sentences with positive tokens
            # logger.info("Results:")
            # logger.info(y_true[i])
            # logger.info(y_pred[i])
            ap = average_precision_score(y_true[i], y_pred[i])
            # logger.info(ap)
            sum_val += ap
            cnt += 1
    return sum_val / cnt  # mean AP


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Required args: [config_path]")
        exit()

    logger.info("Parsing Config.")
    config_dict = parse_config(sys.argv[1])

    dataset = datasets[config_dict["dataset"]]
    labels = get_labels(dataset.labels)
    positive_label = dataset.positive_label

    attn_head_id = None
    attn_layer_id = None
    if config_dict["method"] == "model_attention":
        if len(sys.argv) != 4:
            logger.error("Required args: [config_path] [layer_id] [head_id]")
            exit()
        attn_head_id = int(sys.argv[3])
        attn_layer_id = int(sys.argv[2])

    input_dir = config_dict["results_input_dir"].format(
        method=config_dict["method"],
        experiment_name=config_dict["experiment_name"].format(
            attn_layer_id, attn_head_id
        ),
        model_name=config_dict["model_name"],
        dataset_name=config_dict["dataset"],
        datetime=config_dict.get("datetime", ""),
    )

    str2mode = {"dev": Split.dev, "train": Split.train, "test": Split.test}
    mode = str2mode[config_dict["dataset_split"]]
    data_config = dict(
        labels=labels,
        max_seq_length=config_dict["max_seq_length"],
        overwrite_cache=dataset.overwrite_cache,
        make_all_labels_equal_max=False,
        default_label=config_dict["test_label_dummy"],
        is_seq_class=False,
        lowercase=config_dict["lowercase"],
        mode=mode,
        model_type="token_eval",
    )

    logger.info("Reading Token Results.")
    results_dataset = TSVClassificationDataset(
        input_dir,
        tokenizer=None,
        file_name=config_dict["results_input_filename"],
        **data_config
    )

    logger.info("Reading gold labels.")
    eval_dataset = TSVClassificationDataset(
        dataset.data_dir,
        tokenizer=None,
        file_name=dataset.file_name_token,
        **data_config
    )
    print(len(eval_dataset.examples))
    print(len(results_dataset.examples))
    logger.info("Apply threshold and top count")
    y_pred = choose_top_and_threshold(
        results_dataset.examples,
        config_dict["importance_threshold"],
        int(config_dict["top_count"]),
        default_label=labels[0] if labels[0] != positive_label else labels[1],
        pos_label=positive_label,
    )

    y_true = []
    for example in eval_dataset.examples:
        y_true.append(example.labels)

    logger.info("Get pred scores")
    res = get_pred_scores(y_true, y_pred, label=positive_label)

    y_pred_values = list(
        map(
            lambda ex: list(map(lambda x: max(float(x), 0.0), ex.labels)),
            results_dataset.examples,
        )  # make labels be within 0 and 1
    )
    y_true_values = list(
        map(
            lambda ex: list(
                map(lambda l: (1.0 if l == positive_label else 0.0), ex.labels)
            ),
            eval_dataset.examples,
        )
    )
    for i in range(0, len(y_true)):
        if max(y_true_values[i]) > 0:
            logger.info(y_true[i])
            logger.info(y_true_values[i])
            break
    logger.info("Get MAP and Correlation metrics")
    res["MAP"] = get_map(y_true_values, y_pred_values, positive_label)
    res["corr"] = get_corr(y_true_values, y_pred_values)

    logger.info("RESULTS:")
    logger.info(str(res))

    if config_dict.get("eval_results_filename", None) is not None:
        logger.info("saving eval results")
        filename = os.path.join(input_dir, config_dict["eval_results_filename"])
        with open(filename, "w") as fhand:
            fhand.write(str(res))
