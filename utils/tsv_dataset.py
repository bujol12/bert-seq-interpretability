import torch
from torch import nn
from torch.utils.data.dataset import Dataset

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Union, Dict
from enum import Enum

from filelock import FileLock

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available
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


logger = logging.getLogger(__name__)


@dataclass
class InputWords(List[str]):
    """
    A single example words - used to create InputExample
    """


@dataclass
class InputExample:
    """
    A single training/test example for token classification.
    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    words: InputWords
    labels: Optional[List[str]]
    predictions: Optional[List[float]] = None


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    labels: Optional[int] = None


@dataclass
class MaskedDataCollator:
    tokenizer: PreTrainedTokenizer
    do_mask: bool = True
    mask_prob: float = 0.15

    def collate_batch(self, features: List[InputFeatures]) -> Dict[str, torch.Tensor]:
        # In this function we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        if self.do_mask:
            features = self.mask_tokens(features)

        if not isinstance(features[0], dict):
            features = [vars(f) for f in features]

        first = features[0]
        batch = {}

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if "label" in first and first["label"] is not None:
            label = (
                first["label"].item()
                if isinstance(first["label"], torch.Tensor)
                else first["label"]
            )
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = (
                    torch.long if type(first["label_ids"][0]) is int else torch.float
                )
                batch["labels"] = torch.tensor(
                    [f["label_ids"] for f in features], dtype=dtype
                )

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if (
                k not in ("label", "label_ids")
                and v is not None
                and not isinstance(v, str)
            ):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                else:
                    batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)

        return batch

    def mask_tokens(self, inputs: InputFeatures) -> InputFeatures:
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        probability_matrix = torch.full(
            (len(inputs), len(inputs[0].input_ids)), self.mask_prob
        )
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(
                val.input_ids, already_has_special_tokens=True
            )
            for val in inputs
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        if self.tokenizer._pad_token is not None:
            padding_mask = []
            for feat in inputs:
                padding_mask.append(
                    [
                        1 if inp == self.tokenizer.pad_token_id else 0
                        for inp in feat.input_ids
                    ]
                )
            probability_matrix.masked_fill_(torch.tensor(padding_mask), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(
                torch.full((len(inputs), len(inputs[0].input_ids)), 0.8)
            ).bool()
            & masked_indices
        )

        for i in range(0, indices_replaced.size(0)):
            for j in range(0, indices_replaced.size(1)):
                if indices_replaced[i][j]:
                    inputs[i].input_ids[j] = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.mask_token
                    )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(
                torch.full((len(inputs), len(inputs[0].input_ids)), 0.5)
            ).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer),
            (len(inputs), len(inputs[0].input_ids)),
            dtype=torch.long,
        )
        for i in range(0, indices_random.size(0)):
            for j in range(0, indices_random.size(1)):
                if indices_random[i][j]:
                    inputs[i].input_ids[j] = random_words[i][j]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class TSVClassificationDataset(Dataset):
    """
    A Tab separated dataset, input in the format of:
    <token>      <label>
    <token>      <label>
    <token>      <label>
    <token>      <label>
    \n (next sequence)
    <token>      <label>
    <token>      <label>
    <token>      <label>

    For token level classification, the labels are next to coresponding tokens.
    For sequence level classification, all labels for the given sequence are the same.
    """

    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        labels: List[str],
        model_type: str,
        file_name: str,
        default_label: str,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        mode: Split = Split.train,
        make_all_labels_equal_max: bool = False,
        is_seq_class: bool = False,
        lowercase: bool = False,
    ):

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}".format(
                mode.value, tokenizer.__class__.__name__, str(max_seq_length)
            ),
        )
        logger.info(f"Creating features from dataset file at {data_dir}")
        self.examples = read_examples_from_file(
            data_dir, mode, file_name, default_label
        )
        # TODO clean up all this to leverage built-in features of tokenizers
        if tokenizer is not None:
            self.convert_features_dict = dict(
                label_list=labels,
                max_seq_length=max_seq_length,
                tokenizer=tokenizer,
                make_all_labels_equal_max=make_all_labels_equal_max,
                cls_token_at_end=bool(model_type in ["xlnet"]),
                # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=False,
                # roberta uses an extra separator b/w pairs of sentences,
                # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left=bool(tokenizer.padding_side == "left"),
                pad_token=tokenizer.pad_token_id,
                pad_token_segment_id=tokenizer.pad_token_type_id,
                pad_token_label_id=self.pad_token_label_id,
                is_seq_class=is_seq_class,
                lowercase=lowercase,
            )
            self.features = convert_examples_to_features(
                self.examples, **self.convert_features_dict
            )
            logger.info(f"Saving features into cached file {cached_features_file}")
            torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def write_preds_to_file(self, filename, examples=None):
        if examples is None:
            examples = self.examples

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "w") as file:
            for example in examples:
                res_str = ""
                for i in range(0, len(example.words)):
                    res_str += (
                        example.words[i] + "\t" + str(example.predictions[i]) + "\n"
                    )
                res_str += "\n"
                file.writelines(res_str)


def read_examples_from_file(
    data_dir, mode: Union[Split, str], file_name: str, default_label: str
) -> List[InputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    guid_index = 1
    file_path = os.path.join(data_dir, file_name.format(mode=mode.lower()))
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(
                        InputExample(
                            guid=f"{mode}-{guid_index}", words=words, labels=labels
                        )
                    )
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split("\t")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    assert mode.lower() == "test"
                    # Examples could have no label for mode = "test"
                    labels.append(default_label)
        if words:
            examples.append(
                InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels)
            )
    return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    make_all_labels_equal_max=False,  # used for seq classification
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    is_seq_class: bool = False,
    lowercase: bool = False,
) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            # logger.info((word, label))
            if lowercase and not word in list(tokenizer.special_tokens_map.values()):
                word = word.lower()  # convert all words to lowercase
            word_tokens = tokenizer.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend(
                    [label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1)
                )

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        if (
            is_seq_class and make_all_labels_equal_max
        ):  # skip neg labels of tokens CLS and SEP
            label_max = max(label_ids)
            label_ids = [label_max for i in range(0, len(label_ids))]

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if is_seq_class:
            labels = label_ids[0]  # classify only based on first label

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            if is_seq_class:
                logger.info("labels: %s", str(labels))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None

        if is_seq_class:
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    token_type_ids=segment_ids,
                    labels=labels,
                )
            )
        else:
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    token_type_ids=segment_ids,
                    label_ids=label_ids,
                )
            )
    return features


def get_labels(path: str) -> List[str]:
    """
    Read labels from second elem in the TSV-formatted file.
    """
    labels = set()
    if path:
        with open(path, "r") as f:
            for line in f:
                tokens = line.split()
                if len(tokens) >= 2:
                    labels.add(tokens[1].replace("\n", ""))
    return list(labels)


def compute_seq_classification_metrics(p: EvalPrediction) -> Dict:
    preds_list = np.argmax(p.predictions, axis=1).astype(int)
    out_label_list = p.label_ids.astype(int)
    return {
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
        "accuracy": accuracy_score(out_label_list, preds_list),
    }
