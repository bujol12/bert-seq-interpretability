import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import uuid
import json
import torch

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
    PreTrainedModel,
    RobertaModel,
    BertModel,
    BertPreTrainedModel,
)

from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from transformers import DataCollatorForLanguageModeling

import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from utils.tsv_dataset import (
    TSVClassificationDataset,
    Split,
    get_labels,
    compute_seq_classification_metrics,
    MaskedDataCollator,
)
from utils.arguments import datasets, DataTrainingArguments, ModelArguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SoftAttentionSeqClassModel(nn.Module):
    def __init__(self, config_dict, bert_out_size, num_labels):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.initializer_name = config_dict["initializer_name"]
        self.num_labels = num_labels
        self.gamma = config_dict["soft_attention_gamma"]

        self.dropout = nn.Dropout(p=config_dict["hid_to_attn_dropout"])

        self.attention_evidence = nn.Linear(
            bert_out_size, config_dict["attention_evidence_size"]
        )  # layer for predicting attention weights
        self.attention_weights = nn.Linear(config_dict["attention_evidence_size"], 1)

        self.final_hidden = nn.Linear(
            bert_out_size, config_dict["final_hidden_layer_size"],
        )
        self.result_layer = nn.Linear(
            config_dict["final_hidden_layer_size"], self.num_labels
        )

        if config_dict["attention_activation"] == "sharp":
            self.attention_act = torch.exp
        elif config_dict["attention_activation"] == "soft":
            self.attention_act = torch.sigmoid
        elif config_dict["attention_activation"] == "linear":
            pass
        else:
            raise ValueError(
                "Unknown activation for attention: "
                + str(self.config["attention_activation"])
            )
        self.apply(self.init_weights)

    def __call__(self, *input, **kwargs):
        result = self.forward(*input, **kwargs)
        return result

    def init_weights(self, m):
        if self.initializer_name == "normal":
            self.initializer = nn.init.normal_
        elif self.initializer_name == "glorot":
            self.initializer = nn.init.xavier_normal_
        elif self.initializer_name == "xavier":
            self.initializer = nn.init.xavier_uniform_

        if isinstance(m, nn.Linear):
            self.initializer(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        bert_hidden_outputs,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        inp_lengths = (input_ids != 0).sum(dim=1)
        after_dropout = self.dropout(bert_hidden_outputs)
        attn_evidence = self.attention_evidence(after_dropout)
        attn_weights = self.attention_weights(attn_evidence)

        attn_weights = attn_weights.view(
            bert_hidden_outputs.size()[:2]
        )  # batch_size, seq_length
        attn_weights[:, 0] = 0.0  # exclude CLS token

        attn_weights = self.attention_act(attn_weights)

        self.attention_weights_unnormalised = attn_weights

        attn_weights = torch.where(
            self._sequence_mask(inp_lengths, maxlen=input_ids.shape[1]),
            attn_weights,
            torch.zeros_like(attn_weights),  # seq length
        )

        # normalise attn weights
        attn_weights = attn_weights / torch.sum(attn_weights, dim=1, keepdim=True)
        self.attention_weights_normalised = attn_weights

        proc_tensor = torch.bmm(
            after_dropout.transpose(1, 2), attn_weights.unsqueeze(2)
        ).squeeze(dim=2)
        proc_tensor = torch.tanh(self.final_hidden(proc_tensor))

        self.sentence_scores = torch.sigmoid(self.result_layer(proc_tensor))
        self.sentence_scores = self.sentence_scores.view(
            [bert_hidden_outputs.shape[0], self.num_labels]
        )

        outputs = (self.sentence_scores, self.attention_weights_normalised)
        loss = None

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(self.sentence_scores.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    self.sentence_scores.view(-1, self.num_labels), labels.view(-1)
                )
            if self.gamma != 0:
                min_attentions, _ = torch.min(
                    torch.where(
                        self._sequence_mask(inp_lengths, maxlen=input_ids.shape[1]),
                        self.attention_weights_unnormalised,
                        torch.zeros_like(self.attention_weights_unnormalised) + 1e6,
                    )[:, 1:],
                    dim=1,
                )
                # don't include 0 for CLS token
                max_attentions, _ = torch.max(
                    torch.where(
                        self._sequence_mask(inp_lengths, maxlen=input_ids.shape[1]),
                        self.attention_weights_unnormalised,
                        torch.zeros_like(self.attention_weights_unnormalised) - 1e6,
                    ),
                    dim=1,
                )

                l2 = torch.mean(torch.square(min_attentions.view(-1)))
                l3 = torch.mean(torch.square(max_attentions.view(-1) - labels.view(-1)))

                loss += self.gamma * (l2 + l3)
            outputs = (loss,) + outputs

        return outputs

    def _sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1).to(self.device)
        matrix = torch.unsqueeze(lengths, dim=-1)

        mask = row_vector < matrix

        mask.type(dtype)
        return mask


class SeqClassModel(PreTrainedModel):
    model_name: str
    config_dict: Dict
    base_model_prefix = "seq_class"

    def __init__(self, model_config, params_dict):
        super().__init__(model_config)

        self.config_dict = params_dict
        self.model_name = self.config_dict.get("model_name")

        set_seed(self.config_dict["seed"])

        ## straight from HuggingFace BERT/Roberta code
        self.num_labels = model_config.num_labels
        self.initializer_name = self.config_dict["initializer_name"]

        self.bert = AutoModel.from_pretrained(
            self.config_dict["model_name"],
            from_tf=bool(".ckpt" in self.config_dict["model_name"]),
            config=model_config,
        )
        self.post_bert_model = None
        if self.config_dict.get("soft_attention", False):
            self.post_bert_model = SoftAttentionSeqClassModel(
                self.config_dict, self.bert.config.hidden_size, model_config.num_labels,
            )
        else:
            self.dropout = nn.Dropout(model_config.hidden_dropout_prob)
            self.classifier = nn.Linear(
                model_config.hidden_size, model_config.num_labels
            )
            self.dropout.apply(self.init_weights)
            self.classifier.apply(self.init_weights)

    def __call__(self, *input, **kwargs):
        result = self.forward(*input, **kwargs)
        return result

    def init_weights(self, m):
        if self.initializer_name == "normal":
            self.initializer = nn.init.normal_
        elif self.initializer_name == "glorot":
            self.initializer = nn.init.xavier_normal_
        elif self.initializer_name == "xavier":
            self.initializer = nn.init.xavier_uniform_

        if isinstance(m, nn.Linear):
            self.initializer(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )  # hidden states, attentions
        if self.post_bert_model is not None:
            outputs = self.post_bert_model.forward(
                outputs[0],
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
            )  # (loss), logits, word attentions
        else:
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            # logits shape: (batch_size, num_labels), labels shape: (batch_size)
            loss = None
            outputs = (logits,) + outputs[2:]
            if labels is not None:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
