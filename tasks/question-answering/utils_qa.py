# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
"""
Post-processing utilities for question answering.
"""
import collections
import json
import logging
import os
from typing import Optional, Tuple, Dict, List, Union, Any, Callable

from promptsource.templates import DatasetTemplates

import random

import torch
from datasets.arrow_dataset import Dataset
from evaluate import load
from transformers.utils import PaddingStrategy
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)

TASK_MAPPING_DATASET_ARGUMENTS = {
    "wikiqa": ["wiki_qa"],
}

TASK_MAPPING_PROMPT_KEY = {
    "wikiqa": "Decide_good_answer",
}

EXTRA_KEYS_FOR_EVAL = ["id", "idx"]

def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = logging.WARNING,
):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.

    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            The threshold used to select the null answer: if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).

            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        log_level (:obj:`int`, `optional`, defaults to ``logging.WARNING``):
            ``logging`` log level (e.g., ``logging.WARNING``)
    """
    if len(predictions) != 2:
        raise ValueError("`predictions` should be a tuple with two elements (start_logits, end_logits).")
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # Logging.
    logger.setLevel(log_level)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or len(offset_mapping[start_index]) < 2
                        or offset_mapping[end_index] is None
                        or len(offset_mapping[end_index]) < 2
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        if version_2_with_negative and min_null_prediction is not None:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if (
            version_2_with_negative
            and min_null_prediction is not None
            and not any(p["offsets"] == (0, 0) for p in predictions)
        ):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
            scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise EnvironmentError(f"{output_dir} is not a directory.")

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
            )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        if version_2_with_negative:
            logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


def postprocess_qa_predictions_with_beam_search(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    start_n_top: int = 5,
    end_n_top: int = 5,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = logging.WARNING,
):
    """
    Post-processes the predictions of a question-answering model with beam search to convert them to answers that are substrings of the
    original contexts. This is the postprocessing functions for models that return start and end logits, indices, as well as
    cls token predictions.

    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        start_n_top (:obj:`int`, `optional`, defaults to 5):
            The number of top start logits too keep when searching for the :obj:`n_best_size` predictions.
        end_n_top (:obj:`int`, `optional`, defaults to 5):
            The number of top end logits too keep when searching for the :obj:`n_best_size` predictions.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        log_level (:obj:`int`, `optional`, defaults to ``logging.WARNING``):
            ``logging`` log level (e.g., ``logging.WARNING``)
    """
    if len(predictions) != 5:
        raise ValueError("`predictions` should be a tuple with five elements.")
    start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict() if version_2_with_negative else None

    # Logging.
    logger.setLevel(log_level)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_log_prob = start_top_log_probs[feature_index]
            start_indexes = start_top_index[feature_index]
            end_log_prob = end_top_log_probs[feature_index]
            end_indexes = end_top_index[feature_index]
            feature_null_score = cls_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction
            if min_null_score is None or feature_null_score < min_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_start_top`/`n_end_top` greater start and end logits.
            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_index = int(start_indexes[i])
                    j_index = i * end_n_top + j
                    end_index = int(end_indexes[j_index])
                    # Don't consider out-of-scope answers (last part of the test should be unnecessary because of the
                    # p_mask but let's not take any risk)
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or len(offset_mapping[start_index]) < 2
                        or offset_mapping[end_index] is None
                        or len(offset_mapping[end_index]) < 2
                    ):
                        continue

                    # Don't consider answers with a length negative or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_log_prob[i] + end_log_prob[j_index],
                            "start_log_prob": start_log_prob[i],
                            "end_log_prob": end_log_prob[j_index],
                        }
                    )

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0:
            # Without predictions min_null_score is going to be None and None will cause an exception later
            min_null_score = -2e-6
            predictions.insert(0, {"text": "", "start_logit": -1e-6, "end_logit": -1e-6, "score": min_null_score})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction and set the probability for the null answer.
        all_predictions[example["id"]] = predictions[0]["text"]
        if version_2_with_negative:
            scores_diff_json[example["id"]] = float(min_null_score)

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise EnvironmentError(f"{output_dir} is not a directory.")

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
            )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        if version_2_with_negative:
            logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions, scores_diff_json


def get_label_mapping_id(task: str) -> Dict[str, int]:
    """
    Examples
    --------
    >>> get_label_mapping_id("multirc")
    {'No': 0, 'Yes': 1}
    """
    prompt = DatasetTemplates(*TASK_MAPPING_DATASET_ARGUMENTS[task])[TASK_MAPPING_PROMPT_KEY[task]]
    choices = prompt.get_fixed_answer_choices_list()
    if task == "winogrande":
        # to minuscule for more balanced `input_ids` length
        choices = [choice.lower() for choice in choices]
    return {choice: i for i, choice in enumerate(choices)}


class Seq2SeqDataPreProcessor:
    """
    Examples
    --------
    >>> from datasets import load_dataset
    >>> proc = Seq2SeqDataPreProcessor("multirc")
    >>> dataset = load_dataset("super_glue", "multirc", split="train[:4]")
    >>> proc(dataset[0]).keys()
    dict_keys(['inputs', 'targets'])
    >>> len(proc(dataset[:2])['inputs'])
    2
    """

    def __init__(self, benchmark: str, keep_specific_keys: List[str] = None):
        self.benchmark = benchmark
        available_prompts = DatasetTemplates(*TASK_MAPPING_DATASET_ARGUMENTS[self.benchmark])
        self.prompt = available_prompts[TASK_MAPPING_PROMPT_KEY[self.benchmark]]
        self.keep_specific_keys = keep_specific_keys if keep_specific_keys else []

    def __call__(self, examples: Dict[str, Union[List, Any]], batched: Optional[bool] = True) -> Dict[str, List]:
        first_key = list(examples.keys())[0]
        if isinstance(examples[first_key], list) or batched:
            batch_size = len(examples["label"]) if "label" in examples else len(examples[first_key])
            ret = {'inputs': [], 'targets': []}
            for i in range(batch_size):
                result = self.prompt.apply({k: v[i] for k, v in examples.items()})
                ret['inputs'].append(result[0])
                if self.benchmark == "winogrande":
                    ret['targets'].append(result[1].lower())
                else:
                    ret['targets'].append(result[1])
        else:
            result = self.prompt.apply(examples)
            ret = {
                'inputs': result[0],
                'targets': result[1] if self.benchmark != "winogrande" else result[1].lower()
            }
        for key in examples:
            if key not in ret and key in self.keep_specific_keys:
                ret[key] = examples[key]
        return ret


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    keys_to_ignore: Optional[List[str]] = None

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        features_ignored = {}
        if self.keys_to_ignore is None:
            self.keys_to_ignore = []
        for key in self.keys_to_ignore:
            if key in features[0].keys():
                features_ignored[key] = [feature.pop(key) for feature in features]
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return {**features, **features_ignored}


def keep_only_supporting_facts_in_context_for_hotpotqa(examples: Dict[str, Any]):
    """ This is for fxxking long context in HotpotQA. Now keep only supporting facts in context. ^^ """
    new_context = {
        'title': [],
        'sentences': []
    }
    sup_facts = examples['supporting_facts']
    for title, sent_ids in zip(sup_facts['title'], sup_facts['sent_id']):
        vanilla_index = examples['context']['title'].index(title)
        vanilla_sentences = examples['context']['sentences'][vanilla_index]
        if len(vanilla_sentences) <= sent_ids:
            continue
        if title not in new_context['title']:
            new_context['title'].append(title)
            new_context['sentences'].append([vanilla_sentences[sent_ids]])
        else:
            new_context['sentences'][new_context['title'].index(title)].append(
                vanilla_sentences[sent_ids])
    examples['context'] = new_context
    return examples

def get_evaluate_fn(
        task: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        raw_eval_dataset: Optional[Dataset] = None
) -> Callable:
    if task in ["squad", "squad_v2", "hotpotqa"]:
        return get_squad_evaluate_fn(tokenizer)
    elif task == "openbookqa":
        return get_openbookqa_evaluate_fn(tokenizer)
    elif task == "copa":
        return get_copa_evaluate_fn(tokenizer, raw_eval_dataset)
    elif task == "multirc":
        return get_multirc_evaluate_fn(tokenizer)
    elif task == "stsb":
        return get_stsb_evaluate_fn(tokenizer)
    else:
        # including other GLUE tasks, WinoGrande, WikiQA
        return get_cls_evaluate_fn(task, tokenizer)


def get_classification_label_index_and_token_ids(
        task: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> tuple:
    prompt = DatasetTemplates(*TASK_MAPPING_DATASET_ARGUMENTS[task])[TASK_MAPPING_PROMPT_KEY[task]]
    if task in ["sst2", "mrpc"]:
        classification_token_index = 0
        choices = " ".join(prompt.get_fixed_answer_choices_list())
        classification_label_token_ids = tokenizer.encode(choices, add_special_tokens=False)
    else:
        return None, None
    return classification_token_index, classification_label_token_ids


def get_cls_evaluate_fn(
        task: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> Callable:
    """
    Get the evaluate function for GLUE tasks.
    """

    def evaluate_fn(predictions: Union[torch.Tensor, List[int], List[torch.LongTensor]], labels: torch.Tensor,
                    **kwargs):
        try:
            metric = load(*TASK_MAPPING_DATASET_ARGUMENTS[task])
        except FileNotFoundError:
            print(f"[Evaluation warning] No metric found for task {task}, using accuracy instead.")
            metric = load("accuracy")
        label_mapping_id = get_label_mapping_id(task)
        eos_token_id = tokenizer.eos_token_id
        predictions = [p[:p.index(eos_token_id) + 1] if eos_token_id in p else p for
                       p in predictions]
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)  # string lengths batch
        not_even_wrong = 0
        not_even_wrong_predictions = []
        for p in predictions:
            if p not in label_mapping_id:
                not_even_wrong += 1
                not_even_wrong_predictions.append(p)
        if not_even_wrong > 0:
            print(
                f"[Evaluation warning] {not_even_wrong} among total {len(predictions)} predictions are not in the label mapping.")
            print(f"[Evaluation warning] Some of the not even wrong predictions are: {not_even_wrong_predictions[:16]}")
        print(predictions[:32])
        predictions = [label_mapping_id[p] if p in label_mapping_id else random.choice(list(label_mapping_id.values()))
                       for p in predictions]
        references = labels.long().masked_fill(labels == -100, tokenizer.pad_token_id).tolist()
        references = tokenizer.batch_decode(references, skip_special_tokens=True)
        print(references[:32])
        references = [label_mapping_id[r]
                      for r in references]
        return metric.compute(predictions=predictions, references=references)

    return evaluate_fn


def get_stsb_evaluate_fn(
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> Callable:
    """
    Get the evaluate function for GLUE tasks.
    """

    def _is_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def evaluate_fn(predictions: Union[torch.Tensor, List[int], List[torch.LongTensor]], labels: torch.Tensor,
                    **kwargs):
        metric = load(*TASK_MAPPING_DATASET_ARGUMENTS["stsb"])
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)  # string lengths batch
        references = labels.long().masked_fill(labels == -100, tokenizer.pad_token_id).tolist()
        references = tokenizer.batch_decode(references, skip_special_tokens=True)
        not_even_wrong = 0
        not_even_wrong_predictions = []
        not_even_wrong_references = []
        for i, p in enumerate(predictions):
            if not _is_float(p):
                not_even_wrong += 1
                not_even_wrong_predictions.append(p)
                not_even_wrong_references.append(references[i])
        if not_even_wrong > 0:
            print(
                f"[Evaluation warning] {not_even_wrong} among total {len(predictions)} predictions are not in the label mapping.")
        predictions = [float(p) if _is_float(p) else round(random.uniform(0, 25)) / 5 for p in predictions]

        references = [float(r) for r in references]
        return metric.compute(predictions=predictions, references=references)

    return evaluate_fn


def get_multirc_evaluate_fn(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> Callable:
    """
    Get the evaluate function for SuperGLUE-MultiRC task.
    """

    def evaluate_fn(predictions: Union[torch.Tensor, List[int], List[torch.LongTensor]], labels: torch.Tensor,
                    ids: List[Dict[str, int]], **kwargs):
        metric = load(*TASK_MAPPING_DATASET_ARGUMENTS['multirc'])
        label_mapping_id = get_label_mapping_id('multirc')
        # cut up tokens after the first [EOS] token, predictions is a list of token id list
        eos_token_id = tokenizer.eos_token_id
        predictions = [p[:p.index(eos_token_id) + 1] if eos_token_id in p else p for
                       p in predictions]
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)  # string lengths batch
        not_even_wrong = 0
        not_even_wrong_preds = []
        for p in predictions:
            if p not in label_mapping_id:
                not_even_wrong += 1
                not_even_wrong_preds.append(p)
        if not_even_wrong > 0:
            print(
                f"[Evaluation warning] {not_even_wrong} among total {len(predictions)} predictions are not in the label mapping.")
            print(f"[Evaluation warning] Some of the not even wrong predictions are: {not_even_wrong_preds[:16]}")
        predictions = [label_mapping_id[p] if p in label_mapping_id else random.choice([0, 1])
                       for p in predictions]
        references = labels.long().masked_fill(labels == -100, tokenizer.pad_token_id).tolist()
        references = tokenizer.batch_decode(references, skip_special_tokens=True)
        predictions = [
            {'prediction': p, 'idx': id} for p, id in zip(predictions, ids)
        ]
        references = [label_mapping_id[r] for r in references]
        results = metric.compute(predictions=predictions, references=references)
        return {
            'f1_a': results['f1_a'],
            'f1_m': results['f1_m'],
            'exact_match': results['exact_match']
        }

    return evaluate_fn


def get_squad_evaluate_fn(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> Callable:
    """
    Get the evaluate function for SQuAD tasks.
    """

    def evaluate_fn(predictions: Union[torch.Tensor, List[int], List[torch.LongTensor]], labels: torch.Tensor,
                    ids: List[str], **kwargs):
        metric = load(*TASK_MAPPING_DATASET_ARGUMENTS['squad'])
        # cut up tokens after the first [EOS] token, predictions is a list of token id list
        eos_token_id = tokenizer.eos_token_id
        predictions = [p[:p.index(eos_token_id) + 1] if eos_token_id in p else p for
                       p in predictions]
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)  # string lengths batch
        references = labels.long().masked_fill(labels == -100, tokenizer.pad_token_id).tolist()
        references = tokenizer.batch_decode(references, skip_special_tokens=True)
        for i in range(8):
            print(predictions[i], references[i])
        predictions = [
            {'prediction_text': p, 'id': id} for p, id in zip(predictions, ids)
        ]
        references = [
            # answer_start is not used in the evaluation, so fake it
            {'answers': {'text': [reference], 'answer_start': [2333]}, 'id': id} for reference, id in
            zip(references, ids)
        ]
        return metric.compute(predictions=predictions, references=references)

    return evaluate_fn


def get_openbookqa_evaluate_fn(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> Callable:
    """
    Get the evaluate function for OpenBookQA tasks.
    """

    def evaluate_fn(predictions: Union[torch.Tensor, List[int], List[torch.LongTensor]], labels: torch.Tensor,
                    **kwargs):
        metric = load("accuracy")
        label_mapping_id = get_label_mapping_id('openbookqa')
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)  # string lengths batch
        predictions = [label_mapping_id[p] if p in label_mapping_id else -1
                       for p in predictions]
        references = labels.long().masked_fill(labels == -100, tokenizer.pad_token_id).tolist()
        references = tokenizer.batch_decode(references, skip_special_tokens=True)
        references = [label_mapping_id[reference]
                      for reference in references]
        return metric.compute(predictions=predictions, references=references)

    return evaluate_fn


def get_copa_evaluate_fn(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                         raw_eval_dataset: Dataset) -> Callable:
    """
    Get the evaluate function for SuperGLUE-COPA task.
    """
    id2choices = {
        item['idx']: [item['choice1'], item['choice2']] for item in raw_eval_dataset
    }
    id2references = {
        item['idx']: item['label'] for item in raw_eval_dataset
    }

    def evaluate_fn(predictions: Union[torch.Tensor, List[int], List[torch.LongTensor]], ids: List[int], **kwargs):
        metric = load('super_glue', 'copa')
        # cut up tokens after the first [EOS] token, predictions is a list of token id list
        eos_token_id = tokenizer.eos_token_id
        predictions = [p[:p.index(eos_token_id) + 1] if eos_token_id in p else p for
                       p in predictions]
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)  # string lengths batch
        predictions = [id2choices[idx].index(p) if p in id2choices[idx] else -1 for idx, p in zip(ids, predictions)]
        references = [id2references[idx] for idx in ids]
        return metric.compute(predictions=predictions, references=references)

    return evaluate_fn


def gather_predictions_references_by_causal_lm_loss(
        ids_list: List[int],
        answer_ids_list: List[int],
        choice_ids_list: List[int],
        losses_list: List[float],
) -> Dict[str, List[int]]:
    assert len(ids_list) == len(answer_ids_list) == len(choice_ids_list) == len(losses_list)
    num_choices = max(choice_ids_list) + 1
    idx_choice_idx_to_loss = {}
    idx_to_answer_idx = {}
    for idx, choice_idx, answer_idx, loss in zip(ids_list, choice_ids_list, answer_ids_list, losses_list):
        idx_choice_idx_to_loss[f"{idx}-{choice_idx}"] = loss
        if idx not in idx_to_answer_idx:
            idx_to_answer_idx[idx] = answer_idx
        else:
            assert idx_to_answer_idx[idx] == answer_idx
    predictions = []
    references = []
    for idx in sorted(idx_to_answer_idx.keys()):
        idx_loss_list = []
        for choice_idx in range(num_choices):
            idx_loss_list.append(idx_choice_idx_to_loss[f"{idx}-{choice_idx}"])
        best_choice_idx = idx_loss_list.index(min(idx_loss_list))
        predictions.append(best_choice_idx)
        references.append(idx_to_answer_idx[idx])
    return {"predictions": predictions, "references": references}

def tokenize_seq2seq(
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        batch: Dict[str, List],
        keep_other_keys=False,
        max_sequence_length=None,
        max_answer_length=None,
) -> Dict[str, List]:
    inputs = tokenizer(batch.pop("inputs"), truncation=True, return_attention_mask=True)
    targets = tokenizer(batch.pop("targets"), truncation=True, padding=False, return_attention_mask=False)
    labels = targets["input_ids"]
    # Replace pad_token_id 0 to -100 in labels
    labels = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in labels]
    ret = {"input_ids": inputs["input_ids"],
           "attention_mask": inputs["attention_mask"],
           "labels": labels}
    # This is for some dataset evaluation like "idx" in MultiRC, "id" in SQuAD
    if keep_other_keys:
        for key in batch:
            ret[key] = batch[key]
    return ret
