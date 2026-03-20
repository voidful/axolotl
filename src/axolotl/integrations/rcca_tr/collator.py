# Copyright 2024 Axolotl AI. All rights reserved.
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
DataCollator for RCCA-TR to handle prior cache fields (prior_target_logp, prior_margin).
"""

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from axolotl.utils.collators.batching import DataCollatorForSeq2Seq

RCCA_TR_FIELDS = ("prior_target_logp", "prior_margin")


@dataclass
class DataCollatorForRCCATR(DataCollatorForSeq2Seq):
    """
    Data collator that extends DataCollatorForSeq2Seq to handle RCCA-TR
    prior cache fields (prior_target_logp, prior_margin).

    These are 1D float lists per sample that need to be padded to batch max length.
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    position_pad_token_id: int = 0
    return_tensors: str = "pt"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    def __call__(self, features, return_tensors=None):
        if isinstance(features[0], list):
            out_features = [{} for _ in features]
            for i, features_ in enumerate(features):
                for feature in features_[0].keys():
                    if feature == "length":
                        continue
                    if feature == "attention_mask":
                        arrays = [
                            (j + 1) * np.array(item[feature])
                            for j, item in enumerate(features_)
                            if feature in item
                        ]
                        out_features[i][feature] = np.concatenate(arrays)
                    else:
                        arrays = [
                            np.array(item[feature]) for item in features_ if feature in item
                        ]
                        out_features[i][feature] = np.concatenate(arrays)
            features = out_features

        if return_tensors is None:
            return_tensors = self.return_tensors

        padding_side = self.tokenizer.padding_side

        # Pad labels and position_ids first (same as KD collator)
        max_len = 0
        for feature_name, pad_token_id in [
            ("labels", self.label_pad_token_id),
            ("position_ids", self.position_pad_token_id),
        ]:
            if feature_name in features[0]:
                feat = [f[feature_name] for f in features]
                max_len = max(len(x) for x in feat)
                if self.pad_to_multiple_of is not None:
                    max_len = (
                        (max_len + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                    ) * self.pad_to_multiple_of

                for f in features:
                    remainder = [pad_token_id] * (max_len - len(f[feature_name]))
                    if isinstance(f[feature_name], list):
                        f[feature_name] = (
                            f[feature_name] + remainder
                            if padding_side == "right"
                            else remainder + f[feature_name]
                        )
                    else:
                        if padding_side == "right":
                            f[feature_name] = np.concatenate(
                                [f[feature_name], remainder]
                            ).astype(np.int64)
                        else:
                            f[feature_name] = np.concatenate(
                                [remainder, f[feature_name]]
                            ).astype(np.int64)

        # Extract RCCA-TR fields before tokenizer.pad()
        rcca_data = {}
        has_rcca_fields = all(field in features[0] for field in RCCA_TR_FIELDS)

        if has_rcca_fields:
            for field in RCCA_TR_FIELDS:
                rcca_data[field] = [f.pop(field) for f in features]

            # Determine max length for padding
            rcca_max_len = max_len or max(
                len(seq) for seq in rcca_data[RCCA_TR_FIELDS[0]]
            )

            # Pad each field to rcca_max_len with 0.0
            padded = {}
            for field in RCCA_TR_FIELDS:
                padded_seqs = []
                for seq in rcca_data[field]:
                    pad_len = rcca_max_len - len(seq)
                    if isinstance(seq, list):
                        padded_seq = (
                            seq + [0.0] * pad_len
                            if padding_side == "right"
                            else [0.0] * pad_len + seq
                        )
                    else:
                        remainder = np.zeros(pad_len, dtype=np.float32)
                        padded_seq = (
                            np.concatenate([seq, remainder])
                            if padding_side == "right"
                            else np.concatenate([remainder, seq])
                        )
                    padded_seqs.append(padded_seq)
                padded[field] = torch.tensor(padded_seqs, dtype=torch.float32)

        # Pad standard fields using tokenizer
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # Add back RCCA-TR fields
        if has_rcca_fields:
            for field in RCCA_TR_FIELDS:
                features[field] = padded[field]

        if (
            "labels" in features
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        return features
