"""Multi Variate Gaussian Distribution."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Any, List, Optional

import torch
from torch import Tensor, nn
import numpy as np


class SelectiveFeatureModel(nn.Module):
    """Selective Feature Modelling."""

    def __init__(self, feature_percentage):
        super().__init__()

        #self.register_buffer("feature_stat", torch.zeros(n_features, n_patches))

        self.feature_percentage = feature_percentage
        self.class_stats = {}

    def forward(self, max_activation_val: Tensor, class_labels:List[str]) -> List[Tensor]:
        """Calculate multivariate Gaussian distribution.

        Args:
          embedding (Tensor): CNN features whose dimensionality is reduced via either random sampling or PCA.

        Returns:
          mean and inverse covariance of the multi-variate gaussian distribution that fits the features.
        """
        device = max_activation_val.device

        class_names = np.unique(class_labels)
        # print(class_labels)
        # print(max_activation_val.shape)


        for class_name in class_names:
            print(class_name)
            self.register_buffer(class_name, torch.Tensor())
            setattr(self,class_name,torch.Tensor())
            # print(max_activation_val.shape)
            # print(np.where(class_labels == class_name))
            class_max_activations = max_activation_val[class_labels == class_name]
            # sorted values and idx for entire feature set
            max_val, max_idx = torch.sort(class_max_activations, descending=True)
            reduced_range = int(max_val.shape[1] * 0.10)
            top_max_val = max_val[:, 0:reduced_range]
            # indexes of top 10% FEATURES HAVING MAX VALUE
            top_max_idx = max_idx[:, 0:reduced_range]
            # out of sorted top 10, what features are affiliated the most
            idx, repetitions = torch.unique(top_max_idx, return_counts=True)
            sorted_repetition, sorted_repetition_idx = torch.sort(repetitions, descending=True)
            sorted_idx = idx[sorted_repetition_idx]

            sorted_idx_normalized = sorted_repetition / class_max_activations.shape[0]
            sorted_idx_normalized = sorted_idx_normalized / sorted_idx_normalized.sum()
            #print(torch.cat((sorted_idx.unsqueeze(0), sorted_idx_normalized.unsqueeze(0))))
            self.register_buffer(class_name, torch.Tensor())
            setattr(self, class_name,
                    torch.cat((sorted_idx.unsqueeze(0), sorted_idx_normalized.unsqueeze(0))))

    def fit(self, max_val_features: Tensor, class_labels:List[str]):
        """Fit multi-variate gaussian distribution to the input embedding.

        Args:
            embedding (Tensor): Embedding vector extracted from CNN.

        Returns:
            Mean and the covariance of the embedding.
        """
        self.forward(max_val_features,class_labels)
