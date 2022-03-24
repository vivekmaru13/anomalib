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

from random import sample
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from pytorch_lightning.callbacks import EarlyStopping
from kornia import gaussian_blur2d
from omegaconf import DictConfig, ListConfig
from torch import Tensor, nn, optim
from sklearn.metrics import accuracy_score

from anomalib.core.model import AnomalyModule


__all__ = ["ClassificationLightning"]


class ClassificationModel(nn.Module):
    """Padim Module.

    Args:
        layers (List[str]): Layers used for feature extraction
        input_size (Tuple[int, int]): Input size for the model.
        tile_size (Tuple[int, int]): Tile size
        tile_stride (int): Stride for tiling
        apply_tiling (bool, optional): Apply tiling. Defaults to False.
        backbone (str, optional): Pre-trained model backbone. Defaults to "resnet18".
    """

    def __init__(
        self,
        layers: List[str],
        input_size: Tuple[int, int],
        backbone: str = "resnet18",
        apply_tiling: bool = False,
        tile_size: Optional[Tuple[int, int]] = None,
        tile_stride: Optional[int] = None,
    ):
        super().__init__()
        backbone_obj = getattr(torchvision.models, backbone)
        self.backbone = backbone_obj(pretrained=True)
        self.num_filters = self.backbone.fc.in_features

        self.layers = list(self.backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*self.layers)
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False
        num_target_class = 6
        self.classifier = nn.Linear(self.num_filters, num_target_class)
        self.loss = None

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Forward-pass image-batch (N, C, H, W) into model to extract features.

        Args:
            input_tensor: Image-batch (N, C, H, W)
            input_tensor: Tensor:

        Returns:
            Features from single/multiple layers.

        Example:
            >>> x = torch.randn(32, 3, 224, 224)
            >>> features = self.extract_features(input_tensor)
            >>> features.keys()
            dict_keys(['layer1', 'layer2', 'layer3'])

            >>> [v.shape for v in features.values()]
            [torch.Size([32, 64, 56, 56]),
            torch.Size([32, 128, 28, 28]),
            torch.Size([32, 256, 14, 14])]
        """

        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(input_tensor).flatten(1)
        x = self.classifier(features)

        return x


class ClassificationLightning(AnomalyModule):


    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__(hparams)
        self.layers = hparams.model.layers
        self.model = ClassificationModel(
            layers=hparams.model.layers,
            input_size=hparams.model.input_size,
            tile_size=hparams.dataset.tiling.tile_size,
            tile_stride=hparams.dataset.tiling.stride,
            apply_tiling=hparams.dataset.tiling.apply,
            backbone=hparams.model.backbone,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers by creating an SGD optimizer.

        Returns:
            (Optimizer): SGD optimizer
        """
        return optim.SGD(
            params=self.model.classifier.parameters(),
            lr=self.hparams.model.lr,
            momentum=self.hparams.model.momentum,
        )

    def training_step(self, batch, _):  # pylint: disable=arguments-differ
        """Training Step of PADIM. For each batch, hierarchical features are extracted from the CNN.

        Args:
            batch (Dict[str,Tensor]): Input batch
            _: Index of the batch.

        Returns:
            Hierarchical feature map
        """

        self.model.feature_extractor.eval()
        x = self.model(batch["image"])
        loss = F.cross_entropy(x, batch["class_index"])
        return {"loss": loss}


    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        """Validation Step of PADIM.

        Similar to the training step, hierarchical features are extracted from the CNN for each batch.

        Args:
            batch: Input batch
            _: Index of the batch.

        Returns:
            Dictionary containing images, features, true labels and masks.
            These are required in `validation_epoch_end` for feature concatenation.
        """
        _, preds = torch.max(self.model(batch["image"]), 1)
        batch["pred_labels"] = preds

        return batch

    def validation_step_end(self, val_step_outputs):
        return val_step_outputs

    def test_step_end(self, test_step_outputs):
        return test_step_outputs

    def validation_epoch_end(self, outputs):
        results = torch.hstack([x["pred_labels"] for x in outputs])
        labels = torch.hstack([x["class_index"] for x in outputs])

        accuracy = accuracy_score(labels.cpu(), results.cpu())

        #print(f"accuracy score: {accuracy}")

    def test_epoch_end(self, outputs):
        results = torch.hstack([x["pred_labels"] for x in outputs])
        labels = torch.hstack([x["class_index"] for x in outputs])

        accuracy = accuracy_score(labels.cpu(), results.cpu())

        print(f"accuracy score: {accuracy}")


