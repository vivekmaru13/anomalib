"""PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.

Paper https://arxiv.org/abs/2011.08785
"""

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

import torch
import torch.nn.functional as F
import torchvision
from kornia import gaussian_blur2d
from omegaconf import DictConfig, ListConfig
from torch import Tensor, nn
import numpy as np

from anomalib.core.model import AnomalyModule
from anomalib.core.model.feature_extractor import FeatureExtractor
from anomalib.core.model.multi_variate_gaussian import MultiVariateGaussian
from anomalib.core.model.selective_feature_modelling import SelectiveFeatureModel
from anomalib.data.tiler import Tiler

__all__ = ["PadimLightning"]


DIMS = {
    "resnet18": {"orig_dims": 448, "reduced_dims": 100, "emb_scale": 4},
    "wide_resnet50_2": {"orig_dims": 1792, "reduced_dims": 550, "emb_scale": 4},
}


class PadimModel(nn.Module):
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
        self.backbone = getattr(torchvision.models, backbone)
        self.layers = layers
        self.apply_tiling = apply_tiling
        self.feature_extractor = FeatureExtractor(backbone=self.backbone(pretrained=True), layers=self.layers)
        self.dims = DIMS[backbone]
        # pylint: disable=not-callable
        # Since idx is randomly selected, save it with model to get same results
        #self.register_buffer(
        #    "idx",
        #    torch.tensor(sample(range(0, DIMS[backbone]["orig_dims"]), DIMS[backbone]["reduced_dims"])),
        #)
        self.register_buffer(
            "idx",
            torch.tensor(range(0, DIMS[backbone]["orig_dims"])),
        )
        self.idx: Tensor
        self.loss = None
        self.anomaly_map_generator = AnomalyMapGenerator(image_size=input_size)
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)

        n_features = DIMS[backbone]["orig_dims"]
        patches_dims = torch.tensor(input_size) / DIMS[backbone]["emb_scale"]
        n_patches = patches_dims.prod().int().item()
        self.gaussian = MultiVariateGaussian(n_features, n_patches)
        self.feature_model = SelectiveFeatureModel(0.1)

        if apply_tiling:
            assert tile_size is not None
            assert tile_stride is not None
            self.tiler = Tiler(tile_size, tile_stride)

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

        if self.apply_tiling:
            input_tensor = self.tiler.tile(input_tensor)
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
            embeddings = self.generate_embedding(features)
        if self.apply_tiling:
            embeddings = self.tiler.untile(embeddings)

        if self.training:
            output = embeddings
        else:
            output = self.anomaly_map_generator(
                embedding=embeddings, mean=self.gaussian.mean, inv_covariance=self.gaussian.inv_covariance
            )

        return output

    def generate_embedding(self, features: Dict[str, Tensor]) -> Tensor:
        """Generate embedding from hierarchical feature map.

        Args:
            features (Dict[str, Tensor]): Hierarchical feature map from a CNN (ResNet18 or WideResnet)

        Returns:
            Embedding vector
        """

        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        # subsample embeddings
        idx = self.idx.to(embeddings.device)
        embeddings = torch.index_select(embeddings, 1, idx)
        return embeddings


class AnomalyMapGenerator:
    """Generate Anomaly Heatmap.

    Args:
        image_size (Union[ListConfig, Tuple]): Size of the input image. The anomaly map is upsampled to this dimension.
        sigma (int, optional): Standard deviation for Gaussian Kernel. Defaults to 4.
    """

    def __init__(self, image_size: Union[ListConfig, Tuple], sigma: int = 4):
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)
        self.sigma = sigma

    @staticmethod
    def compute_distance(embedding: Tensor, stats: List[Tensor]) -> Tensor:
        """Compute anomaly score to the patch in position(i,j) of a test image.

        Ref: Equation (2), Section III-C of the paper.

        Args:
            embedding (Tensor): Embedding Vector
            stats (List[Tensor]): Mean and Covariance Matrix of the multivariate Gaussian distribution

        Returns:
            Anomaly score of a test image via mahalanobis distance.
        """

        batch, channel, height, width = embedding.shape
        embedding = embedding.reshape(batch, channel, height * width)

        # calculate mahalanobis distances
        mean, inv_covariance = stats
        delta = (embedding - mean).permute(2, 0, 1)

        distances = (torch.matmul(delta, inv_covariance) * delta).sum(2).permute(1, 0)
        distances = distances.reshape(batch, height, width)
        distances = torch.sqrt(distances)
        max_activation_val, _ = torch.max((torch.matmul(delta, inv_covariance) * delta), 0)

        return (distances, max_activation_val)

    def up_sample(self, distance: Tensor) -> Tensor:
        """Up sample anomaly score to match the input image size.

        Args:
            distance (Tensor): Anomaly score computed via the mahalanobis distance.

        Returns:
            Resized distance matrix matching the input image size
        """

        score_map = F.interpolate(
            distance.unsqueeze(1),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )
        return score_map

    def smooth_anomaly_map(self, anomaly_map: Tensor) -> Tensor:
        """Apply gaussian smoothing to the anomaly map.

        Args:
            anomaly_map (Tensor): Anomaly score for the test image(s).

        Returns:
            Filtered anomaly scores
        """

        kernel_size = 2 * int(4.0 * self.sigma + 0.5) + 1
        anomaly_map = gaussian_blur2d(anomaly_map, (kernel_size, kernel_size), sigma=(self.sigma, self.sigma))

        return anomaly_map

    def compute_anomaly_map(self, embedding: Tensor, mean: Tensor, inv_covariance: Tensor) -> Tensor:
        """Compute anomaly score.

        Scores are calculated based on embedding vector, mean and inv_covariance of the multivariate gaussian
        distribution.

        Args:
            embedding (Tensor): Embedding vector extracted from the test set.
            mean (Tensor): Mean of the multivariate gaussian distribution
            inv_covariance (Tensor): Inverse Covariance matrix of the multivariate gaussian distribution.

        Returns:
            Output anomaly score.
        """

        score_map, max_activation_val = self.compute_distance(
            embedding=embedding,
            stats=[mean.to(embedding.device), inv_covariance.to(embedding.device)],
        )
        up_sampled_score_map = self.up_sample(score_map)
        smoothed_anomaly_map = self.smooth_anomaly_map(up_sampled_score_map)

        return (smoothed_anomaly_map, max_activation_val)

    def __call__(self, **kwds):
        """Returns anomaly_map.

        Expects `embedding`, `mean` and `covariance` keywords to be passed explicitly.

        Example:
        >>> anomaly_map_generator = AnomalyMapGenerator(image_size=input_size)
        >>> output = anomaly_map_generator(embedding=embedding, mean=mean, covariance=covariance)

        Raises:
            ValueError: `embedding`. `mean` or `covariance` keys are not found

        Returns:
            torch.Tensor: anomaly map
        """

        if not ("embedding" in kwds and "mean" in kwds and "inv_covariance" in kwds):
            raise ValueError(f"Expected keys `embedding`, `mean` and `covariance`. Found {kwds.keys()}")

        embedding: Tensor = kwds["embedding"]
        mean: Tensor = kwds["mean"]
        inv_covariance: Tensor = kwds["inv_covariance"]

        return self.compute_anomaly_map(embedding, mean, inv_covariance)


class PadimLightning(AnomalyModule):
    """PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.

    Args:
        hparams (Union[DictConfig, ListConfig]): Model params
    """

    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__(hparams)
        self.layers = hparams.model.layers
        self.model = PadimModel(
            layers=hparams.model.layers,
            input_size=hparams.model.input_size,
            tile_size=hparams.dataset.tiling.tile_size,
            tile_stride=hparams.dataset.tiling.stride,
            apply_tiling=hparams.dataset.tiling.apply,
            backbone=hparams.model.backbone,
        ).eval()

        self.stats: List[Tensor] = []
        self.automatic_optimization = False

    @staticmethod
    def configure_optimizers():
        """PADIM doesn't require optimization, therefore returns no optimizers."""
        return None

    def training_step(self, batch, _):  # pylint: disable=arguments-differ
        """Training Step of PADIM. For each batch, hierarchical features are extracted from the CNN.

        Args:
            batch (Dict[str,Tensor]): Input batch
            _: Index of the batch.

        Returns:
            Hierarchical feature map
        """

        self.model.feature_extractor.eval()
        embeddings = self.model(batch["image"])
        return {"embeddings": embeddings.cpu()}

    def training_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        """Fit a multivariate gaussian model on an embedding extracted from deep hierarchical CNN features.

        Args:
            outputs (List[Dict[str, Tensor]]): Batch of outputs from the training step

        Returns:
            None
        """

        embeddings = torch.vstack([x["embeddings"] for x in outputs])
        self.stats = self.model.gaussian.fit(embeddings)

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
        map, max_activation_val = self.model(batch["image"])
        batch["anomaly_maps"] = map
        batch["max_activation_val"] = max_activation_val

        return batch

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        """Validation epoch end for padim

        Compute threshold and metrics by parent class & perform class feature modelling.

        Args:
            outputs (List[Dict[str, Tensor]]): Batch of outputs from the validation step

        Returns:
            None
        """
        super().validation_epoch_end(outputs)
        max_val_features = torch.vstack([x["max_activation_val"] for x in outputs])
        class_labels = np.hstack([x["class"] for x in outputs])
        #print(len(class_labels))
        self.class_stats = self.model.feature_model.fit(max_val_features, class_labels)

    def test_epoch_end(self, outputs) -> None:
        class_stats = {}
        max_activation_val = torch.vstack([x["max_activation_val"] for x in outputs])
        class_labels = np.hstack([x["class"] for x in outputs])

        class_names = np.unique(class_labels)
        #print(class_labels)
        #print(max_activation_val.shape)

        results={}
        for class_name in class_names:
            results[class_name] = []
        # sorted values and idx for entire feature set
        max_val, max_idx = torch.sort(max_activation_val, descending=True)
        reduced_range = int(max_val.shape[1] * 0.10)
        top_max_val = max_val[:, 0:reduced_range]
        # indexes of top 10% FEATURES HAVING MAX VALUE
        top_max_idx = max_idx[:, 0:reduced_range]
        correct_class = []
        for idx,feature in enumerate(top_max_idx):
            scores=[]
            for class_name in class_names:
                #print(class_name)
                stats = getattr(self.model.feature_model, class_name).cpu()
                score = stats[1][np.isin(stats[0],feature)].sum()
                scores.append(score)
            scores[np.where(class_names=='good')[0][0]]=0
            if 'combined' in class_names:
                scores[np.where(class_names=='combined')[0][0]]=0
            if 'thread' in class_names:
                scores[np.where(class_names=='thread')[0][0]]=0
            predicted_class = class_names[scores.index(max(scores))]
            if class_labels[idx] not in  ['good','thread','combined']:
                #print(f"predicted: {predicted_class}, actual: {class_labels[idx]}")
                if predicted_class == class_labels[idx]:
                    correct_class.append(1)
                    results[class_labels[idx]].append(1)
                else:
                    correct_class.append(0)
                    results[class_labels[idx]].append(0)
        print("*********************")
        print(class_names)
        for class_name in class_names:
            if class_name not in  ['good','thread','combined']:
                print(f"{class_name} accuracy: {sum(results[class_name])/len(results[class_name])}")
        print(f"average accuracy: {sum(correct_class)/len(correct_class)}")

        super().test_epoch_end(outputs)