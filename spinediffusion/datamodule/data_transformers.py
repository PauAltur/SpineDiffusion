import torch.nn as nn
from torchvision.transforms import v2

from .transforms.augmenting import RandomRotationAugmentation
from .transforms.closing import Closing
from .transforms.normalizing import ConstantNormalization, SpineLengthNormalization
from .transforms.projecting import ProjectToPlane
from .transforms.resampling import Resample3DCurve, ResamplePointCloud
from .transforms.tensoring import Tensorize

TRANSFORMS = {
    "constant_normalize": ConstantNormalization,
    "spine_length_normalize": SpineLengthNormalization,
    "resample_3d_curve": Resample3DCurve,
    "project_to_plane": ProjectToPlane,
    "resample_point_cloud": ResamplePointCloud,
    "close_depthmap": Closing,
    "tensorize": Tensorize,
    "random_rotation": RandomRotationAugmentation,
}


class SpinePreprocessor(nn.Module):
    """Preprocesses spine data using a series of transforms."""

    def __init__(self, transform_dict: dict):
        """Initialize the SpinePreprocessor.

        Args:
            transform_dict (dict): A dictionary of transforms to apply to the data.
                It has keys with the transform names, that must correspond to the keys
                in the TRANSFORMS dictionary, and values with the parameters to instantiate
                the transform + a "transform_number" subkey to order the transform.
        """
        super().__init__()
        self.transform_dict = transform_dict

    def _instantiate_transforms(self):
        """Instantiate the transforms from the transform dictionary in order."""
        transforms = []

        for i in range(len(self.transform_dict)):
            for key, value in self.transform_dict.items():
                if value["transform_number"] == i:
                    transform = TRANSFORMS[key](**value["params"])
                    transforms.append(transform)

        self.transforms = v2.Compose(transforms)

    def forward(self, data_id: dict) -> dict:
        """Apply the transforms to the data_id dictionary.

        Args:
            data_id (dict): The data to preprocess.

        Returns:
            dict: The preprocessed data.
        """
        return self.transforms(data_id)


class SpineAugmenter(nn.Module):
    """Augments spine data using a series of augmentations."""

    def __init__(self, augment_dict: dict):
        """Initialize the SpineAugmenter.

        Args:
            augment_dict (dict): A dictionary of augmentations to apply to the data.
                It has keys with the augmentation names, that must correspond to the keys
                in the TRANSFORMS dictionary, and values with the parameters to instantiate
                the augmentation + a "transform_number" subkey to order the augmentations
                and a "num_aug" subkey which determines how many augmented samples to compute per
                original sample and augmentation.
        """
        super().__init__()
        self.augment_dict = augment_dict

    def _instantiate_augmentations(self):
        """Instantiate the augmentations from the augment dictionary in order."""
        self.augmentations = []

        for i in range(len(self.augment_dict)):
            for key, value in self.augment_dict.items():
                if value["transform_number"] == i:
                    augment = TRANSFORMS[key](**value)
                    self.augmentations.append(augment)

    def forward(self, data_id: dict, unique_id: str) -> dict:
        """Apply the augmentations to the data_id dictionary.

        Args:
            data_id (dict): The sample to augment.
            unique_id (str): The unique identifier of the sample.

        Returns:
            data_id (dict): The augmented sample.
        """
        data_id_aug = {}
        aug_count = 0
        for augment in self.augmentations:
            for _ in range(augment.num_aug):
                aug_id = f"{unique_id}_{aug_count}"
                data_id_aug[aug_id] = augment(data_id)
                aug_count += 1

        return data_id_aug
