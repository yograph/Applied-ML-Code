"""
One instance of ConvNet
"""

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import torchvision.models


# torchvision.models.convnext_small(*, weights: Optional[ConvNeXt_Small_Weights] = None, progress: bool = True, **kwargs: Any) → ConvNeXt

class Creating_Convnet():
    """
    Class to create a single ConvNet model.
    """
    def creating_single_model(self) -> torchvision.models.ConvNeXt:
        """
        Creates a ConvNeXt small model with pre-trained weights.
        Returns:
            model: A ConvNeXt small model with pre-trained weights.
        """
        model = torchvision.models.convnext_small(weights=torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        return model