"""
Grad-CAM explanation utility for SpectraShield.

Generates a Grad-CAM heatmap from the final convolutional
layer (features[12]) of SpectraShieldCNN.

The explanation targets the FAKE logit.
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GradCAM:
    """Grad-CAM implementation for SpectraShieldCNN."""

    def __init__(self, model):
        self.model = model
        self.model.eval()

        self.activations = None
        self.gradients = None

        # Final convolutional layer: Conv2d(128, 256) at features[12]
        self.target_layer = self.model.features[12]

        # Register hooks
        self.forward_handle = (
            self.target_layer.register_forward_hook(
                self._forward_hook
            )
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                self._backward_hook
            )
        )

    def _forward_hook(self, module, inputs, output):
        """Store feature-map activations."""
        self.activations = output

    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):
        """Store gradients flowing through the target layer."""
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        """
        Generate a normalized Grad-CAM heatmap.

        Args:
            input_tensor:
                Tensor of shape [1, 1, 128, 128].

        Returns:
            numpy.ndarray:
                Heatmap of shape [128, 128],
                normalized to [0, 1].
        """

        if input_tensor.ndim != 4:
            raise ValueError(
                "Grad-CAM input must have shape "
                "[batch, channels, height, width]."
            )

        if input_tensor.shape[0] != 1:
            raise ValueError(
                "Grad-CAM currently supports batch size 1."
            )

        self.model.eval()

        # Clear previous values
        self.activations = None
        self.gradients = None

        # Clear model gradients
        self.model.zero_grad(set_to_none=True)

        # Grad-CAM requires gradients
        with torch.enable_grad():

            output = self.model(
                input_tensor
            )

            # Model has one output:
            # the FAKE logit
            fake_logit = output[:, 0]

            # Backward pass
            fake_logit.backward()

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        # ----------------------------------------------------
        # Grad-CAM calculation
        # ----------------------------------------------------

        activations = self.activations
        gradients = self.gradients

        # Global average pooling over spatial dimensions
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted combination of feature maps
        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        # Keep only positive influence
        cam = F.relu(cam)

        # Resize to input resolution
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # Remove batch/channel dimensions
        heatmap = (
            cam[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        # Normalize to [0, 1]
        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()

        if heatmap_max - heatmap_min > 1e-8:

            heatmap = (
                heatmap - heatmap_min
            ) / (
                heatmap_max - heatmap_min
            )

        else:

            heatmap = np.zeros_like(
                heatmap,
                dtype=np.float32
            )

        return heatmap.astype(
            np.float32
        )

    def close(self):
        """Remove registered hooks."""

        self.forward_handle.remove()
        self.backward_handle.remove()

    def __del__(self):
        """Clean up hooks when object is destroyed."""

        try:
            self.close()
        except Exception:
            pass