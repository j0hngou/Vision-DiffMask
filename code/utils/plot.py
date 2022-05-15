import cv2
import numpy as np
import torch

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer import Trainer
from torch import Tensor


def unnormalize(
    images: Tensor,
    mean: tuple[int] = (0.5, 0.5, 0.5),
    std: tuple[int] = (0.5, 0.5, 0.5),
) -> Tensor:
    """Reverts the normalization transformation applied before ViT.

    Args:
        images (Tensor): a batch of images.
        mean (tuple[int], optional): The means used for normalization. Defaults to (0.5, 0.5, 0.5).
        std (tuple[int], optional): The stds used for normalization.. Defaults to (0.5, 0.5, 0.5).

    Returns:
        Tensor: the batch of images unnormalized.
    """
    unnormalized_images = images.clone()
    mean, std = [0.5], [0.5]
    for i, (m, s) in enumerate(zip(mean, std)):
        unnormalized_images[:, i, :, :].mul_(s).add_(m)

    return unnormalized_images


def draw_mask_on_image(image: Tensor, mask: Tensor) -> Tensor:
    """This function overlays a dimming mask on the image.

    Args:
        image (Tensor): a float torch tensor in [0, 1].
        mask (Tensor): a float torch tensor in [0, 1].

    Returns:
        The default image with the cam overlay.
    """
    masked_image = image * mask

    return masked_image


def draw_heatmap_on_image(
    image: Tensor, mask: Tensor, colormap: int = cv2.COLORMAP_JET
) -> Tensor:
    """This function overlays a heatmap on the image.

    Args:
        image (Tensor): a float torch tensor in [0, 1].
        mask (Tensor): a float torch tensor in [0, 1].
        colormap (int): the OpenCV colormap to be used.

    Returns:
        The default image with the cam overlay.
    """
    # Save the device of the image
    original_device = image.device

    # Convert image & mask to numpy
    image = image.permute(1, 2, 0).cpu().numpy()
    mask = mask.cpu().numpy()

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    # Overlay heatmap on image
    masked_image = image + heatmap
    masked_image = masked_image / np.max(masked_image)

    return torch.tensor(masked_image).permute(2, 0, 1).to(original_device)


class DrawMaskCallback(Callback):
    def __init__(self, sample_images: Tensor, log_every_n_steps: int = 200):
        self.sample_images = unnormalize(sample_images)
        self.log_every_n_steps = log_every_n_steps
        self.batches = 0

    def _log_masks(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Predict mask
        with torch.no_grad():
            pl_module.eval()
            masks = pl_module.get_mask(self.sample_images)
            pl_module.train()

        # Draw mask on sample images
        sample_images = [image for image in self.sample_images]

        sample_images_with_mask = [
            draw_mask_on_image(image, mask) for image, mask in zip(sample_images, masks)
        ]

        sample_images_with_heatmap = [
            draw_heatmap_on_image(image, mask)
            for image, mask in zip(sample_images, masks)
        ]
        
        # Merge sample images into one image
        samples = torch.cat([
            torch.cat(sample_images, dim=2),
            torch.cat(sample_images_with_mask, dim=2),
            torch.cat(sample_images_with_heatmap, dim=2),
        ], dim=1)

        # Compute masking percentage
        masked_pixels_percentage = 100 * (1 - masks.mean().item())

        # Log with wandb
        trainer.logger.log_image(
            key="Masked images",
            images=[samples],
            caption=[f"Percentage of masked pixels: {masked_pixels_percentage}"],
        )

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Transfer sample images to correct device
        self.sample_images = self.sample_images.to(pl_module.device)
        
        # Log sample images
        self._log_masks(trainer, pl_module)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        # Log sample images every n steps
        self.batches += 1
        if self.batches % self.log_every_n_steps == 0:
            self._log_masks(trainer, pl_module)
