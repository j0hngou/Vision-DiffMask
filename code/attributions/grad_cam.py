import torch

from pytorch_grad_cam import GradCAM
from torch import Tensor
from transformers import ViTForImageClassification


def grad_cam(images: Tensor, vit: ViTForImageClassification, use_cuda: bool = False, target_layer_idx: int = -2,
             layernorm_after: bool = True) -> Tensor:
    """Performs the Grad-CAM method on a batch of images (https://arxiv.org/pdf/1610.02391.pdf)."""

    # Wrap the ViT model to be compatible with GradCAM
    vit = ViTWrapper(vit, target_layer_idx, layernorm_after)
    vit.eval()

    # Create GradCAM object
    cam = GradCAM(
        model=vit,
        target_layers=[vit.target_layer],
        reshape_transform=_reshape_transform,
        use_cuda=use_cuda,
    )

    # Compute GradCAM masks
    grayscale_cam = cam(
        input_tensor=images,
        targets=None,
        eigen_smooth=True,
        aug_smooth=True,
    )

    return torch.from_numpy(grayscale_cam)


def _reshape_transform(tensor):
    C, P, E = tensor.shape
    H = torch.sqrt(torch.tensor(P - 1)).int()
    W = torch.sqrt(torch.tensor(P - 1)).int()
    result = tensor[:, 1:, :].reshape(tensor.size(0), H, W, tensor.size(2))

    # Bring the channels to the first dimension
    result = result.transpose(2, 3).transpose(1, 2)

    return result


class ViTWrapper(torch.nn.Module):
    """ViT Wrapper to use with Grad-CAM."""

    def __init__(self, vit: ViTForImageClassification, target_layer_idx: int, layernorm_after: bool):
        super().__init__()
        self.vit = vit
        self.target_layer_idx = target_layer_idx
        self.layernorm_after = layernorm_after

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x).logits

    @property
    def target_layer(self):
        if self.layernorm_after:
            return self.vit.vit.encoder.layer[self.target_layer_idx].layernorm_after
        else:
            return self.vit.vit.encoder.layer[self.target_layer_idx].layernorm_before
