import pytorch_lightning as pl
import torch

from .gates import DiffMaskGateInput
from math import sqrt
from optimizer import LookaheadRMSprop
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import functional as TF
from transformers import (
    get_constant_schedule_with_warmup,
    get_constant_schedule,
    ViTForImageClassification,
)
from typing import Callable, List, Optional, Tuple, Union
from utils.setters import vit_setter


class ImageInterpretationNet(pl.LightningModule):
    """Applies DiffMask to a Vision Transformer (ViT) model.

    This module consists of a pre-trained ViT model and a list of DiffMask gates. The
    input is passed through the ViT and the hidden states are collected. Then the hidden
    states are passed as input to DiffMask, which returns a predicted mask for the input.

    Args:
        model (ViTForImageClassification): a ViT from Hugging Face
        lr (dict): a dictionary with learning rates for diffmask, beta and lambda
        mu (float): the tolorance of the KL divergence between y and y_hat
    """

    def __init__(
        self,
        model: ViTForImageClassification,
        lr: dict[str, float] = {"diff_mask": 3e-4, "beta": 1e-3, "lamda": 3e-1},
        mu: float = 0.1,
    ):
        assert (
            "diff_mask" in lr and "beta" in lr and "lamda" in lr
        ), "lr dictionary must contain diff_mask, beta and lamda"

        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        # Freeze ViT's parameters
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False

        # Create DiffMask
        # TODO: check arguments & modify DiffMask signature & num layers
        self.diff_mask = DiffMaskGateInput(
            hidden_size=self.model.config.hidden_size,
            hidden_attention=self.model.config.hidden_size // 4,
            max_position_embeddings=1,
            num_hidden_layers=self.model.config.num_hidden_layers+1,
        )

        # Create Lagrangian for dual optimization
        self.lamda = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.ones(()))]
            * (self.model.config.num_hidden_layers + 2)
        )

    def forward(self, x: Tensor) -> Tensor:
        # Forward input through freezed ViT & collect hidden states
        hidden_states = self.model(x, output_hidden_states=True).hidden_states

        # Forward hidden states through DiffMask
        log_expected_L0 = self.diff_mask(hidden_states=hidden_states, layer_pred=None)[-1]
        
        # Calculate mask
        mask = log_expected_L0.sum(-1).exp()
        mask = mask[:, 1:]
        
        # Reshape mask to match input shape
        B, C, H, W = x.shape    # batch, channels, height, width
        B, P = mask.shape       # batch, patches
        
        N = int(sqrt(P))    # patches per side
        S = int(H / N)      # patch size
        
        mask = mask.reshape(B, 1, N, N)
        mask = mask.repeat(1, C, S, S)

        return mask

    def training_step(self, batch: Tuple[Tensor, Tensor], *args, **kwargs) -> dict:
        images, labels = batch

        # Pass original image from ViT and collect logits & hidden states
        outputs = self.model(images, output_hidden_states=True)
        logits_unmasked, hidden_states_unmasked = outputs.logits, outputs.hidden_states

        # Forward hidden states through DiffMask
        # TODO: check why out-of-bounds without -1
        deepest_layer = torch.randint(len(hidden_states_unmasked) - 1, ()).item()
        hidden_states_masked, _, expected_L0, _, _, = self.diff_mask(
            hidden_states=hidden_states_unmasked,
            layer_pred=deepest_layer,
        )

        # Set new hidden states in the ViT and forward again
        hidden_states_masked = [hidden_states_masked] + [None] * (
            len(hidden_states_unmasked) - 1
        )
        logits_masked, _ = vit_setter(self.model, images, hidden_states_masked)

        # Calculate loss
        loss_c = (
            torch.distributions.kl_divergence(
                torch.distributions.Categorical(logits=logits_unmasked),
                torch.distributions.Categorical(logits=logits_masked),
            )
            - self.hparams.mu
        )

        loss_g = expected_L0.mean(-1)

        loss = torch.mean(self.lamda[deepest_layer] * loss_c + loss_g, dim=-1)

        # Log Metrics
        self.log("train/loss", loss)
        self.log("train/loss_c", loss_c.mean())
        self.log("train/loss_g", loss_g.mean())
        self.log("train/lamda", self.lamda[deepest_layer].mean())
        self.log("trian/expected_L0", expected_L0.exp().sum(-1).mean())
        self.log("train/acc", TF.accuracy(logits_masked, labels))
        self.log("train/deepest_layer", deepest_layer)

        return loss

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        optimizers = [
            LookaheadRMSprop(
                params=[
                    {
                        "params": self.diff_mask.g_hat.parameters(),
                        "lr": self.hparams.lr["diff_mask"],
                    },
                    {
                        "params": [self.diff_mask.placeholder],
                        "lr": self.hparams.lr["beta"],
                    },
                ],
                centered=True,
            ),
            LookaheadRMSprop(
                params=self.lamda.parameters(), lr=self.hparams.lr["lamda"]
            ),
        ]

        schedulers = [
            {
                "scheduler": get_constant_schedule_with_warmup(optimizers[0], 12 * 100),
                "interval": "step",
            },
            get_constant_schedule(optimizers[1]),
        ]

        return optimizers, schedulers

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[Optimizer, LightningOptimizer],
        optimizer_idx: int = 0,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ):
        # Optimizer 0: Minimize loss w.r.t DiffMask's parameters
        if optimizer_idx == 0:
            # Gradient ascent on phi & beta
            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()
            for g in optimizer.param_groups:
                for p in g["params"]:
                    p.grad = None

        # Optimizer 1: Maximize loss w.r.t. langrangian lamda
        elif optimizer_idx == 1:
            # Reverse the sign of lamda's gradients
            for i in range(len(self.lamda)):
                if self.lamda[i].grad:
                    self.lamda[i].grad *= -1

            # Gradient ascent on lamda
            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()
            for g in optimizer.param_groups:
                for p in g["params"]:
                    p.grad = None

            # Clip lamda
            for i in range(len(self.lamda)):
                self.lamda[i].data.clamp_(0, 200)
