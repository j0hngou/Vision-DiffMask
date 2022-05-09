import pytorch_lightning as pl
import torch

from .gates import DiffMaskGateInput
from optimizer import LookaheadRMSprop
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
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
        # TODO: check arguments & modify DiffMask signature
        self.diff_mask = DiffMaskGateInput(
            hidden_size=self.model.config.hidden_size,
            hidden_attention=self.model.config.hidden_size // 4,
            max_position_embeddings=1,
            num_hidden_layers=self.model.config.num_hidden_layers,
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
        expected_L0_full = self.diff_mask(hidden_states=hidden_states, layer_pred=None)

        return expected_L0_full

    def training_step(self, batch: Tuple[Tensor, Tensor], *args, **kwargs) -> dict:
        images, _ = batch

        # Pass original image from ViT and collect logits & hidden states
        outputs = self.model(images, output_hidden_states=True)
        logits_unmasked, hidden_states_unmasked = outputs.logits, outputs.hidden_states

        # Forward hidden states through DiffMask
        deepest_layer = torch.randint(len(hidden_states_unmasked), ()).item()
        hidden_states_masked, _, expected_L0, _, _, = self.diff_mask(
            hidden_states=hidden_states_unmasked,
            layer_pred=deepest_layer,
        )

        # Set new hidden states in the ViT and forward again
        hidden_states_masked = [hidden_states_masked] + [None] * (len(hidden_states_unmasked) - 1)
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

        # TODO: Log Metrics

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
                        "params": self.diff_mask.placeholder.parameters()
                        if isinstance(
                            self.diff_mask.placeholder, torch.nn.ParameterList
                        )
                        else [self.diff_mask.placeholder],
                        "lr": self.hparams.lr["beta"],
                    },
                ],
                centered=True,
            ),
            LookaheadRMSprop(
                params=[self.lamda]
                if isinstance(self.lamda, torch.Tensor)
                else self.lamda.parameters(),
                lr=self.hparams.lr["lamda"],
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
        if optimizer_idx == 0:
            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()
            for g in optimizer.param_groups:
                for p in g["params"]:
                    p.grad = None

        elif optimizer_idx == 1:
            for i in range(len(self.lamda)):
                if self.lamda[i].grad:
                    self.lamda[i].grad *= -1

            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()
            for g in optimizer.param_groups:
                for p in g["params"]:
                    p.grad = None

            for i in range(len(self.lamda)):
                self.lamda[i].data = torch.where(
                    self.lamda[i].data < 0,
                    torch.full_like(self.lamda[i].data, 0),
                    self.lamda[i].data,
                )
                self.lamda[i].data = torch.where(
                    self.lamda[i].data > 200,
                    torch.full_like(self.lamda[i].data, 200),
                    self.lamda[i].data,
                )
