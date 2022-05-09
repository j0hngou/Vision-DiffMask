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
from utils.metrics import accuracy_precision_recall_f1


class ImageInterpretationNet(pl.LightningModule):
    def __init__(
        self,
        model: ViTForImageClassification,
        lr: float = 3e-4,
        eps: float = 0.1,
        eps_valid: float = 0.8,
        acc_valid: float = 0.75,
        lr_placeholder: float = 1e-3,
        lr_alpha: float = 0.3,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False

        self.gate = DiffMaskGateInput(
            hidden_size=model.config.hidden_size,
            hidden_attention=model.config.hidden_size // 4,
            max_position_embeddings=1,
            num_hidden_layers=model.config.num_hidden_layers,
        )

        self.alpha = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.ones(()))
                for _ in range(self.model.config.num_hidden_layers + 2)
            ]
        )

        self.register_buffer(
            "running_acc", torch.ones((self.model.config.num_hidden_layers + 2,))
        )
        self.register_buffer(
            "running_l0", torch.ones((self.model.config.num_hidden_layers + 2,))
        )
        self.register_buffer(
            "running_steps", torch.zeros((self.model.config.num_hidden_layers + 2,))
        )

    def forward_explainer(
        self, x: Tensor, attribution: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, int]]:
        outputs = self.model(x, output_hidden_states=True)
        logits_orig, hidden_states = outputs.logits, outputs.hidden_states

        # TODO: deal with these
        # layer_pred = torch.randint(len(hidden_states), ()).item()
        layer_pred = 4
        layer_drop = 0

        (
            new_hidden_state,
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
        ) = self.gate(
            hidden_states=hidden_states,
            layer_pred=None
            if attribution
            else layer_pred,  # if attribution, we get all the hidden states
        )

        if attribution:
            return expected_L0_full
        else:
            new_hidden_states = (
                [None] * layer_drop
                + [new_hidden_state]
                + [None] * (len(hidden_states) - layer_drop - 1)
            )

            logits, _ = vit_setter(self.model, x, new_hidden_states)

        return (
            logits,
            logits_orig,
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
            layer_drop,
            layer_pred,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).logits

    def training_step(self, batch: Tuple[Tensor, Tensor], *args, **kwargs) -> dict:
        x, y = batch

        (
            logits,
            logits_orig,
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
            layer_drop,
            layer_pred,
        ) = self.forward_explainer(x)

        loss_c = (
            torch.distributions.kl_divergence(
                torch.distributions.Categorical(logits=logits_orig),
                torch.distributions.Categorical(logits=logits),
            )
            - self.hparams.eps
        )

        loss_g = expected_L0.mean(-1)

        loss = self.alpha[layer_pred] * loss_c + loss_g

        acc, _, _, _ = accuracy_precision_recall_f1(
            logits.argmax(-1), logits_orig.argmax(-1), average=True
        )

        l0 = expected_L0.exp().mean(-1)

        outputs_dict = {
            "loss_c": loss_c.mean(-1),
            "loss_g": loss_g.mean(-1),
            "alpha": self.alpha[layer_pred].mean(-1),
            "acc": acc,
            "l0": l0.mean(-1),
            "layer_pred": layer_pred,
            "r_acc": self.running_acc[layer_pred],
            "r_l0": self.running_l0[layer_pred],
            "r_steps": self.running_steps[layer_pred],
            "debug_loss": loss.mean(-1),
        }

        outputs_dict = {
            "loss": loss.mean(-1),
            **outputs_dict,
            "log": outputs_dict,
            "progress_bar": outputs_dict,
        }

        self.log('loss', outputs_dict['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('loss_c', outputs_dict['loss_c'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('loss_g', outputs_dict['loss_g'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('acc', outputs_dict['acc'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('l0', outputs_dict['l0'], on_step=True, on_epoch=True, prog_bar=True)

        outputs_dict = {
            "{}{}".format("" if self.training else "val_", k): v
            for k, v in outputs_dict.items()
        }

        if self.training:
            self.running_acc[layer_pred] = (
                self.running_acc[layer_pred] * 0.9 + acc * 0.1
            )
            self.running_l0[layer_pred] = (
                self.running_l0[layer_pred] * 0.9 + l0.mean(-1) * 0.1
            )
            self.running_steps[layer_pred] += 1

        return outputs_dict

    def validation_epoch_end(self, outputs: List[dict]):
        outputs_dict = {
            k: [e[k] for e in outputs if k in e]
            for k in ("val_loss_c", "val_loss_g", "val_acc", "val_l0")
        }

        outputs_dict = {k: sum(v) / len(v) for k, v in outputs_dict.items()}

        outputs_dict["val_loss_c"] += self.hparams.eps

        outputs_dict = {
            "val_loss": outputs_dict["val_l0"]
            if outputs_dict["val_loss_c"] <= self.hparams.eps_valid
            and outputs_dict["val_acc"] >= self.hparams.acc_valid
            else torch.full_like(outputs_dict["val_l0"], float("inf")),
            **outputs_dict,
            "log": outputs_dict,
        }

        return outputs_dict

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        optimizers = [
            LookaheadRMSprop(
                params=[
                    {"params": self.gate.g_hat.parameters(), "lr": self.hparams.lr},
                    {
                        "params": self.gate.placeholder.parameters()
                        if isinstance(self.gate.placeholder, torch.nn.ParameterList)
                        else [self.gate.placeholder],
                        "lr": self.hparams.lr_placeholder,
                    },
                ],
                centered=True,
            ),
            LookaheadRMSprop(
                params=[self.alpha]
                if isinstance(self.alpha, torch.Tensor)
                else self.alpha.parameters(),
                lr=self.hparams.lr_alpha,
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
            for i in range(len(self.alpha)):
                if self.alpha[i].grad:
                    self.alpha[i].grad *= -1

            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()
            for g in optimizer.param_groups:
                for p in g["params"]:
                    p.grad = None

            for i in range(len(self.alpha)):
                self.alpha[i].data = torch.where(
                    self.alpha[i].data < 0,
                    torch.full_like(self.alpha[i].data, 0),
                    self.alpha[i].data,
                )
                self.alpha[i].data = torch.where(
                    self.alpha[i].data > 200,
                    torch.full_like(self.alpha[i].data, 200),
                    self.alpha[i].data,
                )
