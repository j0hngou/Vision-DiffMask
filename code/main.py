import argparse
import pytorch_lightning as pl
import torch

from datamodules import CIFAR10DataModule
from models.interpretation import ImageInterpretationNet
from transformers import ViTFeatureExtractor, ViTForImageClassification, \
    ConvNextForImageClassification, ConvNextFeatureExtractor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utils.plot import DrawMaskCallback


def get_experiment_name(args: argparse.Namespace):
    # Convert to dictionary
    args = vars(args)

    # Create a list with non-experiment arguments
    non_experiment_args = [
        "add_blur",
        "add_noise",
        "add_rotation",
        "batch_size",
        "data_dir",
        "enable_progress_bar",
        "log_every_n_steps",
        "num_epochs",
        "num_workers",
        "sample_images",
        "seed",
        "vit_model",
        "convnext_model",
    ]

    # Create experiment name from experiment arguments
    return "-".join(
        [
            f"{name}={value}"
            for name, value in sorted(args.items())
            if name not in non_experiment_args
        ]
    )


def main(args: argparse.Namespace):
    # Seed
    pl.seed_everything(args.seed)

    # Load model and feature extractor
    if args.model == "ViT":
        model = ViTForImageClassification.from_pretrained(args.vit_model)
        feature_extractor = ViTFeatureExtractor.from_pretrained(
            args.vit_model, return_tensors="pt"
        )
    elif args.model == "ConvNeXt":
        model = ConvNextForImageClassification.from_pretrained(args.convnext_model)
        feature_extractor = ConvNextFeatureExtractor.from_pretrained(
            args.convnext_model, return_tensors="pt"
        )
    else:
        return

    # Load CIFAR10 datamodule
    dm = CIFAR10DataModule(
        batch_size=args.batch_size,
        feature_extractor=feature_extractor,
        noise=args.add_noise,
        rotation=args.add_rotation,
        blur=args.add_blur,
        num_workers=args.num_workers,
    )

    # Setup datamodule to sample images for the mask callback
    dm.prepare_data()
    dm.setup("fit")

    # Create Vision DiffMask for the model
    diffmask = ImageInterpretationNet(
        model_cfg=model.config,
        alpha=args.alpha,
        lr=args.lr,
        eps=args.eps,
        lr_placeholder=args.lr_placeholder,
        lr_alpha=args.lr_alpha,
        mul_activation=args.mul_activation,
        add_activation=args.add_activation,
        placeholder=not args.no_placeholder,
        weighted_layer_pred=args.weighted_layer_distribution,
    )
    diffmask.set_model(model)

    # Create wandb logger instance
    wandb_logger = WandbLogger(
        name=get_experiment_name(args),
        project="Patch-DiffMask",
    )

    # Create checkpoint callback
    ckpt_cb = ModelCheckpoint(
        # TODO: add more args (probably monitor some metric)
        dirpath=f"checkpoints/{wandb_logger.version}",
        every_n_train_steps=args.log_every_n_steps,
    )

    # Sample images & create mask callback
    sample_images = torch.stack([dm.val_data[i][0] for i in range(8)])
    mask_cb1 = DrawMaskCallback(sample_images, log_every_n_steps=args.log_every_n_steps, key='1')
    sample_images = torch.stack([dm.val_data[i][0] for i in range(8, 16)])
    mask_cb2 = DrawMaskCallback(sample_images, log_every_n_steps=args.log_every_n_steps, key='2')
    sample_images = torch.stack([dm.val_data[i][0] for i in range(16, 24)])
    mask_cb3 = DrawMaskCallback(sample_images, log_every_n_steps=args.log_every_n_steps, key='3')

    # Train
    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=[ckpt_cb, mask_cb1, mask_cb2, mask_cb3],
        enable_progress_bar=args.enable_progress_bar,
        logger=wandb_logger,
        max_epochs=args.num_epochs,
    )

    trainer.fit(diffmask, dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Trainer
    parser.add_argument(
        "--enable_progress_bar",
        action="store_true",
        help="Whether to enable the progress bar (NOT recommended when logging to file).",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility.",
    )
    
    # Logging
    parser.add_argument(
        "--sample_images",
        type=int,
        default=8,
        help="Number of images to sample for the mask callback.",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=200,
        help="Number of steps between logging media & checkpoints.",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="ViT",
        choices=["ViT", "ConvNeXt"],
        help="Model to use.",
    )

    # Classification model
    parser.add_argument(
        "--vit_model",
        type=str,
        default="tanlq/vit-base-patch16-224-in21k-finetuned-cifar10",
        help="Pre-trained Vision Transformer (ViT) model to load.",
    )

    parser.add_argument(
        "--convnext_model",
        default="convnext_cifar10",
        type=str,
        help="Pre-trained ConvNeXt model to load.",
    )

    # Interpretation model
    parser.add_argument(
        "--alpha",
        type=float,
        default=5.0,
        help="Intial value for the Lagrangian",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate for diffmask.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.1,
        help="KL divergence tolerance.",
    )
    parser.add_argument(
        "--no_placeholder",
        action="store_true",
        help="Whether to not use placeholder",
    )
    parser.add_argument(
        "--lr_placeholder",
        type=float,
        default=1e-3,
        help="Learning for mask vectors.",
    )
    parser.add_argument(
        "--lr_alpha",
        type=float,
        default=0.3,
        help="Learning rate for lagrangian optimizer.",
    )
    parser.add_argument(
        "--mul_activation",
        type=float,
        default=15.0,
        help="Value to mutliply gate activations.",
    )
    parser.add_argument(
        "--add_activation",
        type=float,
        default=4.0,
        help="Value to add to gate activations.",
    )
    parser.add_argument(
        "--weighted_layer_distribution",
        action="store_true",
        help="Whether to use a weighted distribution when picking a layer in DiffMask forward.",
    )

    # Datamodule
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="The batch size to use.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        help="The dataset to use.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/",
        help="The data directory to use.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="The number of workers to use.",
    )
    parser.add_argument(
        "--add_noise",
        action="store_true",
        help="Use gaussian noise augmentation.",
    )
    parser.add_argument(
        "--add_rotation",
        action="store_true",
        help="Use rotation augmentation.",
    )
    parser.add_argument(
        "--add_blur",
        action="store_true",
        help="Use blur augmentation.",
    )

    args = parser.parse_args()

    main(args)
