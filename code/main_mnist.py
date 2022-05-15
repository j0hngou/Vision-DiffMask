import argparse
import pytorch_lightning as pl
import torch

from datamodules import MNISTDataModule
from models.interpretation import ImageInterpretationNet
from models.classification import ImageClassificationNet
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utils.plot import DrawMaskCallback
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        "num_epochs",
        "num_workers",
        "vit_model",
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

    alphas = [1., 5., 10., 15., 20.]
    lrs = [1e-5, 1e-4, 1e-3, 1e-2]
    eps = [0, 0.001, 0.05, 0.1, 0.15]
    lrs_alpha = [0.3, 0.03, 0.003, 0.0003]
    mul_activations = [15., 10., 5., 3., 1.]
    add_activations = [15., 10., 5., 3., 1.]
    mnist_cfg = ViTConfig(image_size=112, num_channels=1, num_labels=10)
    mnist_fe = ViTFeatureExtractor(
        size=mnist_cfg.image_size,
        image_mean=[0.5],
        image_std=[0.5],
        return_tensors="pt",
    )
    vit = ViTForImageClassification(mnist_cfg)
    model = ImageClassificationNet(vit).load_from_checkpoint(
        checkpoint_path="/home/john/Desktop/MSc AI/DL2/Patch-DiffMask/checkpoints/MNIST/ImageClassificationNet_MNIST.ckpt",
    )
    model.model.save_pretrained('/home/john/Desktop/MSc AI/DL2/Patch-DiffMask/checkpoints/MNIST/ImageClassificationNet_MNISTtest.ckpt')
    # vit.load_state_dict(torch.load('/home/john/Desktop/MSc AI/DL2/Patch-DiffMask/checkpoints/MNIST/ImageClassificationNet_MNIST.ckpt'))
    # vit.save_pretrained('/home/john/Desktop/MSc AI/DL2/Patch-DiffMask/checkpoints/MNIST/ImageClassificationNet_MNISTtest.ckpt')
    exit()
    model = ImageClassificationNet(vit)
    # Load pre-trained Transformer
    checkpoint = torch.load(args.vit_ckpt)
    model.load_state_dict(checkpoint['state_dict'], map_location=device)
    model.eval()
    for alpha in alphas:
        for lr in lrs:
            for epsilon in eps:
                for lr_alpha in lrs_alpha:
                    for mul_activation in mul_activations:
                        for add_activation in add_activations:
                            pl.seed_everything(args.seed)



                            # Load MNIST datamodule
                            dm = MNISTDataModule(
                                batch_size=5,
                                feature_extractor=mnist_fe,
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
                            )
                            diffmask.set_vision_transformer(model)

                            # Create wandb logger instance
                            name = f"alpha={alpha}_lr={lr}_eps={epsilon}_lr_alpha={lr_alpha}_mul_activation={mul_activation}_add_activation={add_activation}"
                            wandb_logger = WandbLogger(
                                name=name,
                                project="Patch-DiffMask",
                            )

                            # Create checkpoint callback
                            ckpt_cb = ModelCheckpoint(
                                # TODO: add more args (probably monitor some metric)
                                dirpath=f"checkpoints/{wandb_logger.version}"
                            )

                            # Sample images & create mask callback
                            sample_images, _ = next(iter(dm.val_dataloader()))
                            mask_cb = DrawMaskCallback(sample_images)

                            # Train
                            trainer = pl.Trainer(
                                accelerator="auto",
                                callbacks=[ckpt_cb, mask_cb],
                                logger=wandb_logger,
                                max_epochs=args.num_epochs,
                            )

                            trainer.fit(diffmask, dm)
                            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Trainer
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility.",
    )

    # Classification model
    parser.add_argument(
        "--vit_model",
        type=str,
        default="tanlq/vit-base-patch16-224-in21k-finetuned-cifar10",
        help="Pre-trained Vision Transformer (ViT) model to load.",
    )

    parser.add_argument(
        "--vit_ckpt",
        type=str,
        default="../checkpoints/MNIST/ImageClassificationNet_MNIST.ckpt",
        help="Pre-trained Vision Transformer (ViT) model to load.",
    )

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/mnist",
        help="Directory containing MNIST dataset.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for dataloader.",
    )

    # Augmentation
    parser.add_argument(
        "--add_blur",
        action="store_true",
        help="Add Gaussian blur to images.",
    )
    parser.add_argument(
        "--add_noise",
        action="store_true",
        help="Add noise to images.",
    )
    parser.add_argument(
        "--add_rotation",
        action="store_true",
        help="Add random rotation to images.",
    )

    # Interpretation model
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Alpha parameter for the Vision DiffMask.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate for the Vision DiffMask.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.0,
        help="Epsilon parameter for the Vision DiffMask.",
    )
    parser.add_argument(
        "--lr_placeholder",
        type=float,
        default=1e-5,
        help="Learning rate for the placeholder.",
    )
    parser.add_argument(
        "--lr_alpha",
        type=float,
        default=0.3,
        help="Learning rate for the alpha parameter",
    )

    # Interpretation model
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Intial value for the Lagrangian",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for diffmask.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.1,
        help="KL divergence tolerance.",
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
        default=10.0,
        help="Value to mutliply gate activations.",
    )
    parser.add_argument(
        "--add_activation",
        type=float,
        default=5.0,
        help="Value to add to gate activations.",
    )

    # Datamodule
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="The batch size to use.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
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
