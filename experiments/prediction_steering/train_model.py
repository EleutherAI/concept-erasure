from argparse import ArgumentParser
from dataclasses import dataclass

import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from models import ConvNext, Mlp, MlpMixer, ResNet, Swin, ViT
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10

# Use faster matmul precision
torch.set_float32_matmul_precision("high")


@dataclass
class LogSpacedCheckpoint(Callback):
    """Save checkpoints at log-spaced intervals"""

    dirpath: str

    base: float = 2.0
    next: int = 1

    def on_train_batch_end(self, trainer: Trainer, *_):
        if trainer.global_step >= self.next:
            self.next = round(self.next * self.base)
            trainer.save_checkpoint(self.dirpath + f"/step={trainer.global_step}.ckpt")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument(
        "--patch-size", type=int, default=4, help="patch size for mixer and resmlp"
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=("convnext", "mixer", "resmlp", "resnet", "swin", "vit"),
    )
    args = parser.parse_args()

    final = nn.Flatten(0) if args.net == "resmlp" else nn.Identity()
    train_trf = T.Compose(
        [
            T.RandAugment(),
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.ToTensor(),
            final,
        ]
    )

    # Split the "train" set into train and validation
    train = CIFAR10(
        "/home/nora/Data/cifar10-train", download=True, train=True, transform=train_trf
    )

    # Test set is entirely separate
    nontrain = CIFAR10(
        "/home/nora/Data/cifar10-test",
        download=True,
        train=False,
        transform=T.ToTensor(),
    )
    torch.manual_seed(0)
    val, test = random_split(nontrain, [0.1, 0.9])

    k = 10  # Number of classes

    # Create the data module
    dm = pl.LightningDataModule.from_datasets(
        train, val, test, batch_size=128, num_workers=8
    )

    model_cls = {
        "convnext": ConvNext,
        "mixer": MlpMixer,
        "resmlp": Mlp,
        "resnet": ResNet,
        "swin": Swin,
        "vit": ViT,
    }[args.net]
    model = model_cls(k, patch_size=args.patch_size)

    trainer = pl.Trainer(
        callbacks=[
            LearningRateMonitor(),
            LogSpacedCheckpoint(f"/home/nora/Data/cifar-ckpts/{args.name}"),
        ],
        logger=WandbLogger(
            name=args.name, project="concept-erasure", entity="eleutherai"
        ),
        max_steps=2**16,
        precision="16-mixed",
    )
    trainer.fit(model, dm)
    trainer.test(model, dm)
