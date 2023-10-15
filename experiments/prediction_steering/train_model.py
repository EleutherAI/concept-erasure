from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sized

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from models import Mlp, MlpMixer, ResNet, ViT
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor, nn
from torch.utils.data import Dataset, random_split
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm

from concept_erasure import LeaceFitter, OracleFitter, QuadraticFitter

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


class LeacedDataset(Dataset):
    """Wrapper for a dataset of (X, Z) pairs that erases Z from X"""

    def __init__(
        self,
        inner: Dataset[tuple[Tensor, ...]],
        eraser: Callable,
        transform: Callable[[Tensor], Tensor] = lambda x: x,
        p: float = 1.0,
    ):
        # Pylance actually keeps track of the intersection type
        assert isinstance(inner, Sized), "inner dataset must be sized"
        assert len(inner) > 0, "inner dataset must be non-empty"

        self.cache: dict[int, tuple[Tensor, Tensor]] = {}
        self.dataset = inner
        self.eraser = eraser
        self.transform = transform
        self.p = p

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if idx not in self.cache:
            x, z = self.dataset[idx]
            x = self.eraser(x, z)
            self.cache[idx] = x, z
        else:
            x, z = self.cache[idx]

        return self.transform(x), z

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument(
        "--eraser",
        type=str,
        choices=("none", "leace", "oleace", "qleace"),
        default="none",
    )
    parser.add_argument(
        "--patch-size", type=int, default=4, help="patch size for mixer and resmlp"
    )
    parser.add_argument("--net", type=str, choices=("mixer", "resmlp", "resnet", "vit"))
    args = parser.parse_args()

    # Split the "train" set into train and validation
    nontest = CIFAR10("/home/nora/Data/cifar10", download=True, transform=T.ToTensor())
    train, val = random_split(nontest, [0.9, 0.1])

    X = torch.from_numpy(nontest.data).div(255)
    Y = torch.tensor(nontest.targets)

    # Test set is entirely separate
    test = CIFAR10(
        "/home/nora/Data/cifar10-test",
        download=True,
        train=False,
        transform=T.ToTensor(),
    )

    k = 10  # Number of classes
    final = nn.Identity() if args.net in ("mixer", "resnet", "vit") else nn.Flatten(0)
    train_trf = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            final,
        ]
    )

    if args.eraser != "none":
        cache_dir = Path(f"/home/nora/Data/cifar10-{args.eraser}.pt")
        if cache_dir.exists():
            eraser = torch.load(cache_dir)
            print("Loaded cached eraser")
        else:
            print("No eraser cached; fitting a fresh one")
            cls = {
                "leace": LeaceFitter,
                "oleace": OracleFitter,
                "qleace": QuadraticFitter,
            }[args.eraser]

            fitter = cls(3 * 32 * 32, k, dtype=torch.float64)
            for x, y in tqdm(train):
                y = torch.as_tensor(y).view(1)
                if args.eraser != "qleace":
                    y = F.one_hot(y, k)

                fitter.update(x.view(1, -1), y)

            eraser = fitter.eraser
            torch.save(eraser, cache_dir)

            print(f"Saved eraser to {cache_dir}")
    else:

        def eraser(x, y):
            return x

    train = LeacedDataset(train, eraser, transform=train_trf)
    val = LeacedDataset(val, eraser, transform=final)
    test = LeacedDataset(test, eraser, transform=final)

    # Create the data module
    dm = pl.LightningDataModule.from_datasets(
        train, val, test, batch_size=128, num_workers=8
    )

    model_cls = {
        "mixer": MlpMixer,
        "resmlp": Mlp,
        "resnet": ResNet,
        "vit": ViT,
    }[args.net]
    model = model_cls(k, patch_size=args.patch_size)

    checkpointer = LogSpacedCheckpoint(f"/home/nora/Data/cifar-ckpts/{args.name}")

    trainer = pl.Trainer(
        callbacks=[checkpointer],
        logger=WandbLogger(
            name=args.name, project="concept-erasure", entity="eleutherai"
        ),
        max_epochs=200,
    )
    trainer.fit(model, dm)
    trainer.test(model, dm)
