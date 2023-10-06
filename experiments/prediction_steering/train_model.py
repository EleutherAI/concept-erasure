from argparse import ArgumentParser
from itertools import pairwise
from pathlib import Path
from typing import Callable, Sized

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics as tm
import torchvision as tv
import torchvision.transforms as T
from concept_erasure import LeaceFitter, OracleFitter, QuadraticFitter
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor, nn
from torch.optim import RAdam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, random_split
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm


# Use faster matmul precision
torch.set_float32_matmul_precision('high')


class Mlp(pl.LightningModule):
    def __init__(self, k, h=512):
        super().__init__()
        self.save_hyperparameters()

        self.build_net()
        self.train_acc = tm.Accuracy("multiclass", num_classes=k)
        self.val_acc = tm.Accuracy("multiclass", num_classes=k)
        self.test_acc = tm.Accuracy("multiclass", num_classes=k)
    
    def build_net(self):
        sizes = [3 * 32 * 32] + [self.hparams['h']] * 4

        self.net = nn.Sequential(
            *[
                MlpBlock(
                    in_dim, out_dim, device=self.device, dtype=self.dtype, residual=True
                )
                for in_dim, out_dim in pairwise(sizes)
            ]
        )
        # ResNet initialization
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        
        self.net.append(
            nn.Linear(self.hparams['h'], self.hparams['k'])
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
    
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)

        self.train_acc(y_hat, y)
        self.log(
            "train_acc", self.train_acc, on_epoch=True, on_step=False
        )
        # Log the norm of the weights
        fc = self.net[-1] if isinstance(self.net, nn.Sequential) else None
        if isinstance(fc, nn.Linear):
            self.log("weight_norm", fc.weight.data.norm())

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)

        self.val_acc(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)

        self.test_acc(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = SGD(self.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
        return [opt], [CosineAnnealingLR(opt, T_max=200)]


class MlpMixer(Mlp):
    def build_net(self):
        from mlp_mixer_pytorch import MLPMixer
        self.net = MLPMixer(
            image_size = 32,
            channels = 3,
            patch_size = 4,
            num_classes = self.hparams['k'],
            dim = 512,
            depth = 6,
            dropout = 0.1,
        )

    def configure_optimizers(self):
        opt = RAdam(self.parameters(), lr=1e-4)
        return [opt], [CosineAnnealingLR(opt, T_max=200)]


class ResNet(Mlp):
    def build_net(self):
        self.net = tv.models.resnet18(pretrained=False, num_classes=self.hparams['k'])


class ViT(MlpMixer):
    def build_net(self):
        from vit_pytorch import ViT
        self.net = ViT(
            image_size = 32,
            patch_size = 4,
            num_classes = self.hparams['k'],
            dim = 512,
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1
        )


class MlpBlock(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device=None, dtype=None, residual: bool = True,
    ):
        super().__init__()

        self.linear1 = nn.Linear(
            in_features, out_features, bias=False, device=device, dtype=dtype
        )
        self.linear2 = nn.Linear(
            out_features, out_features, bias=False, device=device, dtype=dtype
        )
        self.bn1 = nn.BatchNorm1d(out_features, device=device, dtype=dtype)
        self.bn2 = nn.BatchNorm1d(out_features, device=device, dtype=dtype)
        self.downsample = (
            nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)
            if in_features != out_features
            else None
        )
        self.residual = residual

    def forward(self, x):
        identity = x

        out = self.linear1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)

        out = self.linear2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        if self.residual:
            out += identity

        out = nn.functional.relu(out)
        return out


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
        "--eraser", type=str, choices=("none", "leace", "oleace", "qleace")
    )
    parser.add_argument("--net", type=str, choices=("mixer", "resmlp", "resnet", "vit"))
    args = parser.parse_args()

    # Split the "train" set into train and validation
    nontest = CIFAR10(
        "/home/nora/Data/cifar10", download=True, transform=T.ToTensor()
    )
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
    train_trf = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        final,
    ])

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
        eraser = lambda x, y: x

    train = LeacedDataset(train, eraser, transform=train_trf)
    val = LeacedDataset(val, eraser, transform=final)
    test = LeacedDataset(test, eraser, transform=final)

    # Create the data module
    dm = pl.LightningDataModule.from_datasets(train, val, test, batch_size=128, num_workers=8)

    model_cls = {
        "mixer": MlpMixer,
        "resmlp": Mlp,
        "resnet": ResNet,
        "vit": ViT,
    }[args.net]
    model = model_cls(k)

    trainer = pl.Trainer(
        callbacks=[
            # EarlyStopping(monitor="val_loss", patience=5),
        ],
        logger=WandbLogger(name=args.name, project="mdl", entity="eleutherai"),
        max_epochs=200,
    )
    trainer.fit(model, dm)
    trainer.test(model, dm)
