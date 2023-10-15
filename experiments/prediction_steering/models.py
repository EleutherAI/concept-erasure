from itertools import pairwise
from typing import Literal

import pytorch_lightning as pl
import torch
import torchmetrics as tm
import torchvision as tv
from torch import nn
from torch.optim import RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR


class Mlp(pl.LightningModule):
    def __init__(self, k, h=512, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.build_net()
        self.train_acc = tm.Accuracy("multiclass", num_classes=k)
        self.val_acc = tm.Accuracy("multiclass", num_classes=k)
        self.test_acc = tm.Accuracy("multiclass", num_classes=k)

    def build_net(self):
        sizes = [3 * 32 * 32] + [self.hparams["h"]] * 4

        self.net = nn.Sequential(
            *[
                MlpBlock(
                    in_dim,
                    out_dim,
                    device=self.device,
                    dtype=self.dtype,
                    residual=True,
                    act="gelu",
                )
                for in_dim, out_dim in pairwise(sizes)
            ]
        )
        self.net.append(nn.Linear(self.hparams["h"], self.hparams["k"]))

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)

        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
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
        opt = RAdam(self.parameters(), lr=1e-4)
        return [opt], [CosineAnnealingLR(opt, T_max=200)]


class MlpMixer(Mlp):
    def build_net(self):
        from mlp_mixer_pytorch import MLPMixer

        self.net = MLPMixer(
            image_size=32,
            channels=3,
            patch_size=self.hparams.get("patch_size", 4),
            num_classes=self.hparams["k"],
            dim=512,
            depth=6,
            dropout=0.1,
        )


class ResNet(Mlp):
    def build_net(self):
        self.net = tv.models.resnet18(pretrained=False, num_classes=self.hparams["k"])


class ViT(MlpMixer):
    def build_net(self):
        from vit_pytorch import ViT

        self.net = ViT(
            image_size=32,
            patch_size=self.hparams.get("patch_size", 4),
            num_classes=self.hparams["k"],
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1,
        )


class MlpBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
        residual: bool = True,
        *,
        act: Literal["relu", "gelu"] = "relu",
        norm: Literal["batch", "layer"] = "batch",
    ):
        super().__init__()

        self.linear1 = nn.Linear(
            in_features, out_features, bias=False, device=device, dtype=dtype
        )
        self.linear2 = nn.Linear(
            out_features, out_features, bias=False, device=device, dtype=dtype
        )
        self.act_fn = nn.ReLU() if act == "relu" else nn.GELU()

        norm_cls = nn.BatchNorm1d if norm == "batch" else nn.LayerNorm
        self.bn1 = norm_cls(out_features, device=device, dtype=dtype)
        self.bn2 = norm_cls(out_features, device=device, dtype=dtype)
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
        out = self.act_fn(out)

        out = self.linear2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        if self.residual:
            out += identity

        out = self.act_fn(out)
        return out
