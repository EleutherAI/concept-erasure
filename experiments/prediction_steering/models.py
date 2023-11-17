from itertools import pairwise
from typing import Literal

import pytorch_lightning as pl
import torch
import torchmetrics as tm
import torchvision as tv
from swin_transformer_v2 import SwinTransformerV2
from torch import nn
from torch.optim import SGD, RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    ConvNextV2Config,
    ConvNextV2ForImageClassification,
    get_cosine_schedule_with_warmup,
)


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
        if not isinstance(y_hat, torch.Tensor):
            y_hat = y_hat["logits"]

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
        if not isinstance(y_hat, torch.Tensor):
            y_hat = y_hat["logits"]

        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.val_acc(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        if not isinstance(y_hat, torch.Tensor):
            y_hat = y_hat["logits"]

        loss = torch.nn.functional.cross_entropy(y_hat, y)

        self.test_acc(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = RAdam(self.parameters(), lr=0.0005)
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
        model = tv.models.regnet_y_400mf(num_classes=self.hparams["k"])
        model.stem[0].stride = 1
        model.stem.insert(0, nn.Upsample(scale_factor=2))
        self.net = model

    def configure_optimizers(self):
        opt = SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-5)
        return [opt], [CosineAnnealingLR(opt, T_max=50)]


class ConvNext(Mlp):
    def build_net(self):
        cfg = ConvNextV2Config(
            image_size=32,
            # Femto architecture
            depths=[2, 2, 6, 2],
            drop_path_rate=0.1,
            hidden_sizes=[48, 96, 192, 384],
            num_labels=10,
            # The default of 4 x 4 patches shrinks the image too aggressively for
            # low-resolution images like CIFAR-10
            patch_size=2,
        )
        model = ConvNextV2ForImageClassification(cfg)

        # HuggingFace initialization is terrible; use PyTorch init instead
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.reset_parameters()

        self.net = model

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=0.005, weight_decay=0.05)
        schedule = get_cosine_schedule_with_warmup(opt, 2000, self.trainer.max_steps)
        return [opt], [{"scheduler": schedule, "interval": "step"}]


class ClassificationModelWrapper(nn.Module):
    """
    Wraps a Swin Transformer V2 model to perform image classification.
    """

    def __init__(
        self,
        model: SwinTransformerV2,
        number_of_classes: int = 10,
        output_channels: int = 384,
    ) -> None:
        """
        Constructor method
        :param model: (SwinTransformerV2) Swin Transformer V2 model
        :param number_of_classes: (int) Number of classes to predict
        :param output_channels: (int) Output channels of the last feature map of the
        Swin Transformer V2 model
        """
        # Call super constructor
        super(ClassificationModelWrapper, self).__init__()
        # Save model
        self.model: SwinTransformerV2 = model
        # Init adaptive average pooling layer
        self.pooling: nn.Module = nn.AdaptiveAvgPool2d(1)
        # Init classification head
        self.classification_head: nn.Module = nn.Linear(
            in_features=output_channels, out_features=number_of_classes
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, channels,
        height, width]
        :return: (torch.Tensor) Output classification of the shape [batch size, number
        of classes]
        """
        # Compute features
        features: list[torch.Tensor] = self.model(input)
        # Compute classification
        classification: torch.Tensor = self.classification_head(
            self.pooling(features[-1]).flatten(start_dim=1)
        )
        return classification


class Swin(Mlp):
    def build_net(self):
        model = SwinTransformerV2(
            in_channels=3,
            embedding_channels=48,
            depths=(2, 2, 6, 2),
            dropout_path=0.2,
            input_resolution=(32, 32),
            number_of_heads=(3, 6, 12, 24),
            patch_size=2,
            window_size=8,
        )
        model = ClassificationModelWrapper(model, number_of_classes=self.hparams["k"])

        self.net = model

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            # lr=0.002,
            weight_decay=0.05,
        )
        schedule = get_cosine_schedule_with_warmup(opt, 2_000, 2**16)
        return [opt], [{"scheduler": schedule, "interval": "step"}]


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
