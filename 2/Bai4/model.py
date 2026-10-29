import torch
from torch import nn


class PretrainedResNet50(nn.Module):
    """Backbone microsoft/resnet-50 và classifier mới cho VinaFood21."""

    def __init__(
        self, num_classes=21, freeze_backbone=False,
        model_id="microsoft/resnet-50",
    ):
        super().__init__()
        try:
            from transformers import ResNetForImageClassification
        except ImportError as error:
            raise ImportError("Cài transformers: pip install transformers") from error
        base = ResNetForImageClassification.from_pretrained(model_id)
        self.resnet = base.resnet
        self.classifier = nn.Linear(2048, num_classes)
        self.freeze_backbone = freeze_backbone
        self.model_id = model_id
        if freeze_backbone:
            for parameter in self.resnet.parameters():
                parameter.requires_grad = False
            self.resnet.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.resnet.eval()
        return self

    def forward(self, images):
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.resnet(images).pooler_output
        else:
            features = self.resnet(images).pooler_output
        return self.classifier(features.flatten(1))


def n_params(model, trainable_only=False):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )
