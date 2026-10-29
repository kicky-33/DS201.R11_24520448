import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs):
        outputs = torch.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return torch.relu(outputs + self.shortcut(inputs))


class ResNet18(nn.Module):
    """ResNet-18 [2, 2, 2, 2] huấn luyện từ đầu cho ảnh RGB."""

    def __init__(self, num_classes=21):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_stage(64, 64, stride=1)
        self.layer2 = self._make_stage(64, 128, stride=2)
        self.layer3 = self._make_stage(128, 256, stride=2)
        self.layer4 = self._make_stage(256, 512, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make_stage(in_channels, out_channels, stride):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels),
        )

    def forward(self, inputs):
        outputs = self.stem(inputs)
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            outputs = stage(outputs)
        return self.fc(self.gap(outputs).flatten(1))


def n_params(model, trainable_only=False):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


if __name__ == "__main__":
    model = ResNet18()
    with torch.inference_mode():
        output = model(torch.zeros(2, 3, 96, 96))
    print("logits:", tuple(output.shape), "params:", f"{n_params(model):,}")
    assert output.shape == (2, 21)
    assert n_params(model) == 11_187_285
