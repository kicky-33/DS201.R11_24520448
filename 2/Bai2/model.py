import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    """Bốn nhánh Inception chạy song song và nối theo chiều channel."""

    def __init__(self, in_ch, f1, f3r, f3, f5r, f5, fp):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, f1, kernel_size=1),
            nn.ReLU(),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, f3r, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(f3r, f3, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, f5r, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(f5r, f5, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, fp, kernel_size=1),
            nn.ReLU(),
        )
        self.out_channels = f1 + f3 + f5 + fp

    def forward(self, x):
        return torch.cat(
            [self.b1(x), self.b2(x), self.b3(x), self.b4(x)],
            dim=1,
        )


class GoogLeNetSmall(nn.Module):
    """GoogLeNet rút gọn 4 Inception block cho ảnh RGB 64x64."""

    def __init__(self, num_classes=21):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.blocks = nn.ModuleList([
            InceptionBlock(32, 16, 16, 24, 4, 8, 8),
            InceptionBlock(56, 32, 32, 48, 8, 16, 16),
            InceptionBlock(112, 48, 48, 64, 12, 24, 24),
            InceptionBlock(160, 64, 64, 96, 16, 32, 32),
        ])
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(224, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index in (1, 2):
                x = self.pool(x)
        return self.classifier(self.gap(x).flatten(1))


def n_params(module, trainable_only=False):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


if __name__ == "__main__":
    model = GoogLeNetSmall(num_classes=21)
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        features = model.stem(x)
        for index, block in enumerate(model.blocks, start=1):
            features = block(features)
            print(
                f"block {index}: channels={features.shape[1]}, "
                f"shape={tuple(features.shape)}, params={n_params(block):,}"
            )
            if index in (2, 3):
                features = model.pool(features)
        logits = model(x)
    print("logits:", tuple(logits.shape), "params:", f"{n_params(model):,}")
    assert logits.shape == (2, 21)
    assert n_params(model) == 179_877
