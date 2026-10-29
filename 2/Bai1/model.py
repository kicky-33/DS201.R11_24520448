import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """LeNet-5 điều chỉnh cho ảnh RGB 64x64 và 21 lớp VinaFood21."""

    def __init__(self, in_ch=3, img_size=64, num_classes=21, pool="avg"):
        super().__init__()
        if pool not in {"avg", "max"}:
            raise ValueError("pool phải là 'avg' hoặc 'max'")

        pool_layer = nn.AvgPool2d if pool == "avg" else nn.MaxPool2d
        self.pool_name = pool
        self.c1 = nn.Conv2d(in_ch, 6, kernel_size=5)
        self.s2 = pool_layer(2, 2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = pool_layer(2, 2)

        n_flat = self._flat_dim(in_ch, img_size)
        self.f5 = nn.Linear(n_flat, 120)
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, num_classes)

    def _flat_dim(self, in_ch, img_size):
        with torch.no_grad():
            x = torch.zeros(1, in_ch, img_size, img_size)
            x = self.s4(self.c3(self.s2(self.c1(x))))
        return int(x.flatten(1).shape[1])

    def forward(self, x):
        x = self.s2(torch.relu(self.c1(x)))
        x = self.s4(torch.relu(self.c3(x)))
        x = x.flatten(1)
        x = torch.relu(self.f5(x))
        x = torch.relu(self.f6(x))
        return self.out(x)


def n_params(module, trainable_only=False):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


if __name__ == "__main__":
    for pool in ("avg", "max"):
        model = LeNet5(pool=pool)
        logits = model(torch.rand(2, 3, 64, 64))
        print(pool, "logits:", tuple(logits.shape), "params:", f"{n_params(model):,}")
        assert logits.shape == (2, 21)
        assert n_params(model) == 339_421
