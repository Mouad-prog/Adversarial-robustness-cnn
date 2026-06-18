import torch.nn as nn

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))

class CIFAR10Net(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch), nn.GELU(),
        )
        self.stage1 = nn.Sequential(ResBlock(base_ch, base_ch, stride=1),     ResBlock(base_ch, base_ch, stride=1))
        self.stage2 = nn.Sequential(ResBlock(base_ch, base_ch * 2, stride=2), ResBlock(base_ch * 2, base_ch * 2, stride=1))
        self.stage3 = nn.Sequential(ResBlock(base_ch * 2, base_ch * 4, stride=2), ResBlock(base_ch * 4, base_ch * 4, stride=1))
        self.stage4 = nn.Sequential(ResBlock(base_ch * 4, base_ch * 8, stride=2), ResBlock(base_ch * 8, base_ch * 8, stride=1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(base_ch * 8, 10))
    def forward(self, x):
        return self.head(self.pool(self.stage4(self.stage3(self.stage2(self.stage1(self.stem(x)))))))
