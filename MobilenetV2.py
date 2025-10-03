import torch.nn as nn
cfg = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ConvReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding,
                      groups=groups, bias=True),
            nn.ReLU6(inplace=True)
        )

# InvertedResidual block without BatchNorm
class InvertedResidual_woBN(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=True)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2_woBN(nn.Module):
    def __init__(self, num_classes, dropout):
        super(MobileNetV2, self).__init__()
        self.init_channels = 3
        self.loss = 0
        self.layer1 = self._make_layers(cfg[0])
        self.layer2 = self._make_layers(cfg[1])
        self.layer3 = self._make_layers(cfg[2])
        self.layer4 = self._make_layers(cfg[3])
        self.layer5 = self._make_layers(cfg[4])
        self.layer6 = self._make_layers(cfg[5])
        self.layer7 = self._make_layers(cfg[6])
        self.final_conv = ConvReLU(self.init_channels, 1280, kernel_size=1)
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1280, num_classes)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layers(self, cfg):
        layers = []
        t, c, n, s = cfg
        for i in range(n):
            stride = s if i == 0 else 1
            layers.append(InvertedResidual_woBN(self.init_channels, c, stride=stride, expand_ratio=t))
            self.init_channels = c
        return nn.Sequential(*layers)    

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.final_conv(out)                    # last 1x1 conv
        out = nn.functional.adaptive_avg_pool2d(out, 1)  # global avg pool → [B, 1280, 1, 1]
        out = out.view(out.size(0), -1)               # flatten → [B, 1280]
        out = self.classifier(out)
        return out

class MobileNetV2(nn.Module):
    def __init__(self, num_classes, dropout):
        super(MobileNetV2, self).__init__()
        self.init_channels = 3
        self.loss = 0
        self.layer1 = self._make_layers(cfg[0])
        self.layer2 = self._make_layers(cfg[1])
        self.layer3 = self._make_layers(cfg[2])
        self.layer4 = self._make_layers(cfg[3])
        self.layer5 = self._make_layers(cfg[4])
        self.layer6 = self._make_layers(cfg[5])
        self.layer7 = self._make_layers(cfg[6])
        self.final_conv = ConvBNReLU(self.init_channels, 1280, kernel_size=1)
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1280, num_classes)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layers(self, cfg):
        layers = []
        t, c, n, s = cfg
        for i in range(n):
            stride = s if i == 0 else 1
            layers.append(InvertedResidual(self.init_channels, c, stride=stride, expand_ratio=t))
            self.init_channels = c
        return nn.Sequential(*layers)    

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.final_conv(out)                    # last 1x1 conv
        out = nn.functional.adaptive_avg_pool2d(out, 1)  # global avg pool → [B, 1280, 1, 1]
        out = out.view(out.size(0), -1)               # flatten → [B, 1280]
        out = self.classifier(out)
        return out

