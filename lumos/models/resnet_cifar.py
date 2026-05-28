import torch
import torch.nn as nn
import torch.nn.functional as F

def _shortcut_is_pruned(shortcut):
    """Check if a shortcut Sequential contains any fully-pruned (0-channel) conv."""
    for m in shortcut.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)) and m.out_channels == 0:
            return True
    return False


class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        conv_pruned = self.conv1.out_channels == 0 or self.conv2.out_channels == 0
        shortcut_pruned = _shortcut_is_pruned(self.shortcut)

        # Both branches pruned: pass input through unchanged
        if conv_pruned and shortcut_pruned:
            return x

        # Conv branch pruned: only use shortcut
        if conv_pruned:
            return F.relu(self.shortcut(x))

        # Shortcut pruned: only use conv branch (no residual add)
        if shortcut_pruned:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.relu(out)

        # Normal path: both branches active
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
 
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        conv_pruned = (self.conv1.out_channels == 0
                       or self.conv2.out_channels == 0
                       or self.conv3.out_channels == 0)
        shortcut_pruned = _shortcut_is_pruned(self.shortcut)

        if conv_pruned and shortcut_pruned:
            return x

        if conv_pruned:
            return F.relu(self.shortcut(x))

        if shortcut_pruned:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            return F.relu(out)

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
 
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, width_mult=1):
        super(ResNet, self).__init__()
        base = 64 * width_mult
        self.in_planes = base
 
        self.conv1 = nn.Conv2d(3, base, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base)
        self.layer1 = self._make_layer(block, base,     num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base*2,   num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base*4,   num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base*8,   num_blocks[3], stride=2)
        self.linear = nn.Linear(base*8*block.expansion, num_classes)
 
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
 
    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        out = F.relu(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1,1))
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)

        if return_features:
            return out, feature
        else:
            return out
 
def resnet18(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)
 
def resnet18_2x(num_classes=10, **kwargs):
    """ResNet18 with 2× channel width (128-256-512-1024). ~4× more params → higher L0 prune ratio."""
    return ResNet(BasicBlock, [2,2,2,2], num_classes, width_mult=2)
 
def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes)
 
def resnet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)
 
def resnet101(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes)
 
def resnet152(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes)