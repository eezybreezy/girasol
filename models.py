import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb

class FCN(nn.Module):
    def __init__(self, num_classes:int=1000, **kwargs):
        super(FCN, self).__init__()
        self.l1 = nn.Linear(28*28, 256)
        self.l2 = nn.Linear(256, 256)
        self.act = nn.ReLU()
        # self.l2 = nn.Linear(256, num_classes)              
    
    def forward(self, x):
        x = x.reshape(1,28*28)
        x = self.l1(x)
        x = self.act(x)
        return self.act(self.l2(x))
        # return x

class LeNet(nn.Module):
    def __init__(self, num_classes=10, cifar:bool=False):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        # out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(out))
        # out = self.fc3(out)
        return out



class AlexNet(nn.Module):
    """
    AlexNet model from `https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py`
    Dropout layers removed.
    """

    def __init__(self, num_classes: int = 1000, **kwargs) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            # nn.Linear(256 * 6 * 6, num_classes),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(4096, num_classes),
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
        )
        
        with torch.no_grad():
          for _i in self.modules():
            if hasattr(_i, 'weight'):
            #   _i.weight.data += np.random.uniform(low=-0.9, high=0.9, size = _i.weight.shape).astype(np.float32)
                nn.init.kaiming_normal_(_i.weight, mode="fan_out", nonlinearity="relu")

        # self.classifier[0].bias.data += torch.empty(self.classifier[0].bias.data.shape).fill_(2.0)
        # self.classifier[2].bias.data += torch.empty(self.classifier[2].bias.data.shape).fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class BasicBlock(nn.Module):
    """
    ResNet for CIFAR code from 
    ` https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py`
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        #  out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.conv1(x))
        #  out = self.bn2(self.conv2(out))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cifar:bool=False, manipulate=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        if cifar:
          self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
          self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        if cifar:
          self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        else:
          self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        # self.lineara = nn.Linear(50176//4, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = 3*self.conv1(x)
        out = F.relu(out)    
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # out = self.linear(out)
        # out = self.lineara(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.conv1(x))
        # out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.conv2(out))
        # out = self.bn3(self.conv3(out))
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out    


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
  
def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)