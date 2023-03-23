import torch
import torch.nn as nn

class Block(nn.Module):
    """
    layer1_1/2_1/3_1 set the stride to 2 and the others set the stride to 1.
    """
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                        nn.BatchNorm2d(num_features=out_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(num_features=out_channels))
        
        self.projection = nn.Sequential()
        """
        if stride=2, input size and output size are different. So project to match input and output size.
        """
        if stride == 2:
            self.projection = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                                nn.BatchNorm2d(num_features=out_channels))
    def forward(self, x):
        out = self.conv(x)
        out += self.projection(x)
        out = nn.ReLU(inplace=True)(out) # Since taking the ReLU results in a negative number of zero, we add shortcut and then take the ReLU.
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_layers:list, num_classes:int):
        super().__init__()
        self.block = block
        self.num_layers = num_layers
        self.num_classes = num_classes
        """
        Conv1 and maxpool are common.
        ResNet20 consists of [3, 3, 3] blocks with Block
        ResNet32 consists of [5, 5, 5] blocks with Block
        ResNet44 consists of [7, 7, 7] blocks with Block
        ResNet56 consists of [9, 9, 9] blocks with Block
        ResNet110 consists of [18, 18, 18] blocks with Block.
        ResNet1202 consists of [200, 200, 200] blocks with Block.
        """
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1) # (B, in_channels, H, W) -> (B, out_channels, H, W)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.layer1 = self.blocks(block, in_channels=16, out_channels=16, num_layers=num_layers[0])
        self.layer2 = self.blocks(block, in_channels=16, out_channels=32, num_layers=num_layers[1])
        self.layer3 = self.blocks(block, in_channels=32, out_channels=64, num_layers=num_layers[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
    def blocks(self, block, in_channels, out_channels, num_layers):
        layers = []
        for i in range(num_layers):
            if i == 0 and in_channels != out_channels:
                new_layer = block(in_channels=in_channels, out_channels=out_channels, stride=2)
                layers.append(new_layer)
            else:
                new_layer = block(in_channels=out_channels, out_channels=out_channels, stride=1)
                layers.append(new_layer)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # input = (B, 3, 32, 32)
        print(f"input size: {x.shape}")
        out = nn.ReLU(inplace=True)(self.bn1(self.conv1(x))) # (B, 3, 32, 32) -> (B, 16, 32, 32)
        print(f"size after conv1: {out.shape}")
        out = self.layer1(out) # (B, 16, 32, 32) -> (B, 16, 32, 32)
        print(f"size after layer1: {out.shape}")
        out = self.layer2(out) # (B, 16, 32, 32) -> (B, 32, 16, 16)
        print(f"size after layer2: {out.shape}")
        out = self.layer3(out) # (B, 32, 16, 16) -> (B, 64, 8, 8)
        print(f"size after layer3: {out.shape}")
        out = self.avgpool(out) # (B, 64, 8, 8) -> (B, 64, 1, 1)
        out = torch.flatten(out, 1) # (B, 64, 1, 1) -> (B, 64)
        out = self.fc(out) # (B, 64) -> (B, 10)
        return out  
    
def ResNet20(num_classes=10):
    return ResNet(Block, [3, 3, 3], num_classes)

def ResNet32(num_classes=10):
    return ResNet(Block, [5, 5, 5], num_classes)

def ResNet44(num_classes=10):
    return ResNet(Block, [7, 7, 7], num_classes)

def ResNet56(num_classes=10):
    return ResNet(Block, [9, 9, 9], num_classes)

def ResNet110(num_classes=10):
    return ResNet(Block, [18, 18, 18], num_classes)

def ResNet1202(num_classes=10):
    return ResNet(Block, [200, 200, 200], num_classes)
