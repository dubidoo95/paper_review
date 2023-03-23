import torch
import torch.nn as nn

class Block(nn.Module):
    """
    This is used on 18-layers and 34-layers ResNet.
    Conv3_1/4_1/5_1 set the stride to 2 and the others set the stride to 1.
    """
    mul = 1
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

class Bottleneck(nn.Module):
    """
    This is used on ResNet over 50-layers.
    Conv3_1/4_1/5_1 set the stride to 2 and the others set the stride to 1.
    """
    mul = 4 # multiply 1x1 conv by 4.
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                        nn.BatchNorm2d(num_features=out_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(num_features=out_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels=out_channels, out_channels=4*out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(num_features=4*out_channels)
        )
        self.projection = nn.Sequential()
        """
        if stride=2, input size and output size are different. So project to match input and output size.
        """
        if stride == 2:
            self.projection = nn.Sequential(
                                nn.Conv2d(in_channels, 4*out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                                nn.BatchNorm2d(num_features=4*out_channels))
    def forward(self, x):
        out = self.conv(x)
        out += self.projection(x)
        out = nn.ReLU(inplace=True)(out)
        return out

class ResNet(nn.Module):
    """
    Conv1 and maxpool are common.
    ResNet18 consists of [2, 2, 2, 2] blocks with Block
    ResNet34 consists of [3, 4, 6, 3] blocks with Block
    ResNet50 consists of [3, 4, 6, 3] blocks with Bottleneck
    ResNet101 consists of [3, 4, 23, 3] blocks with Bottleneck
    ResNet152 consists of [3, 8, 36, 3] blocks with bottleneck.
    """
    def __init__(self, block, num_layers:list, num_classes:int):
        super().__init__()
        self.in_channels = 64
        self.block = block
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3) # (B, in_channels, H, W) -> (B, out_channels, H/2+1/2, W/2+1/2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self.blocks(block, out_channels=64, num_layers=num_layers[0])
        self.conv3_x = self.blocks(block, out_channels=128, num_layers=num_layers[1])
        self.conv4_x = self.blocks(block, out_channels=256, num_layers=num_layers[2])
        self.conv5_x = self.blocks(block, out_channels=512, num_layers=num_layers[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(block.mul*512, num_classes)
        
    def blocks(self, block, out_channels, num_layers):
        layers = []
        for i in range(num_layers):
            if i == 0 and self.in_channels != block.mul * out_channels: # Use stride value 2 in conv3_1, conv4_1 and conv5_1.
                new_layer = block(in_channels=self.in_channels, out_channels=out_channels, stride=2)
                layers.append(new_layer)
            else:
                new_layer = block(in_channels=self.in_channels, out_channels=out_channels, stride=1)
                layers.append(new_layer)
            self.in_channels = block.mul * out_channels # If use bottleneck, out_channels are four times what they are. So multiply in_channels by 4.
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # input = (B, 3, 224, 224)
        out = self.conv1(x) # (B, 3, 224, 224) -> (B, 64, 112, 112)
        out = self.maxpool(out) # (B, 64, 112, 112) -> (B, 64, 56, 56)
        out = self.conv2_x(out) # (B, 64, 56, 56) -> (B, 64, 56, 56)
        out = self.conv3_x(out) # (B, 64, 56, 56) -> (B, 128, 28, 28)
        out = self.conv4_x(out) # (B, 128, 28, 28) -> (B, 256, 14, 14)
        out = self.conv5_x(out) # (B, 256, 14, 14) -> (B, 512, 7, 7)
        out = self.avgpool(out) # (B, 512, 7, 7) -> (B, 512, 1, 1)
        out = torch.flatten(out, 1) # (B, 512, 1, 1) -> (B, 512)
        out = self.fc(out) # (B, 512) -> (B, num_classes)
        return out  
    
def ResNet18(num_classes=1000):
    return ResNet(Block, [2, 2, 2, 2], num_classes)

def ResNet34(num_classes=1000):
    return ResNet(Block, [3, 4, 6, 3], num_classes)

def ResNet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def ResNet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def ResNet152(num_classes=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)
