import torch
import torch.nn as nn

class DSC(nn.Module):
    """
    Define depthwise seperable convolution.
    Depthwise seperable convolution consists of 3x3 depthwise convolution and 1x1 pointwise convolution.
    In channels, out channels and groups are same value in depthwise convolution.
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.depthwise_convolution = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False), 
                            nn.BatchNorm2d(num_features=in_channels),
                            nn.ReLU(inplace=True)
        )
        self.pointwise_convolution = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False), 
                            nn.BatchNorm2d(num_features=out_channels),
                            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.depthwise_convolution(x)
        out = self.pointwise_convolution(out)
        return out

class MobileNet(nn.Module):
    def __init__(self, alpha=1, num_classes=1000):
        """
        alpha means width multiplier.
        num_classes means how many classes to classify.
        """
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=int(32*self.alpha), kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = DSC(in_channels=int(32*self.alpha), out_channels=int(64*self.alpha), stride=1)
        self.conv3 = DSC(in_channels=int(64*self.alpha), out_channels=int(128*self.alpha), stride=2)
        self.conv4 = DSC(in_channels=int(128*self.alpha), out_channels=int(128*self.alpha), stride=1)
        self.conv5 = DSC(in_channels=int(128*self.alpha), out_channels=int(256*self.alpha), stride=2)
        self.conv6 = DSC(in_channels=int(256*self.alpha), out_channels=int(256*self.alpha), stride=1)
        self.conv7 = DSC(in_channels=int(256*self.alpha), out_channels=int(512*self.alpha), stride=2)
        self.conv8 = nn.Sequential(
                        DSC(in_channels=int(512*self.alpha), out_channels=int(512*self.alpha), stride=1),
                        DSC(in_channels=int(512*self.alpha), out_channels=int(512*self.alpha), stride=1),
                        DSC(in_channels=int(512*self.alpha), out_channels=int(512*self.alpha), stride=1),
                        DSC(in_channels=int(512*self.alpha), out_channels=int(512*self.alpha), stride=1),
                        DSC(in_channels=int(512*self.alpha), out_channels=int(512*self.alpha), stride=1)
        )
        self.conv9 = DSC(in_channels=int(512*self.alpha), out_channels=int(1024*self.alpha), stride=2)
        self.conv10 = DSC(in_channels=int(1024*self.alpha), out_channels=int(1024*self.alpha), stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(1024*self.alpha), num_classes)
        self.init_weights()

    # weights initialization function
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # input = (B, 3, 224, 224) if alpha = 1
        out = self.conv1(x) # (B, 32, 112, 112)
        out = self.conv2(out) # (B, 64, 112, 112)
        out = self.conv3(out) # (B, 128, 56, 56)
        out = self.conv4(out) # (B, 128, 56, 56)
        out = self.conv5(out) # (B, 256, 28, 28)
        out = self.conv6(out) # (B, 256, 28, 28)
        out = self.conv7(out) # (B, 512, 14, 14)
        out = self.conv8(out) # (B, 512, 14, 14)
        out = self.conv9(out) # (B, 1024, 7, 7)
        out = self.conv10(out) # (B, 1024, 4, 4)
        out = self.avgpool(out) # (B, 1024, 1, 1)
        out = out.view(out.size(0), -1) # (B, 1024)
        out = self.fc(out) # (B, 1000)
        return out
