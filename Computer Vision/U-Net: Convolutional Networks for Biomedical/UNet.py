import torch
import torch.nn as nn
from torchvision import transforms

class Conv_block(nn.Module):
    """
    Create conv block with convoluion layer, batchnormalization and relu function to use in U-Net network.
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, stride:int=1, padding:int=0 bias=True): 
        """
        output size = (input size - kernel size + 2 * padding) / stride + 1
        """        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding) 
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding) 
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x) # (B, in_channels, H, W) -> (B, out_channels, H-2, W-2)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.conv2(x) # (B, in_channels, H-2, W-2) -> (B, out_channels, H-4, W-4)
        x = self.batchnorm(x)
        x = self.relu(x)

        return x

class UNet(nn.Module):
    """
    In downsampling, network makes 5 features(enc1, enc2, enc3, enc4, bottom).
    Bottom is merged with encs(enc4, enc3, enc2, enc1 in order) and enters decoding process.
    It is restored to the image size with positional features through a series of upconv layers.
    """
    def __init__(self, num_classes):
        """
        num_classes means the number of classes to classify.
        """
        super().__init__()
        self.num_classes = num_classes
        self.enc1 = Conv_block(in_channels=3, out_channels=64)
        self.enc2 = Conv_block(in_channels=64, out_channels=128)
        self.enc3 = Conv_block(in_channels=128, out_channels=256)
        self.enc4 = Conv_block(in_channels=256, out_channels=512)
        
        self.bottom = Conv_block(in_channels=512, out_channels=1024)
                
        self.upconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.dec1 = Conv_block(in_channels=1024, out_channels=512)
        
        self.upconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.dec2 = Conv_block(in_channels=512, out_channels=256)

        self.upconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.dec3 = Conv_block(in_channels=256, out_channels=128)
        
        self.upconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.dec4 = Conv_block(in_channels=128, out_channels=64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # (B, C, H, W) -> (B, C, H/2, W/2). Contracting image size to extract features.
        self.fc = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        # input = (B, 3, 572, 572)
        encode_1 = self.enc1(x) # (B, 64, 568, 568)
        
        encode_2 = self.pool(encode_1) # (B, 64, 284, 284)
        encode_2 = self.enc2(encode_2) # (B, 128, 280, 280)
        
        encode_3 = self.pool(encode_2) # (B, 128, 140, 140)
        encode_3 = self.enc3(encode_3) # (B, 256, 136, 136)
        
        encode_4 = self.pool(encode_3) # (B, 256, 68, 68)
        encode_4 = self.enc4(encode_4) # (B, 512, 64, 64)
        
        out = self.pool(encode_4) # (B, 512, 32, 32)
        out = self.bottom(out) # (B, 1024, 28, 28)
        
        out = self.upconv1(out) # (B, 512, 56, 56)
        out = torch.cat((transforms.CenterCrop((out.shape[2], out.shape[3]))(encode_4), out), dim=1) # (B, 1024, 56, 56)
        out = self.dec1(out) # (B, 512, 52, 52)
        
        out = self.upconv2(out) # (B, 256, 104, 104)
        out = torch.cat((transforms.CenterCrop((out.shape[2], out.shape[3]))(encode_3), out), dim=1) # (B, 512, 104, 104)
        out = self.dec2(out) # (B, 256, 100, 100)
        
        out = self.upconv3(out) # (B, 128, 200, 200)
        out = torch.cat((transforms.CenterCrop((out.shape[2], out.shape[3]))(encode_2), out), dim=1) # (B, 256, 200, 200)
        out = self.dec3(out) # (B, 128, 196, 196)

        out = self.upconv4(out) # (B, 64, 392, 392)
        out = torch.cat((transforms.CenterCrop((out.shape[2], out.shape[3]))(encode_1), out), dim=1) # (B, 128, 392, 392)
        out = self.dec4(out) # (B, 64, 388, 388)
        out = self.fc(out) # (B, 1, 388, 388)
        return out
