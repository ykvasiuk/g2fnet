import torch
from torch import nn
import torch.nn.functional as F


class Conv3dCircularPad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(Conv3dCircularPad, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=0)
        self.padding = padding

    def forward(self, x):
        x = F.pad(x, [self.padding] * 6, mode='circular')
        return self.conv(x)
    
    
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = Conv3dCircularPad(in_channels, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = Conv3dCircularPad(out_channels, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x += shortcut
        return self.relu(x)    

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels, depth):
        super().__init__()
        self.depth = depth
        self.encoders = torch.nn.ModuleList([ResidualBlock3D(in_channels,base_channels)])
        in_ = base_channels
        for i in range(depth-1):
            self.encoders.append(ResidualBlock3D(in_,2*in_))
            in_ *= 2
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)    
        self.bottleneck = ResidualBlock3D(in_,2*in_)
        self.dropout = nn.Dropout(p=0.35)
        
        self.upscalers = torch.nn.ModuleList()
        self.decoders = torch.nn.ModuleList()
        for i in range(depth):
            self.upscalers.append(nn.ConvTranspose3d(2*in_, in_, kernel_size=2, stride=2))
            self.decoders.append(ResidualBlock3D(2*in_, in_))
            in_ //= 2
            
        self.final_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1) 
            
    def forward(self, x):
        enc_outs = []
        
        enc = self.encoders[0](x)
        enc_outs.append(enc)
        
        for i in range(1, self.depth):
            enc = self.encoders[i](self.pool(enc))
            
            enc_outs.append(enc)
        
        bn = self.bottleneck(self.pool(enc))
        bn = self.dropout(bn)
        
        dec = bn
        for i in range(self.depth):
            dec = self.upscalers[i](dec)
            
            dec = torch.cat([dec, enc_outs[-1-i]], axis=1)
            dec = self.decoders[i](dec)
        
        return self.final_conv(dec)
        