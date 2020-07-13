import torch
import torch.nn as nn
import torch.nn.functional as F
from warp_layer import SpatialTransformer


class NetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers=2, kernel_sz=3, pad = 1):
        '''
        ConvBlock = consistent convs
        for each conv, conv(5x5) -> BN -> activation(PReLU)
        params:
        in/out channels: output/input channels
        layers: number of convolution layers
        '''
        super(NetConvBlock, self).__init__()
        self.layers = layers
        self.afs = torch.nn.ModuleList() # activation functions
        self.convs = torch.nn.ModuleList() # convolutions
        self.bns = torch.nn.ModuleList()
        # first conv
        self.convs.append(nn.Conv3d( \
                in_channels, out_channels, kernel_size=kernel_sz, padding=pad))
        self.bns.append(nn.BatchNorm3d(out_channels))
        self.afs.append(nn.PReLU(out_channels))
        #self.afs.append(nn.ELU())
        for i in range(self.layers-1):
            self.convs.append(nn.Conv3d( \
                    out_channels, out_channels, kernel_size=kernel_sz, padding=pad))
            self.bns.append(nn.BatchNorm3d(out_channels))
            self.afs.append(nn.PReLU(out_channels))

    def forward(self, x):
        out = x
        for i in range(self.layers):
            out = self.convs[i](out)
            out = self.bns[i](out)
            out = self.afs[i](out)
        return out

class NetInBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers=1, k_size=3, pad_size=1):
        super(NetInBlock, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.convb = NetConvBlock(in_channels, out_channels, layers=layers, kernel_sz=k_size, pad=pad_size)

    def forward(self, x):
        out = self.bn(x)
        out = self.convb(x)
        out = torch.add(out, x)
        return out

class NetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers):
        super(NetDownBlock, self).__init__()
        self.down = nn.Conv3d( \
                in_channels, out_channels, kernel_size=2, stride=2)
        self.af= nn.PReLU(out_channels)
        self.bn = nn.BatchNorm3d(out_channels)
        self.convb = NetConvBlock(out_channels, out_channels, layers=layers)

    def forward(self, x):
        down = self.down(x)
        down = self.bn(down)
        down = self.af(down)
        out = self.convb(down)
        out = torch.add(out, down)
        return out

class NetUpBlock(nn.Module):
    def __init__(self, in_channels, br_channels, out_channels, layers):
        super(NetUpBlock, self).__init__()
        self.up = nn.ConvTranspose3d(\
                in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.af= nn.PReLU(out_channels)
        self.convb = NetConvBlock( \
                out_channels+br_channels, out_channels, layers=layers)

    def forward(self, x, bridge):
        up = self.up(x)
        up = self.bn(up)
        up = self.af(up)
        out = torch.cat([up, bridge], 1)
        out = self.convb(out)
        out = torch.add(out, up)
        return out


class NetOutBlock(nn.Module):
    def __init__(self, \
            in_channels, br_channels, out_channels, classes, layers=1):
        super(NetOutBlock, self).__init__()
        self.up = nn.ConvTranspose2d(\
                in_channels, out_channels, kernel_size=2, stride=2)
        self.bn_up = nn.BatchNorm2d(out_channels)
        self.af_up= nn.PReLU(out_channels)
        self.convb = NetConvBlock( \
                out_channels+br_channels, out_channels, layers=layers)
        self.conv = nn.Conv2d(out_channels, classes, kernel_size=1)
        self.bn_out = nn.BatchNorm2d(classes)
        self.af_out= nn.PReLU(classes)

    def forward(self, x, bridge):
        up = self.up(x)
        up = self.bn_up(up)
        up = self.af_up(up)
        out = torch.cat([up, bridge], 1)
        out = self.convb(out)
        out = torch.add(out, up)
        out = self.conv(out)
        out = self.bn_out(out)
        out = self.af_out(out)
        return out


class NetOutSingleBlock(nn.Module):
    def __init__(self, in_channels, classes):
        super(NetOutSingleBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, classes, kernel_size=1)
        self.bn_out = nn.BatchNorm3d(classes)
        self.af_out = nn.PReLU(classes)
        #self.af_out = nn.PReLU(classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn_out(out)
        out = self.af_out(out)
        return out



class Net_3d_ALL(nn.Module):
    def __init__(self, classes_num = 3):
        classes = classes_num
        super(Net_3d_ALL, self).__init__()
        self.in_block = NetInBlock(2, 24, 2)
        self.down_block1 = NetDownBlock(24, 48, 2)#48*48*48
        self.down_block2 = NetDownBlock(48, 64, 2)#24*24*24
        self.down_block3 = NetDownBlock(64, 128, 2)#12*12*12
        self.down_block4 = NetDownBlock(128, 256, 2)#6*6*6

        self.up_block3 = NetUpBlock(256, 128, 256, 2)#12
        self.up_block4 = NetUpBlock(256, 64, 160, 2)#24
        self.out24_1 = NetInBlock(160, 64, 2)
        self.out24_2 = NetInBlock(64, 32, 2)
        self.out24_3 = NetOutSingleBlock(32, classes)
        self.up_block5 = NetUpBlock(160, 48, 80, 2)#48
        self.out48_1 = NetInBlock(80, 32, 2)
        self.out48_2 = NetInBlock(64, 32, 2)
        self.out48_3 = NetOutSingleBlock(32, classes)
        self.up_block6 = NetUpBlock(80, 24, 32, 2)#96
        self.out96_1 = NetInBlock(64, 32, 1)
        self.out96_block = NetOutSingleBlock(32, classes)
        self.warp_layer96 = SpatialTransformer(96,96,96)
        self.warp_layer48 = SpatialTransformer(48,48,48)
        self.warp_layer24 = SpatialTransformer(24,24,24)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')#Interpolate(scale_factor=(2, 2, 2), mode='trilinear')

        

    def forward_branch(self, combined_image):
        br1 = self.in_block(combined_image)
        br2 = self.down_block1(br1)
        br3 = self.down_block2(br2)
        br4 = self.down_block3(br3)
        out = self.down_block4(br4)
        #out = self.down_block6(br6)
        #out = self.up_block1(out, br6)
        out = self.up_block3(out, br4)
        out = self.up_block4(out, br3)
        out24 = self.out24_1(out)
        out24 = self.out24_2(out24)
        deform24 = self.out24_3(out24)
        out = self.up_block5(out, br2)
        out48 = self.out48_1(out)
        out48 = torch.cat((out48, self.upsample(out24)), 1)
        out48 = self.out48_2(out48)
        deform48 = self.out48_3(out48)
        out = self.up_block6(out)
        out96 = torch.cat((out, self.upsample(out48)), 1)
        out96 = self.out96_1(out96)
        
        deform96 = self.out96_block(out96)
        
        #outputs = F.log_softmax(outputs)

        return deform24, deform48, deform96

    def forward(self, combined12, combined21):
        field12_24, field12_48, field12_96 = self.forward_branch(combined12)
        field21_24, field21_48, field21_96 = self.forward_branch(combined21)

        return field12_24, field12_48, field12_96, field21_24, field21_48, field21_96
