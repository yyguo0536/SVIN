import torch
import torch.nn as nn
import torch.nn.functional as F
from .warp_layer import SpatialTransformer


class NetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers=2):
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
                in_channels, out_channels, kernel_size=3, padding=1))
        self.bns.append(nn.BatchNorm3d(out_channels))
        self.afs.append(nn.PReLU(out_channels))
        #self.afs.append(nn.ELU())
        for i in range(self.layers-1):
            self.convs.append(nn.Conv3d( \
                    out_channels, out_channels, kernel_size=3, padding=1))
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
    def __init__(self, in_channels, out_channels, layers=1):
        super(NetInBlock, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.convb = NetConvBlock(in_channels, out_channels, layers=layers)

    def forward(self, x):
        out = self.bn(x)
        out = self.convb(x)
        #out = torch.add(out, x)
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




class NetUpBlock_DI(nn.Module):
    def __init__(self, in_channels, br_channels1, br_channels2, out_channels, layers):
        super(NetUpBlock_DI, self).__init__()
        self.up = nn.ConvTranspose3d(\
                in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.af= nn.PReLU(out_channels)
        self.convb = NetConvBlock( \
                out_channels+br_channels1+br_channels2, out_channels, layers=layers)

    def forward(self, x, bridge1, bridge2):
        up = self.up(x)
        up = self.bn(up)
        up = self.af(up)
        out = torch.cat([up, bridge1, bridge2], 1)
        out = self.convb(out)
        out = torch.add(out, up)
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





class Net_3d_Scale(nn.Module):
    def __init__(self, classes_num = 3):
        classes = classes_num
        super(Net_3d_Scale, self).__init__()
        self.img_in_block = NetInBlock(4, 24, 2)
        self.img_down_block1 = NetDownBlock(24, 48, 2)#48*48*48
        self.img_down_block2 = NetDownBlock(48, 64, 2)#24*24*24
        self.img_down_block3 = NetDownBlock(64, 128, 2)#12*12*12
        self.img_down_block4 = NetDownBlock(128, 256, 2)#6*6*6

        self.def_in_block = NetInBlock(6, 24, 2)
        self.def_down_block1 = NetDownBlock(24, 48, 2)#48*48*48
        self.def_down_block2 = NetDownBlock(48, 64, 2)#24*24*24
        self.def_down_block3 = NetDownBlock(64, 128, 2)#12*12*12
        self.def_down_block4 = NetDownBlock(128, 256, 2)#6*6*6

        self.fc_block1 = nn.AdaptiveMaxPool3d((1,1,1))
        self.fc_block2 = nn.Linear(512, 1)
        self.fc_block3 = nn.Sigmoid()

        self.down_24_1 = nn.Sequential()
        self.down_24_1.add_module('conv_1', nn.Conv3d(7,24,kernel_size=3, stride=2, padding=1))
        self.down_24_1.add_module('conv_2', nn.Conv3d(24,48,kernel_size=3, stride=2))
        self.down_24_1.add_module('conv_3', nn.AdaptiveMaxPool3d((1,1,1)))

        
        self.up_block3 = NetUpBlock_DI(512, 128, 128, 256, 2)#12
        self.up_block4 = NetUpBlock_DI(256, 64, 64, 160, 2)#24
        self.out24_1 = NetInBlock(160, 64, 2)
        self.out24_2 = NetInBlock(64, 32, 2)
        self.out24_3 = NetOutSingleBlock(32, classes)
        self.up_block5 = NetJustUpBlock(160, 80, 2)#48
        self.out48_1 = NetInBlock(80, 32, 2)
        self.out48_2 = NetInBlock(64, 32, 2)
        self.out48_3 = NetOutSingleBlock(32, classes)

        self.down_48_1 = nn.Sequential()
        self.down_48_1.add_module('conv_1', nn.Conv3d(7,24,kernel_size=3, stride=2, padding=1))
        self.down_48_1.add_module('conv_2', nn.Conv3d(24,48,kernel_size=3, stride=2, padding=1))
        self.down_48_1.add_module('conv_3', nn.Conv3d(48,96,kernel_size=3, stride=2))
        self.down_48_1.add_module('conv_4', nn.AdaptiveMaxPool3d((1,1,1)))

        self.up_block6 = NetJustUpBlock(80, 32, 2)#96
        self.out96_1 = NetInBlock(64, 32, 1)
        self.out96_block = NetOutSingleBlock(32, classes)

        self.down_96_1 = nn.Sequential()
        self.down_96_1.add_module('conv_1', nn.Conv3d(7,24,kernel_size=3, stride=2, padding=1))
        self.down_96_1.add_module('conv_2', nn.Conv3d(24,48,kernel_size=3, stride=2, padding=1))
        self.down_96_1.add_module('conv_3', nn.Conv3d(48,96,kernel_size=3, stride=2, padding=1))
        self.down_96_1.add_module('conv_4', nn.Conv3d(96,192,kernel_size=3, stride=2))
        self.down_96_1.add_module('conv_5', nn.AdaptiveMaxPool3d((1,1,1)))

        self.fc_def2 = nn.Linear(336, 1)


        self.warp_layer96 = SpatialTransformer(96,96,96)
        self.warp_layer48 = SpatialTransformer(48,48,48)
        self.warp_layer24 = SpatialTransformer(24,24,24)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')#Interpolate(scale_factor=(2, 2, 2), mode='trilinear')

        

    def forward(self, combined_image, combined_def96, combined_def48, combined_def24):
        img_br1 = self.img_in_block(combined_image)
        img_br2 = self.img_down_block1(img_br1)
        img_br3 = self.img_down_block2(img_br2)
        img_br4 = self.img_down_block3(img_br3)
        img_out = self.img_down_block4(img_br4)

        def_br1 = self.def_in_block(combined_def96)
        def_br2 = self.def_down_block1(def_br1)
        def_br3 = self.def_down_block2(def_br2)
        def_br4 = self.def_down_block3(def_br3)
        def_out = self.def_down_block4(def_br4)

        combined_feature = torch.cat([img_out, def_out], 1)

        fc1 = self.fc_block1(combined_feature)
        fc1 = fc1.view(1, -1)
        fc2 = self.fc_block2(fc1)
        fc_img = self.fc_block3(fc2)
        
        #out = self.down_block6(br6)
        #out = self.up_block1(out, br6)
        out = self.up_block3(combined_feature, img_br4, def_br4)
        out = self.up_block4(out, img_br3, def_br3)
        out24 = self.out24_1(out)
        out24 = self.out24_2(out24)
        deform24 = self.out24_3(out24)
        df_fc24 = self.down_24_1(deform24[:,:-1,:,:,:]-combined_def24)
        out = self.up_block5(out)
        out48 = self.out48_1(out)
        out48 = torch.cat((out48, self.upsample(out24)), 1)
        out48 = self.out48_2(out48)
        deform48 = self.out48_3(out48)
        df_fc48 = self.down_48_1(deform48[:,:-1,:,:,:]-combined_def48)
        out = self.up_block6(out)
        out96 = torch.cat((out, self.upsample(out48)), 1)
        out96 = self.out96_1(out96)
        
        deform96 = self.out96_block(out96)
        df_fc96 = self.down_96_1(deform96[:,:-1,:,:,:]-combined_def96)

        df_fc96 = df_fc96.view(1,-1)
        df_fc48 = df_fc48.view(1,-1)
        df_fc24 = df_fc24.view(1,-1)
        fc_def = self.fc_def2(torch.cat((df_fc96, df_fc48, df_fc24), 1))
        fc_def = self.fc_block3(fc_def)
        
        return deform24, deform48, deform96, fc_img, fc_def

