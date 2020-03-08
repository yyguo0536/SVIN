import torch 
import torch.nn as nn
from .warp_layer import Dense3DSpatialTransformer


class UNet3D(nn.Module):
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet3D, self).__init__()
        self.ec0 = self.encoder(self.in_channel, 32, padding=1, bias=False, batchnorm=True)
        self.ec1 = self.encoder(32, 32, bias=False, padding=1, batchnorm=True)
        self.ec2 = self.encoder(32, 64, bias=False, padding=1, batchnorm=True)
        self.ec3 = self.encoder(64, 64, bias=False, padding=1, batchnorm=True)
        self.ec4 = self.encoder(64, 128, bias=False, padding=1, batchnorm=True)
        self.ec5 = self.encoder(128, 128, bias=False, padding=1, batchnorm=True)
        self.ec6 = self.encoder(128, 256, bias=False, padding=1, batchnorm=True)
        self.ec7 = self.encoder(256, 256, bias=False, padding=1, batchnorm=True)

        #self.pool0 = nn.MaxPool3d(2)
        #self.pool1 = nn.MaxPool3d(2)
        #self.pool2 = nn.MaxPool3d(2)
        self.pool0 = self.encoder(32, 32, kernel_size=2, stride=2, batchnorm=True)
        self.pool1 = self.encoder(64, 64, kernel_size=2, stride=2, batchnorm=True)
        self.pool2 = self.encoder(128, 128, kernel_size=2, stride=2, batchnorm=True)

        self.dc9 = self.decoder(256, 256, kernel_size=2, stride=2, bias=False)
        self.dc8 = self.decoder(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc7 = self.decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc6 = self.decoder(128, 128, kernel_size=2, stride=2, bias=False)
        self.dc5 = self.decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc4 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc3 = self.decoder(64, 64, kernel_size=2, stride=2, bias=False)
        self.dc2 = self.decoder(32 + 64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc1 = self.decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc0 = self.decoder(32, n_classes, kernel_size=1, stride=1, bias=False)


    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer


    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
        return layer

    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        syn0 = torch.add(x, syn0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        #e2 = torch.add(e1,e2)
        syn1 = self.ec3(e2)
        syn1 = torch.add(e2, syn1)
        #del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        syn2 = torch.add(e4, syn2)
        #del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)
        e7 = torch.add(e6, e7)
        #del e5, e6

        d9 = torch.cat((self.dc9(e7), syn2),dim=1)
        #del e7, syn2

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        d7 = torch.add(d8,d7)
        #del d9, d8

        d6 = torch.cat((self.dc6(d7), syn1),dim=1)
        #del d7, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        d4 = torch.add(d5,d4)
        #del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0),dim=1)
        #del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        d1 = torch.add(d2,d1)
        #del d3, d2

        d0 = self.dc0(d1)
        return d0

















class UNet3D_deep(nn.Module):
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet3D_deep, self).__init__()
        self.ec0 = self.encoder(self.in_channel, 32, padding=1, bias=False, batchnorm=True)
        self.ec1 = self.encoder(32, 32, bias=False, padding=1, batchnorm=True)
        self.ec2 = self.encoder(32, 64, bias=False, padding=1, batchnorm=True)
        self.ec3 = self.encoder(64, 64, bias=False, padding=1, batchnorm=True)
        self.ec4 = self.encoder(64, 128, bias=False, padding=1, batchnorm=True)
        self.ec5 = self.encoder(128, 128, bias=False, padding=1, batchnorm=True)
        self.ec6 = self.encoder(128, 256, bias=False, padding=1, batchnorm=True)
        self.ec7 = self.encoder(256, 256, bias=False, padding=1, batchnorm=True)
        self.ec8 = self.encoder(256, 512, bias=False, padding=1, batchnorm=True)
        self.ec9 = self.encoder(512, 512, bias=False, padding=1, batchnorm=True)

        #self.pool0 = nn.MaxPool3d(2)
        #self.pool1 = nn.MaxPool3d(2)
        #self.pool2 = nn.MaxPool3d(2)
        self.pool0 = self.encoder(32, 32, kernel_size=2, stride=2, batchnorm=True)
        self.pool1 = self.encoder(64, 64, kernel_size=2, stride=2, batchnorm=True)
        self.pool2 = self.encoder(128, 128, kernel_size=2, stride=2, batchnorm=True)
        self.pool3 = self.encoder(256, 256, kernel_size=2, stride=2, batchnorm=True)
        
        self.dc12 = self.decoder(512, 512, kernel_size=2, stride=2, bias=False)
        self.dc11 = self.decoder(256 + 512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc10 = self.decoder(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc9 = self.decoder(256, 256, kernel_size=2, stride=2, bias=False)
        self.dc8 = self.decoder(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc7 = self.decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc6 = self.decoder(128, 128, kernel_size=2, stride=2, bias=False)
        self.dc5 = self.decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc4 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc3 = self.decoder(64, 64, kernel_size=2, stride=2, bias=False)
        self.dc2 = self.decoder(32 + 64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc1 = self.decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc0 = self.decoder(32, n_classes, kernel_size=1, stride=1, bias=False)


    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer


    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
        return layer

    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        syn0 = torch.add(x, syn0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        #e2 = torch.add(e1,e2)
        syn1 = self.ec3(e2)
        syn1 = torch.add(e2, syn1)
        #del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        syn2 = torch.add(e4, syn2)
        #del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        syn3 = self.ec7(e6)
        syn3 = torch.add(e6, syn3)
        #del e5, e6

        e7 = self.pool3(syn3)
        e8 = self.ec8(e7)
        e9 = self.ec9(e8)
        e9 = torch.add(e9, e8)
        #del e5, e6

        d12 =  torch.cat((self.dc12(e9), syn3),dim=1)

        d11 = self.dc11(d12)
        d10 = self.dc10(d11)
        d10 = torch.add(d11,d10)

        d9 = torch.cat((self.dc9(d10), syn2),dim=1)
        #del e7, syn2

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        d7 = torch.add(d8,d7)
        #del d9, d8

        d6 = torch.cat((self.dc6(d7), syn1),dim=1)
        #del d7, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        d4 = torch.add(d5,d4)
        #del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0),dim=1)
        #del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        d1 = torch.add(d2,d1)
        #del d3, d2

        d0 = self.dc0(d1)
        return d0




class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        use_bias = False

        self.net = [
            nn.Conv3d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)



class UNet3D_modify(nn.Module):
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet3D, self).__init__()
        self.ec0 = self.encoder(self.in_channel, 32, padding=1, bias=False, batchnorm=True)
        self.ec1 = self.encoder(32, 64, bias=False, padding=1, batchnorm=True)
        self.ec2 = self.encoder(64, 64, bias=False, padding=1, batchnorm=True)
        self.ec3 = self.encoder(64, 128, bias=False, padding=1, batchnorm=True)
        self.ec4 = self.encoder(128, 128, bias=False, padding=1, batchnorm=True)
        self.ec5 = self.encoder(128, 256, bias=False, padding=1, batchnorm=True)
        self.ec6 = self.encoder(256, 512, bias=False, padding=1, batchnorm=True)
        self.ec7 = self.encoder(512, 512, bias=False, padding=1, batchnorm=True)

        #self.pool0 = nn.MaxPool3d(2)
        #self.pool1 = nn.MaxPool3d(2)
        #self.pool2 = nn.MaxPool3d(2)
        self.pool0 = self.encoder(64, 64, kernel_size=2, stride=2, batchnorm=True)
        self.pool1 = self.encoder(128, 128, kernel_size=2, stride=2, batchnorm=True)
        self.pool2 = self.encoder(256, 256, kernel_size=2, stride=2, batchnorm=True)

        self.dc9 = self.decoder(512, 512, kernel_size=2, stride=2, bias=False)
        self.dc8 = self.decoder(256 + 512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc7 = self.decoder(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc6 = self.decoder(256, 256, kernel_size=2, stride=2, bias=False)
        self.dc5 = self.decoder(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc4 = self.decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc3 = self.decoder(128, 128, kernel_size=2, stride=2, bias=False)
        self.dc2 = self.decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc1 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc0 = self.decoder(64, n_classes, kernel_size=1, stride=1, bias=False)


    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer


    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
        return layer

    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        syn0 = torch.add(x, syn0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        e2 = torch.add(e1,e2)
        syn1 = self.ec3(e2)
        #syn1 = torch.add(e2, syn1)
        #del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        syn2 = torch.add(e4, syn2)
        #del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)
        e7 = torch.add(e6, e7)
        #del e5, e6

        d9 = torch.cat((self.dc9(e7), syn2))
        #del e7, syn2

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        d7 = torch.add(d8,d7)
        #del d9, d8

        d6 = torch.cat((self.dc6(d7), syn1))
        #del d7, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        d4 = torch.add(d5,d4)
        #del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0))
        #del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        d1 = torch.add(d2,d1)
        #del d3, d2

        d0 = self.dc0(d1)
        return d0






class UNet3D_FLOW(nn.Module):
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet3D_FLOW, self).__init__()
        self.ec0 = self.encoder(self.in_channel, 32, padding=1, bias=False, batchnorm=True)
        self.ec1 = self.encoder(64, 64, bias=False, padding=1, batchnorm=True)
        self.ec2 = self.encoder(128, 128, bias=False, padding=1, batchnorm=True)
        self.ec3 = self.encoder(256, 256, bias=False, padding=1, batchnorm=True)
        self.ec4 = self.last_layer(512, 512)
        
        self.pool1 = self.down_layer(32, 64)
        self.pool2 = self.down_layer(64, 128)
        self.pool3 = self.down_layer(128, 256)
        self.pool4 = self.down_layer(256, 512)

        self.concat_layer = self.last_layer(512+512, 512)

        self.up4 = self.up_layer(512, 256)
        self.dc4 = self.encoder(256, 256)
        self.up3 = self.up_layer(256, 128)
        self.dc3 = self.encoder(128, 128)
        self.up2 = self.up_layer(128, 64)
        self.dc2 = self.encoder(64, 64)
        self.up1 = self.up_layer(64, 32)
        self.dc1 = self.last_layer(32, n_classes)
        self.warp_layer = Dense3DSpatialTransformer(96,96,96)
        self.warp_inverse = Dense3DSpatialTransformer(96,96,96)

    def encoder(self, in_channels, out_channels, kernel_size=3, padding=1,
                bias=True, batchnorm=True):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.PReLU(),
                nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias),
                nn.PReLU(),
                nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, bias=bias),
                nn.PReLU())
        return layer

    def last_layer(self, in_channels, out_channels, kernel_size=1, batchnorm=True):
        layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size),
                nn.BatchNorm3d(out_channels),
                nn.PReLU())
        return layer

    def down_layer(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                bias=True, batchnorm=True):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.PReLU())
        return layer

    def up_layer(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride),
            nn.BatchNorm3d(out_channels),
            nn.PReLU())
        return layer



    def forward_flownet(self, x):
        e0 = self.ec0(x)
        e0 = torch.add(x,e0)
        d1 = self.pool1(e0)
        e1 = self.ec1(d1)
        e1 = torch.add(d1,e1)
        d2 = self.pool2(e1)
        e2 = self.ec2(d2)
        e2 = torch.add(d2,e2)
        d3 = self.pool3(e2)
        e3 = self.ec3(d3)
        e3 = torch.add(d3,e3)
        d4 = self.pool4(e3)
        e4 = self.ec4(d4)
        e4 = torch.add(d4,e4)

        return e4


    def forward(self, fix_image, move_image, cycle=False):
        fix_e4 = self.forward_flownet(fix_image)
        move_e4 = self.forward_flownet(move_image)

        e4 = torch.cat([fix_e4, move_e4],1)

        e4 = self.concat_layer(e4)
        up4 = self.up4(e4)
        dc4 = self.dc4(up4)
        dc4 = torch.add(up4,dc4)
        up3 = self.up3(dc4)
        dc3 = self.dc3(up3)
        dc3 = torch.add(up3,dc3)
        up2 = self.up2(dc3)
        dc2 = self.dc2(up2)
        dc2 = torch.add(up2,dc2)
        up1 = self.up1(dc2)
        dc1 = self.dc1(up1)

        out_image = \
            self.warp_layer(move_image, dc1[:,0,:,:,:], dc1[:,1,:,:,:], dc1[:,2,:,:,:])
        if cycle==True:
            out_image = \
                self.warp_inverse(out_image, -dc1[:,0,:,:,:], -dc1[:,1,:,:,:], -dc1[:,2,:,:,:])

        return dc1, out_image







class UNet3D_seg(nn.Module):
    def __init__(self, in_channel, n_classes, train=True):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.train = train
        super(UNet3D_seg, self).__init__()
        self.ec0 = self.encoder(self.in_channel, 16)
        self.pool1 = self.pooling_layer(16, 32)
        self.ec1 = self.encoder(32,32)
        self.pool2 = self.pooling_layer(32, 64)
        self.ec2 = self.encoder(64, 64)
        self.pool3 = self.pooling_layer(64, 128)
        self.ec3 = self.encoder(128, 128)
        self.pool4 = self.pooling_layer(128, 256)
        self.ec4 = self.encoder(256, 256)

        self.up4 = self.decoder(256,128)
        self.dc3 = self.encoder(128+128, 128)
        self.up3 = self.decoder(128, 64)
        self.dc2 = self.encoder(64+64, 64)
        self.up2 = self.decoder(64, 32)
        self.dc1 = self.encoder(32+32, 32)
        self.up1 = self.decoder(32, 16)
        self.dc0 = self.last_layer(16+16, 16)
        self.out = self.last_layer(16, n_classes)
        self.warp_layer = Dense3DSpatialTransformer(96,96,96)
        self.warp_inverse = Dense3DSpatialTransformer(96,96,96)


    def encoder(self, in_channels, out_channels, kernel_size=3, padding=1,
                batchnorm=True):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, 
                    padding=padding),
                nn.BatchNorm3d(out_channels),
                nn.PReLU(),
                nn.Conv3d(out_channels, out_channels, kernel_size, 
                    padding=padding),
                nn.BatchNorm3d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, 
                    padding=padding),
                nn.PReLU(),
                nn.Conv3d(out_channels, out_channels, kernel_size, 
                    padding=padding),
                nn.PReLU())
        return layer

    def last_layer(self, in_channels, out_channels, kernel_size=3, padding=1,
                batchnorm=True):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, 
                    padding=padding),
                nn.BatchNorm3d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, 
                    padding=padding),
                nn.PReLU())
        return layer


    def pooling_layer(self, in_channels, out_channels, kernel_size=2, 
        stride=2, batchnorm=True):
        layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, 
                    stride=stride),
                nn.BatchNorm3d(out_channels),
                nn.PReLU())
        return layer


    def up_layer(self, in_channels, out_channels, kernel_size=3, padding=1,
                batchnorm=True):
        layer = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size, 
                    stride=stride),
                nn.BatchNorm3d(out_channels),
                nn.PReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride),
            nn.BatchNorm3d(out_channels),
            nn.PReLU())
        return layer

    def forward_seg(self, x):
        e0 = self.ec0(x)
        e0 = torch.add(x,e0)
        d1 = self.pool1(e0)
        e1 = self.ec1(d1)
        e1 = torch.add(d1,e1)
        d2 = self.pool2(e1)
        e2 = self.ec2(d2)
        e2 = torch.add(d2,e2)
        d3 = self.pool3(e2)
        e3 = self.ec3(d3)
        e3 = torch.add(d3,e3)
        d4 = self.pool4(e3)
        e4 = self.ec4(d4)
        e4 = torch.add(d4,e4)

        u3 = self.up4(e4)
        cat3 = torch.cat([u3, e3],1)
        d3 = self.dc3(cat3)
        d3 = torch.add(d3,u3)
        u2 = self.up3(d3)
        cat2 = torch.cat([u2, e2],1)
        d2 = self.dc2(cat2)
        d2 = torch.add(d2,u2)
        u1 = self.up2(d2)
        cat1 = torch.cat([u1, e1],1)
        d1 = self.dc1(cat1)
        d1 = torch.add(d1,u1)
        u0 = self.up1(d1)
        cat0 = torch.cat([u0, e0],1)
        d0 = self.dc0(cat0)
        d0 = torch.add(d0,u0)
        output = self.out(d0)

        return output


    def forward(self, filed, fix_image, move_image, cycle):
        output1 = self.forward_seg(fix_image)
        output2 = self.forward_seg(move_image)
        tmp1 = torch.unsqueeze(output2[:,0,:,:,:],1)
        tmp2 = torch.unsqueeze(output2[:,1,:,:,:],1)
        if self.train == True:
            output2[:,0,:,:,:] = \
                self.warp_layer(tmp1, filed[:,0,:,:,:], filed[:,1,:,:,:], filed[:,2,:,:,:])
            output2[:,1,:,:,:] = \
                self.warp_layer(tmp2, filed[:,0,:,:,:], filed[:,1,:,:,:], filed[:,2,:,:,:])

            if cycle==True:
                output2 = \
                    self.warp_inverse(output2, -filed[:,0,:,:,:], -filed[:,1,:,:,:], -filed[:,2,:,:,:])


        return output1, output2






class UNet3D_FLOW_no(nn.Module):
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet3D_FLOW_no, self).__init__()
        self.ec0 = self.encoder(self.in_channel, 32, padding=1, bias=False, batchnorm=True)
        self.ec1 = self.encoder(64, 64, bias=False, padding=1, batchnorm=True)
        self.ec2 = self.encoder(128, 128, bias=False, padding=1, batchnorm=True)
        self.ec3 = self.encoder(256, 256, bias=False, padding=1, batchnorm=True)
        self.ec4 = self.last_layer(512, 512)
        
        self.pool1 = self.down_layer(32, 64)
        self.pool2 = self.down_layer(64, 128)
        self.pool3 = self.down_layer(128, 256)
        self.pool4 = self.down_layer(256, 512)

        #self.concat_layer = self.last_layer(512+512, 512)

        self.up4 = self.up_layer(512, 512)
        self.dc4 = self.encoder(512+256, 512)
        self.up3 = self.up_layer(512, 256)
        self.dc3 = self.encoder(256+128, 256)
        self.up2 = self.up_layer(256, 128)
        self.dc2 = self.encoder(128+64, 128)
        self.up1 = self.up_layer(128, 64)
        self.dc1 = self.last_layer(64+32, 32)
        self.dc0 = self.last_layer(32, n_classes)
        self.warp_layer = Dense3DSpatialTransformer(96,96,96)
        self.warp_inverse = Dense3DSpatialTransformer(96,96,96)

    def encoder(self, in_channels, out_channels, kernel_size=3, padding=1,
                bias=True, batchnorm=True):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.PReLU(),
                nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias),
                nn.PReLU(),
                nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, bias=bias),
                nn.PReLU())
        return layer

    def last_layer(self, in_channels, out_channels, kernel_size=1, batchnorm=True):
        layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size),
                nn.BatchNorm3d(out_channels),
                nn.PReLU())
        return layer

    def down_layer(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                bias=True, batchnorm=True):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.PReLU())
        return layer

    def up_layer(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride),
            nn.BatchNorm3d(out_channels),
            nn.PReLU())
        return layer





    def forward(self, combined_image, move_image, cycle=False):
        e0 = self.ec0(combined_image)
        #e0 = torch.add(x,e0)
        d1 = self.pool1(e0)
        e1 = self.ec1(d1)
        e1 = torch.add(d1,e1)
        d2 = self.pool2(e1)
        e2 = self.ec2(d2)
        e2 = torch.add(d2,e2)
        d3 = self.pool3(e2)
        e3 = self.ec3(d3)
        e3 = torch.add(d3,e3)
        d4 = self.pool4(e3)
        e4 = self.ec4(d4)
        e4 = torch.add(d4,e4)
        
        #e4 = self.concat_layer(e4)
        up4 = self.up4(e4)
        u4 = torch.cat((up4,e3),1)
        dc4 = self.dc4(u4)
        dc4 = torch.add(up4,dc4)
        up3 = self.up3(dc4)
        u3 = torch.cat((up3,e2),1)
        dc3 = self.dc3(u3)
        dc3 = torch.add(up3,dc3)
        up2 = self.up2(dc3)
        u2 = torch.cat((up2,e1),1)
        dc2 = self.dc2(u2)
        dc2 = torch.add(up2,dc2)
        up1 = self.up1(dc2)
        u1 = torch.cat((up1,e0),1)
        dc1 = self.dc1(u1)
        dc0 = self.dc0(dc1)

        out_image = \
            self.warp_layer(move_image, dc0[:,0,:,:,:], dc0[:,1,:,:,:], dc0[:,2,:,:,:])
        if cycle==True:
            out_image = \
                self.warp_inverse(out_image, -dc0[:,0,:,:,:], -dc0[:,1,:,:,:], -dc0[:,2,:,:,:])

        return dc0, out_image










