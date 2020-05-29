import torch
from torch.nn.modules.module import Module
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(Module):
    def __init__(self, height, width, depth):
        super(SpatialTransformer, self).__init__()
        self.height, self.width, self.depth = height, width, depth
        self.x_t = np.zeros([self.height, self.width, self.depth], dtype=np.float32)
        self.y_t = np.zeros([self.height, self.width, self.depth], dtype=np.float32)
        self.z_t = np.zeros([self.height, self.width, self.depth], dtype=np.float32)

        x_t = np.matmul(np.ones(shape=np.stack([self.height, 1])),
                        np.transpose(np.expand_dims(np.linspace(0.0,
                                                                self.width -1.0, self.width), 1), [1, 0]))

        y_t = np.matmul(np.expand_dims(np.linspace(0.0, self.height-1.0, self.height), 1),
                        np.ones(shape=np.stack([1, self.width])))

        x_t = np.tile(np.expand_dims(x_t, 2), [1, 1, self.depth])
        y_t = np.tile(np.expand_dims(y_t, 2), [1, 1, self.depth])

        z_t = np.linspace(0.0, self.depth-1.0, self.depth)
        z_t = np.expand_dims(np.expand_dims(z_t, 0), 0)
        z_t = np.tile(z_t, [self.height, self.width, 1])

        self.x_t = torch.from_numpy(x_t.astype(np.float32)).cuda()
        self.y_t = torch.from_numpy(y_t.astype(np.float32)).cuda()
        self.z_t = torch.from_numpy(z_t.astype(np.float32)).cuda()

    def forward(self, I, dx_t, dy_t, dz_t):
        #I = torch.unsqueeze(I,1)
        
        bsize = I.shape[0]
        x_mesh = torch.unsqueeze(self.x_t,dim = 0)
        x_mesh = x_mesh.expand(bsize, self.height, self.width ,self.depth)
        y_mesh = torch.unsqueeze(self.y_t,dim = 0)
        y_mesh = y_mesh.expand(bsize, self.height, self.width ,self.depth)
        z_mesh = torch.unsqueeze(self.z_t,dim = 0)
        z_mesh = z_mesh.expand(bsize, self.height, self.width ,self.depth)

        x_new = dx_t + x_mesh
        y_new = dy_t + y_mesh
        z_new = dz_t + z_mesh

        I = F.pad(I, (1,1,1,1,1,1), 'constant', 0)

        num_batch = I.shape[0]
        channels = I.shape[1]
        height = I.shape[2]
        width = I.shape[3]
        depth = I.shape[4]

        out_height = z_new.shape[1]
        out_width = z_new.shape[2]
        out_depth = z_new.shape[3]

        x_new = x_new.unsqueeze(1)
        y_new = y_new.unsqueeze(1)
        z_new = z_new.unsqueeze(1)

        x_new = x_new.expand(bsize, channels, out_height, out_width, out_depth)
        y_new = y_new.expand(bsize, channels, out_height, out_width, out_depth)
        z_new = z_new.expand(bsize, channels, out_height, out_width, out_depth)

        x = x_new.view(channels, -1)
        y = y_new.view(channels, -1)
        z = z_new.view(channels, -1)

        x = x.float() + 1
        y = y.float() + 1
        z = z.float() + 1

        max_x = width-1.0
        max_y = height-1.0
        max_z = depth-1.0

        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        z0 = torch.floor(z).int()
        z1 = z0 + 1


        x0 = torch.clamp(x0, min=0, max=max_x)
        x1 = torch.clamp(x1, min=0, max=max_x)
        y0 = torch.clamp(y0, min=0, max=max_y)
        y1 = torch.clamp(y1, min=0, max=max_y)
        z0 = torch.clamp(z0, min=0, max=max_z)
        z1 = torch.clamp(z1, min=0, max=max_z)


        dim3 = depth
        dim2 = depth*width
        dim1 = depth*width*height
        

        rep = torch.t(torch.unsqueeze(torch.ones([out_height*out_width*out_depth]),1)).cuda()

        rep = rep.int()

        x_channel = (torch.range(0,channels-1)*dim1).cuda()
        x_channel = x_channel.view(-1,1)

        x_channel = torch.mm(x_channel,rep.float())
        base = x_channel.view(channels, -1)


        base_y0 = base.int() + y0*dim2
        base_y1 = base.int() + y1*dim2

        idx_a = base_y0 + x0*dim3 + z0
        idx_b = base_y1 + x0*dim3 + z0
        idx_c = base_y0 + x1*dim3 + z0
        idx_d = base_y1 + x1*dim3 + z0
        idx_e = base_y0 + x0*dim3 + z1
        idx_f = base_y1 + x0*dim3 + z1
        idx_g = base_y0 + x1*dim3 + z1
        idx_h = base_y1 + x1*dim3 + z1



        im_flat = I.view(-1,channels)
        im_flat = im_flat.view(-1)
        im_flat = im_flat.float()

        Ia = torch.gather(im_flat, 0, idx_a.view(-1).long())
        Ib = torch.gather(im_flat, 0, idx_b.view(-1).long())
        Ic = torch.gather(im_flat, 0, idx_c.view(-1).long())
        Id = torch.gather(im_flat, 0, idx_d.view(-1).long())
        Ie = torch.gather(im_flat, 0, idx_e.view(-1).long())
        If = torch.gather(im_flat, 0, idx_f.view(-1).long())
        Ig = torch.gather(im_flat, 0, idx_g.view(-1).long())
        Ih = torch.gather(im_flat, 0, idx_h.view(-1).long())

        Ia = Ia.view(channels, -1)
        Ib = Ib.view(channels, -1)
        Ic = Ic.view(channels, -1)
        Id = Id.view(channels, -1)
        Ie = Ie.view(channels, -1)
        If = If.view(channels, -1)
        Ig = Ig.view(channels, -1)
        Ih = Ih.view(channels, -1)


        x1_f = x1.float()
        y1_f = y1.float()
        z1_f = z1.float()

        dx = x1_f - x
        dy = y1_f - y
        dz = z1_f - z

        wa = dz * dx * dy
        wb = dz * dx * (1-dy)
        wc = dz * (1-dx) * dy
        wd = dz * (1-dx) * (1-dy)
        we = (1-dz) * dx * dy
        wf = (1-dz) * dx * (1-dy)
        wg = (1-dz) * (1-dx) * dy
        wh = (1-dz) * (1-dx) * (1-dy)

        output = wa*Ia + wb*Ib + wc*Ic + wd*Id + we*Ie + wf*If + wg*Ig + wh*Ih
        output = output.view(-1, channels, out_height, out_width, out_depth)
        return output






