import os
import sys

import numpy as np

import torch
import torch.nn
import torch.nn.modules






class distanceLoss(torch.nn.Module):

    def __init__(self):
        super(distanceLoss, self).__init__()
        

    def forward(self, feild):

        dx = feild[:,0,:,:,:]
        dy = feild[:,1,:,:,:]
        #dz = feild[:,2,:,:,:]

        dy = dy * dy
        dx = dx * dx
        #dz = dz * dz

        #y_dy = y_dy * y_dy
        #y_dx = y_dx * y_dx
        #y_dz = y_dz * y_dz

        #z_dy = z_dy * z_dy
        #z_dx = z_dx * z_dx
        #z_dz = z_dz * z_dz

        distance = dx + dy

        

        return (torch.mean(distance/2.0))



class angleLoss(torch.nn.Module):

    def __init__(self, channels=1):
        super(angleLoss, self).__init__()
        self.conv3d=torch.nn.Conv3d(\
            in_channels=1, out_channels=1, kernel_size=(1,3,3), stride=1, groups=channels,\
            padding = (0,1,1),bias=False)
        x_cord = torch.Tensor([1,1,1])
        x_cord = x_cord.repeat(3).view(3,3)
        x_cord = x_cord.expand(1,3,3)
        x_cord[0,1,1] = -8.0
        x_cord = x_cord.expand(channels,1,3,3)
        x_cord = torch.unsqueeze(x_cord,0)


        self.conv3d.weight.data = x_cord
        self.conv3d.weight.requires_grad = False

    def forward(self, feild):

        angles01 = torch.abs(torch.atan2(feild[0,0,:,:,:], feild[0,1,:,:,:]))
        #angles12 = torch.abs(torch.atan2(feild[0,1,:,:,:], feild[0,2,:,:,:]))
        #angles02 = torch.abs(torch.atan2(feild[0,2,:,:,:], feild[0,0,:,:,:]))

        angles01 = torch.unsqueeze(angles01,0)
        #angles02 = torch.unsqueeze(angles02,0)
        #angles12 = torch.unsqueeze(angles12,0)

        angles01 = torch.unsqueeze(angles01,0)
        #angles02 = torch.unsqueeze(angles02,0)
        #angles12 = torch.unsqueeze(angles12,0)

        angle_d = torch.abs(self.conv3d(angles01))

        return (torch.mean(angle_d/8.0))



class gradientLoss(torch.nn.Module):

    def __init__(self):
        super(gradientLoss, self).__init__()

    def forward(self, feild):

        x_dx = torch.abs(feild[:,0,1:,:,:] - feild[:,0,:-1,:,:])
        x_dy = torch.abs(feild[:,0,:,1:,:] - feild[:,0,:,:-1,:])
        x_dz = torch.abs(feild[:,0,:,:,1:] - feild[:,0,:,:,:-1])

        y_dx = torch.abs(feild[:,1,1:,:,:] - feild[:,1,:-1,:,:])
        y_dy = torch.abs(feild[:,1,:,1:,:] - feild[:,1,:,:-1,:])
        y_dz = torch.abs(feild[:,1,:,:,1:] - feild[:,1,:,:,:-1])

        #z_dx = torch.abs(feild[:,2,1:,:,:] - feild[:,2,:-1,:,:])
        #z_dy = torch.abs(feild[:,2,:,1:,:] - feild[:,2,:,:-1,:])
        #z_dz = torch.abs(feild[:,2,:,:,1:] - feild[:,2,:,:,:-1])

        x_dy = x_dy * x_dy
        x_dx = x_dx * x_dx
        x_dz = x_dz * x_dz

        y_dy = y_dy * y_dy
        y_dx = y_dx * y_dx
        y_dz = y_dz * y_dz

        #z_dy = z_dy * z_dy
        #z_dx = z_dx * z_dx
        #z_dz = z_dz * z_dz

        distance = torch.mean(x_dx) + torch.mean(x_dy) + torch.mean(x_dz) +\
                    torch.mean(y_dz) + torch.mean(y_dx) + torch.mean(y_dy) 

        return distance/9.0





class simiangleLoss(torch.nn.Module):

    def __init__(self):
        super(simiangleLoss, self).__init__()
        self.eps = torch.Tensor([0.001]).cuda()

    def forward(self, feild):

        distance = torch.sqrt(feild[:,0,:,:,:]*feild[:,0,:,:,:] +\
                        feild[:,1,:,:,:]*feild[:,1,:,:,:])

        fraction_x1 = feild[:,0,1:-1,:,:] * feild[:,0,:-2,:,:] + \
                        feild[:,1,1:-1,:,:] * feild[:,1,:-2,:,:]
        fraction_x2 = feild[:,0,1:-1,:,:] * feild[:,0,2:,:,:] + \
                        feild[:,1,1:-1,:,:] * feild[:,1,2:,:,:]
        fraction_y1 = feild[:,0,:,1:-1,:] * feild[:,0,:,:-2,:] + \
                        feild[:,1,:,1:-1,:] * feild[:,1,:,:-2,:]
        fraction_y2 = feild[:,0,:,1:-1,:] * feild[:,0,:,2:,:] + \
                        feild[:,1,:,1:-1,:] * feild[:,1,:,2:,:]
        fraction_z1 = feild[:,0,:,:,1:-1] * feild[:,0,:,:,:-2] + \
                        feild[:,1,:,:,1:-1] * feild[:,1,:,:,:-2]
        fraction_z2 = feild[:,0,:,:,1:-1] * feild[:,0,:,:,2:] + \
                        feild[:,1,:,:,1:-1] * feild[:,1,:,:,2:]

        theta1 = fraction_x1/((distance[:,1:-1,:,:]*distance[:,:-2,:,:])+self.eps)
        theta2 = fraction_x2/((distance[:,1:-1,:,:]*distance[:,2:,:,:])+self.eps)
        theta3 = fraction_y1/((distance[:,:,1:-1,:]*distance[:,:,:-2,:])+self.eps)
        theta4 = fraction_y2/((distance[:,:,1:-1,:]*distance[:,:,2:,:])+self.eps)
        theta5 = fraction_z1/((distance[:,:,:,1:-1]*distance[:,:,:,:-2])+self.eps)
        theta6 = fraction_z2/((distance[:,:,:,1:-1]*distance[:,:,:,2:])+self.eps)

        theta1 = torch.acos(theta1)
        theta2 = torch.acos(theta2)
        theta3 = torch.acos(theta3)
        theta4 = torch.acos(theta4)
        theta5 = torch.acos(theta5)
        theta6 = torch.acos(theta6)

        theta = (torch.mean(theta1-theta2)+torch.mean(theta3-theta4)+torch.mean(theta5-theta6))/3

        return torch.abs(theta)








class fieldLoss(torch.nn.Module):

    def __init__(self, channels=1):
        super(fieldLoss, self).__init__()
        self.conv3d=torch.nn.Conv3d(\
            in_channels=1, out_channels=1, kernel_size=(1,3,3), stride=1, groups=channels,\
            padding = (0,1,1),bias=False)
        #x_cord = torch.Tensor([1,1,1])
        #x_cord = x_cord.repeat(3).view(3,3)
        x_cord = torch.Tensor([[0.5,1,0.5],[1,-6,1],[0.5,1,0.5]])
        x_cord = x_cord.expand(1,3,3)
        
        #x_cord[0,1,1] = -6.0
        x_cord = x_cord.expand(channels,1,3,3)
        x_cord = torch.unsqueeze(x_cord,0)


        self.conv3d.weight.data = x_cord
        self.conv3d.weight.requires_grad = False

    def forward(self, feild):

        #x_dx = torch.abs(feild[:,0,1:,:,:] - feild[:,0,:-1,:,:])
        #x_dy = torch.abs(feild[:,0,:,1:,:] - feild[:,0,:,:-1,:])
        #x_dz = torch.abs(feild[:,0,:,:,1:] - feild[:,0,:,:,:-1])

        #y_dx = torch.abs(feild[:,1,1:,:,:] - feild[:,1,:-1,:,:])
        #y_dy = torch.abs(feild[:,1,:,1:,:] - feild[:,1,:,:-1,:])
        #y_dz = torch.abs(feild[:,1,:,:,1:] - feild[:,1,:,:,:-1])

        #z_dx = torch.abs(feild[:,2,1:,:,:] - feild[:,2,:-1,:,:])
        #z_dy = torch.abs(feild[:,2,:,1:,:] - feild[:,2,:,:-1,:])
        #z_dz = torch.abs(feild[:,2,:,:,1:] - feild[:,2,:,:,:-1])

        dx = feild[:,0,:,:,:]
        dy = feild[:,1,:,:,:]
        #dz = feild[:,2,:,:,:]

        dx = torch.unsqueeze(dx,0)
        dy = torch.unsqueeze(dy,0)
        #dz = torch.unsqueeze(dz,0)

        #x_dx = torch.unsqueeze(x_dx,0)
        #x_dy = torch.unsqueeze(x_dy,0)
        #x_dz = torch.unsqueeze(x_dz,0)
        #y_dx = torch.unsqueeze(y_dx,0)
        #y_dy = torch.unsqueeze(y_dy,0)
        #y_dz = torch.unsqueeze(y_dz,0)
        #z_dx = torch.unsqueeze(z_dx,0)
        #z_dy = torch.unsqueeze(z_dy,0)
        #z_dz = torch.unsqueeze(z_dz,0)

        #x_dy = x_dy * x_dy
        #x_dx = x_dx * x_dx
        #x_dz = x_dz * x_dz

        #y_dy = y_dy * y_dy
        #y_dx = y_dx * y_dx
        #y_dz = y_dz * y_dz

        #z_dy = z_dy * z_dy
        #z_dx = z_dx * z_dx
        #z_dz = z_dz * z_dz

        #distance = torch.mean(x_dx) + torch.mean(x_dy) + torch.mean(x_dz) +\
                    #torch.mean(y_dz) + torch.mean(y_dx) + torch.mean(y_dy) +\
                    #torch.mean(z_dx) + torch.mean(z_dy) + torch.mean(z_dz)

        gradient_d = torch.mean(self.conv3d(dy)) + torch.mean(self.conv3d(dx))

        return (torch.abs(gradient_d/9.0))





class similarLoss(torch.nn.Module):

    def __init__(self, channels=1):
        super(similarLoss, self).__init__()
        self.conv3d=torch.nn.Conv3d(\
            in_channels=1, out_channels=1, kernel_size=(1,9,9), stride=1, groups=channels,\
            padding = (0,4,4),bias=False)
        x_cord = torch.Tensor([1,1,1,1,1,1,1,1,1])
        x_cord = x_cord.repeat(9).view(9,9)
        x_cord = x_cord.expand(1,9,9)
        x_cord = x_cord.expand(channels,1,9,9)
        x_cord = torch.unsqueeze(x_cord,0)


        self.conv3d.weight.data = x_cord
        self.conv3d.weight.requires_grad = False

    def forward(self, real, fake):
        I2 = real*real
        J2 = fake*fake
        IJ = real*fake
        dif = real - fake
        dif2 = dif * dif

        dif2_sum = self.conv3d(dif2)

        I_sum = self.conv3d(real)
        J_sum = self.conv3d(fake)
        I2_sum = self.conv3d(I2)
        J2_sum = self.conv3d(J2)
        IJ_sum = self.conv3d(IJ)

        #I_sum = torch.zeros(I2.shape)
        #J_sum = torch.zeros(I2.shape)
        #I2_sum = torch.zeros(I2.shape)
        #J2_sum = torch.zeros(I2.shape)
        #IJ_sum = torch.zeros(I2.shape)

        #for i in range(4,(real.shape[2]-4)):
            #for j in range(4,(real.shape[3]-4)):
                #for k in range(4,(real.shape[4]-4)):
                    #I_sum[:,:,i,j,k] = torch.sum(real[:,:,i-4:i+5,j-4:j+5,k-4:k+5])
                    #J_sum[:,:,i,j,k] = torch.sum(fake[:,:,i-4:i+5,j-4:j+5,k-4:k+5])
                    #I2_sum[:,:,i,j,k] = torch.sum(I2[:,:,i-4:i+5,j-4:j+5,k-4:k+5])
                    #J2_sum[:,:,i,j,k] = torch.sum(J2[:,:,i-4:i+5,j-4:j+5,k-4:k+5])
                    #IJ_sum[:,:,i,j,k] = torch.sum(IJ[:,:,i-4:i+5,j-4:j+5,k-4:k+5])

        win_size = 1*9*9

        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var*J_var + 1e-5)

        dif2_sum = dif2_sum/win_size

        return 1.0*torch.mean(dif2_sum)







class similarLoss1(torch.nn.Module):

    def __init__(self, channels=1):
        super(similarLoss1, self).__init__()
        self.conv3d=torch.nn.Conv3d(\
            in_channels=1, out_channels=1, kernel_size=5, stride=1, groups=channels,\
            padding = 2,bias=False)
        x_cord = torch.Tensor([1,1,1,1,1])
        x_cord = x_cord.repeat(5).view(5,5)
        x_cord = x_cord.expand(5,5,5)
        x_cord = x_cord.expand(channels,5,5,5)
        x_cord = torch.unsqueeze(x_cord,0)


        self.conv3d.weight.data = x_cord
        self.conv3d.weight.requires_grad = False

    def forward(self, real, fake):
        I2 = real*real
        J2 = fake*fake
        IJ = real*fake

        I_sum = self.conv3d(real)
        J_sum = self.conv3d(fake)
        I2_sum = self.conv3d(I2)
        J2_sum = self.conv3d(J2)
        IJ_sum = self.conv3d(IJ)

        #I_sum = torch.zeros(I2.shape)
        #J_sum = torch.zeros(I2.shape)
        #I2_sum = torch.zeros(I2.shape)
        #J2_sum = torch.zeros(I2.shape)
        #IJ_sum = torch.zeros(I2.shape)

        #for i in range(4,(real.shape[2]-4)):
            #for j in range(4,(real.shape[3]-4)):
                #for k in range(4,(real.shape[4]-4)):
                    #I_sum[:,:,i,j,k] = torch.sum(real[:,:,i-4:i+5,j-4:j+5,k-4:k+5])
                    #J_sum[:,:,i,j,k] = torch.sum(fake[:,:,i-4:i+5,j-4:j+5,k-4:k+5])
                    #I2_sum[:,:,i,j,k] = torch.sum(I2[:,:,i-4:i+5,j-4:j+5,k-4:k+5])
                    #J2_sum[:,:,i,j,k] = torch.sum(J2[:,:,i-4:i+5,j-4:j+5,k-4:k+5])
                    #IJ_sum[:,:,i,j,k] = torch.sum(IJ[:,:,i-4:i+5,j-4:j+5,k-4:k+5])

        win_size = 5*5*5

        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var*J_var + 1e-5)

        return 1.0*torch.mean(cc)








class DICELossMultiClass(torch.nn.Module):

    def __init__(self):
        super(DICELossMultiClass, self).__init__()

    def forward(self, output, mask):

        #_, probs = torch.max(output, dim=1)
        #mask = torch.squeeze(mask, 1)
        probs = output.float()
        mask = mask.float()

        num1 = torch.sum(probs * mask)
        #num = torch.sum(num, 3)
        #num = torch.sum(num, 2)
        #num = torch.sum(num, 1)

        # print( num )

        den1 = torch.sum(probs * mask)
        # print(den1.size())
        #den1 = torch.sum(den1, 3)
        #den1 = torch.sum(den1, 2)
        #den1 = torch.sum(den1, 1)

        # print(den1.size())

        den2 = torch.sum(mask * mask)
        # print(den2.size())
        #den2 = torch.sum(den2, 3)
        #den2 = torch.sum(den2, 2)
        #den2 = torch.sum(den2, 1)

        # print(den2.size())
        eps = torch.Tensor([0.0000001]).cuda()
        dice = 2 * (num1 + eps) / (den1 + den2 + eps)
        # dice_eso = dice[:, 1:]
        #dice_eso = dice

        loss = 1.0 - torch.sum(dice) / (dice.size(0))
        return loss



class DiceLoss(torch.nn.Module):
    """Generalised Dice Loss define in
    Sudre, C. et. al. (2017)
    Generalised Dice overlap as a deep learning loss function for highly
    unbalanced segmentations. DLMIA 2017
    """
    def __init__(self, eps = 1e-8):
        super(DiceLoss, self).__init__()
        self.eps = eps 
        self.dice_loss = torch.Tensor()

    def forward(self, input2, target1):
        #raise NotImplementedError("To be implemented")
        
        tshape = target1.size()
        ishape = input2.size()
        _, input1 = torch.max(input2, dim=1)

        input1 = input1.view(ishape[0],-1)
        target = target1.view(tshape[0],-1)
        #_,input = torch.max(input, dim=1)
        dice = torch.zeros((tshape[0], ishape[1]-1),requires_grad=True)
        dice = dice.float()
        for index in torch.arange(1, ishape[1]):
            index = int(index)
            targeti = target == index
            inputi = input1 == index
            for batch_id in torch.arange(0, ishape[0]):
                batch_id = int(batch_id)
                intersection = (targeti * inputi).sum().float()
                A_sum = torch.sum(targeti * inputi).float()
                B_sum = torch.sum(targeti * targeti).float()
                dice[batch_id,index-1] =\
                        (2. * intersection + self.eps)/\
                        (A_sum+B_sum + self.eps)
                self.dice_loss = torch.Tensor([1-torch.mean(dice)])
        return self.dice_loss






class DiceLoss11(torch.nn.Module):
    """Generalised Dice Loss define in
    Sudre, C. et. al. (2017)
    Generalised Dice overlap as a deep learning loss function for highly
    unbalanced segmentations. DLMIA 2017
    """
    def __init__(self, eps = 1e-8):
        super(DiceLoss11, self).__init__()
        self.eps = eps 

    def forward(self, input, target):
        #raise NotImplementedError("To be implemented")
        
        tshape = target.size()
        ishape = input.size()
        _, input = torch.max(input, dim=1)

        input = input.view(ishape[0],-1)
        target = target.view(tshape[0],-1)
        #_,input = torch.max(input, dim=1)
        dice = torch.zeros(tshape[0], ishape[1]-1)
        for index in torch.arange(1, ishape[1]):
            index = int(index)
            targeti = target == index
            inputi = input == index
            for batch_id in torch.arange(0, ishape[0]):
                batch_id = int(batch_id)
                intersection = \
                        torch.nonzero(targeti[batch_id] * inputi[batch_id])
                intersection_sz = intersection.size()
                union = torch.nonzero(targeti[batch_id] + inputi[batch_id])
                union_sz = union.size()
                dice[batch_id,index-1] =\
                        (intersection_sz[0] + self.eps)/\
                        (union_sz[0] + self.eps)
        return (1-dice.mean())

class Dice(torch.nn.Module):
    """1 - dice as loss
    """
    def __init__(self, eps = 1e-8, ifaverage = True):
        super(Dice, self).__init__()
        self.eps = eps
        self.ifaverage = ifaverage

    def forward(self, input, target):
        assert input.data.is_cuda == target.data.is_cuda
        ifgpu = input.data.is_cuda
        input = input.data
        target = target.data
        ishape = input.size()
        print(ishape)
        print(ishape[0:2],-1)
        tshape = target.size()
        input = input.view(ishape[0],ishape[1],-1)
        print(input.size())
        target = target.view(tshape[0],-1)
        print(target.size())
        print(target)
        _,input = torch.max(input, dim=-2)
        print(input.size())
        if ifgpu:
            dice = torch.cuda.FloatTensor(tshape[0], ishape[1]-1)
        else:
            dice = torch.FloatTensor(tshape[0], ishape[1]-1)
        eps = self.eps
        for index in torch.arange(1, ishape[1]):
            index = int(index)
            targeti = target == index
            inputi = input == index
            for batch_id in torch.arange(0, ishape[0]):
                batch_id = int(batch_id)
                # output of torch.sum(tensor) is float
                intersection = \
                        torch.sum(targeti[batch_id] * inputi[batch_id])
                union = \
                        torch.sum(targeti[batch_id]) +\
                        torch.sum(inputi[batch_id])
                dice[batch_id,index-1] =\
                        (2. * intersection + self.eps)/\
                        (union + self.eps)
        if self.ifaverage:
            diceout = torch.mean(dice, dim=0)
            diceout = torch.mean(diceout, dim=0, keepdim=True)
        else:
            diceout = dice

        return torch.autograd.Variable(diceout)

    def backward(self, input):
        raise NotImplementedError(
                "Dice overlap is not differentiable as it is not consistent")






class sumLoss(torch.nn.Module):
    def __init__(self):
        super(sumLoss, self).__init__()
        

    def forward(self, feild1, feild2):
        feild = torch.abs(feild1 + feild2)
        return torch.mean(feild)
        


















