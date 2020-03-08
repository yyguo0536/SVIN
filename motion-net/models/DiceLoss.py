import os
import sys

import numpy as np

import torch
import torch.nn
import torch.nn.modules





class DICELoss(torch.nn.Module):

    def __init__(self):
        super(DICELoss, self).__init__()

    def forward(self, output, mask):

        probs = torch.squeeze(output, 1)
        mask = torch.squeeze(mask, 1)

        intersection = probs * mask
        intersection = torch.sum(intersection, 2)
        intersection = torch.sum(intersection, 1)

        # print( num )

        den1 = probs * probs
        # print(den1.size())
        den1 = torch.sum(den1, 2)
        den1 = torch.sum(den1, 1)

        # print(den1.size())

        den2 = mask * mask
        # print(den2.size())
        den2 = torch.sum(den2, 2)
        den2 = torch.sum(den2, 1)

        # print(den2.size())
        eps = 0.0000001
        dice = 2 * ((intersection + eps) / (den1 + den2 + eps))
        # dice_eso = dice[:, 1:]
        dice_eso = dice

        loss = 1 - torch.sum(dice_eso) / dice_eso.size(0)
        return loss







class DICELossMultiClass(torch.nn.Module):

    def __init__(self):
        super(DICELossMultiClass, self).__init__()

    def forward(self, output, mask):

        #_, probs = torch.max(output, dim=1)
        #mask = torch.squeeze(mask, 1)
        probs = output.float()
        mask = mask.float()

        mask

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

