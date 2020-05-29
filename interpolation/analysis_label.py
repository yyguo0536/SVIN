# - *- coding: utf- 8 - *-import os
from options.test_options import TestOptions
from models import create_model
from util.visualizer import Visualizer
from util import html
from niidata import *
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import SimpleITK as sitk
import numpy as np
import pandas as pd
import random
import eval_data as ed
import time
from motion_model import Net_3d_ALL
from warp_layer import SpatialTransformer
from skimage.measure import compare_ssim, compare_psnr, compare_mse, compare_nrmse

args = TestOptions().parse()

def dice_loss(pred, target):
    """pred: tensor with first dimension as batch
       target :tensor with first dimension as batch
    """
    smooth = 1
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return ((2. * intersection+smooth)/(A_sum+B_sum+smooth))


test_rows=[]
test_msk=[]

data = pd.read_csv(\
        'test data')
testdata_l = data['']
testlabel_l = data['']


deformed_name = []
moved_name = []
fix_name = []
move_initial = []

device = torch.device('cuda:0')
dir_deform = '.pth'

mse_num = []
nrmse_num = []
ssim_num = []
psnr_num = []

if __name__ == '__main__':
    opt = TestOptions().parse()
    

    deform_pred = Net_3d_ALL()
    deform_pred.to(device)
    deform_pred = torch.nn.DataParallel(deform_pred, [0])
    tmp = torch.load(dir_deform)
    deform_pred.load_state_dict(tmp)

    deform_pred.eval()

    warp_layer24 = SpatialTransformer(24,24,24)
    warp_layer24 = warp_layer24.to(device)

    warp_layer48 = SpatialTransformer(48,48,48)
    warp_layer48 = warp_layer48.to(device)

    warp_layer96 = SpatialTransformer(96,96,96)
    warp_layer96 = warp_layer96.to(device)

    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    total_valid = 0
    train_loss = []
    valid_loss = []
    train_loss1 = []

    dice_num = []

    for num in range(0,len(test_rows),5):
        image_list = test_rows[num:num+5]
        label_list = test_msk[num:num+5]

        test_list = AnalysisOfMask_3d( \
            image_list, label_list, classes = args.classes, job = 'seg',
            )
        test_loader = DataLoader(
                test_list, batch_size = args.batchSize, \
                shuffle = None, num_workers=8, pin_memory=False)

        for i, (fix_image, move_image,inter_img, fix_msk, move_msk, inter_msk, index_l, patients) in enumerate(test_loader):
                #if i >= opt.how_many:
                #break
            iter_start_time = time.time()
            img0_96 = fix_image.to(device)
            img1_96 = move_image.to(device)
            imgt_96 = inter_img.to(device)

            msk0_96 = fix_msk.to(device)
            msk1_96 = move_msk.to(device)
            mskt_96 = inter_msk.to(device)

            t = index_l/4
            t = t.to(device)
            t = t.type(torch.float32)

            combined01_96 = torch.cat((img1_96, img0_96),1)
            combined10_96 = torch.cat((img0_96, img1_96),1)

            with torch.no_grad():
                D01_24, D01_48, D01_96, D10_24, D10_48, D10_96 = deform_pred( \
                        combined01_96, combined10_96)

            ##############0t96###########
            D0t_01_96 = (1-t)*t*D01_96
            
            D0t_10_96 = - t * t * warp_layer96(\
                    D10_96, D10_96[:,0,:,:,:], D10_96[:,2,:,:,:], D10_96[:,1,:,:,:])

            D0t_96 = D0t_01_96 + D0t_10_96
            #############1t96############
            D1t_10_96 = (1-t) * t * D10_96
            D1t_01_96 = - t * (1 - t) * warp_layer96(\
                    D01_96, D01_96[:,0,:,:,:], D01_96[:,2,:,:,:], D01_96[:,1,:,:,:])

            D1t_96 = D1t_01_96 + D1t_10_96

            #############generate i image###############

            img_t_01_96 = warp_layer(\
                    img0_96, D0t_96[:,0,:,:,:], D0t_96[:,2,:,:,:], D0t_96[:,1,:,:,:])

            img_t_10_96 = warp_layer(\
                    img1_96, D1t_96[:,0,:,:,:], D1t_96[:,2,:,:,:], D1t_96[:,1,:,:,:])


            input_combine = torch.cat([img0_96, img_t_01_96, img_t_10_96, img1_96], 1)
                

            model.set_input(input_combine, img0_96, img1_96, D0t_96, D1t_96)
            fake_image, D0t_96_test, D1t_96_test, V_t_0 = model.test()

            V_t_1 = 1 - V_t_0

            msk_t_01_96 = warp_layer(\
                    msk0_96, D0t_96_test[:,0,:,:,:], D0t_96_test[:,2,:,:,:], D0t_96_test[:,1,:,:,:])

            msk_t_10_96 = warp_layer(\
                    msk1_96, D1t_96_test[:,0,:,:,:], D1t_96_test[:,2,:,:,:], D1t_96_test[:,1,:,:,:])

            msk_t_01_96[msk_t_01_96>0.4]=1
            msk_t_01_96[msk_t_01_96<0.5]=0

            msk_t_10_96[msk_t_10_96>0.4]=1
            msk_t_10_96[msk_t_10_96<0.5]=0

            maskt_f = (0.5 * V_t_0 * msk_t_01_96 + 0.5 * V_t_1 * msk_t_10_96) / (0.5 * V_t_0 + 0.5 * V_t_1)

            dice_num.append(dice_loss(maskt_f, mskt_96))

            img1_tmp = img1_96.cpu().data.numpy()
            img0_tmp = img0_96.cpu().data.numpy()
            img_inter_tmp = imgt_96.cpu().data.numpy()
            fake_image = fake_image.cpu().data.numpy()

            mse_num.append(compare_mse(img_inter_tmp[0,0,:,:,:], fake_image[0,0,:,:,:]))
            psnr_num.append(compare_psnr(img_inter_tmp[0,0,:,:,:], fake_image[0,0,:,:,:]))
            ssim_num.append(compare_ssim(img_inter_tmp[0,0,:,:,:], fake_image[0,0,:,:,:]))
            nrmse_num.append(compare_nrmse(img_inter_tmp[0,0,:,:,:], fake_image[0,0,:,:,:]))

            img1_tmp = sitk.GetImageFromArray(img1_tmp[0,0,:,:,:])
            img0_tmp = sitk.GetImageFromArray(img0_tmp[0,0,:,:,:])
            img_inter_tmp = sitk.GetImageFromArray(img_inter_tmp[0,0,:,:,:])
            fake_image = sitk.GetImageFromArray(fake_image[0,0,:,:,:])

            img1_name = './test_img/' + patients[0] + '_es' + '.nii'
            img0_name = './test_img/' + patients[0] + '_ed' + '.nii'
            imgt_real_name = './test_img/' + patients[0] + str(index_l.numpy()) + '.nii'
            imgt_fake_name = './test_img/' + patients[0] + str(index_l.numpy()) + '_fake.nii'

                
            #sitk.WriteImage(img1_tmp, img1_name)
            #sitk.WriteImage(img0_tmp, img0_name)
            #sitk.WriteImage(img_inter_tmp, imgt_real_name)
            #sitk.WriteImage(fake_image, imgt_fake_name)

        print(mse_num)
        print(psnr_num)
        print(ssim_num)
        print(nrmse_num)

