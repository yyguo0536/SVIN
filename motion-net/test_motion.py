# - *- coding: utf- 8 - *-import os
from options.test_options import TestOptions
from data import CreateDataLoader
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
from vnet import VNet
import time
from skimage.measure import compare_ssim, compare_psnr, compare_mse, compare_nrmse
from .models.warp_layer import SpatialTransformer

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


args = TestOptions().parse()

test_rows=[]
test_label=[]

data = pd.read_csv(\
        'test data csv file')
testdata_l = data['image']
testlabel_l = data['annotation']

for i in range(len(testdata_l)):
    test_rows.append(testdata_l[i])
    test_label.append(testlabel_l[i])


deformed_name = []
moved_name = []
fix_name = []
move_initial = []

feild1 = []
feild2 = []

dice_num = []

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    #data_loader = CreateDataLoader(opt)
    #dataset = data_loader.load_data()
    accuracy_data = []

    model = create_model(opt)
    visualizer = Visualizer(opt)

    warp_layer = SpatialTransformer(96,96,96)
    warp_layer = warp_layer.to(device)
    # create website
    #web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    #webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, \
                        #Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    for num in range(0,len(test_rows),5):
        image_list = test_rows[num:num+5]
        label_list = test_label[num:num+5]

        test_list = Test_Subjects( \
                            image_list, label_list,\
                            classes = args.classes, job = 'seg',
                            )
        test_loader = DataLoader(
                    test_list, batch_size = args.batchSize, \
                    shuffle = None, num_workers=8, pin_memory=False)

        for i, (fix_image, move_image, fix_mask, move_mask) in enumerate(test_loader):
            #if i >= opt.how_many:
            #break
            combined_image1 = torch.cat((fix_image,move_image),1)
            combined_image2 = torch.cat((move_image,fix_image),1)
            model.set_input(combined_image1, combined_image2)
                
            m_f_feild, f_m_field = model.test()

            warped_fix_mask = warp_layer(\
                    move_mask, m_f_feild[:,0,:,:,:], m_f_feild[:,2,:,:,:], m_f_feild[:,1,:,:,:])

            warped_move_mask = warp_layer(\
                    fix_mask, f_m_field[:,0,:,:,:], f_m_field[:,2,:,:,:], f_m_field[:,1,:,:,:])

            warped_fix_mask[warped_fix_mask>0.4]=1
            warped_fix_mask[warped_fix_mask<0.5]=0

            warped_move_mask_96[warped_move_mask_96>0.4]=1
            warped_move_mask_96[warped_move_mask_96<0.5]=0

            dice_num.append(dice_loss(warped_move_mask, move_mask))
            dice_num.append(dice_loss(warped_fix_mask, fix_mask))

            m_f_feild = sitk.GetImageFromArray(m_f_feild[0,:,:,:,:].cpu().data)
            f_m_field = sitk.GetImageFromArray(f_m_field[0,:,:,:,:].cpu().data)

            label_name = test_rows[num].split('/')
            file_name = label_name[-1].split('.')
            file1 = \
                    label_name[-2] + '_' + str(index_list[1].numpy()[0]) \
                    + 'to' + str(index_list[0].numpy()[0]) + '_deform' + '.nii'
            file2 = \
                    label_name[-2] + '_' + str(index_list[0].numpy()[0]) \
                    + 'to' + str(index_list[1].numpy()[0])+ '_deform' + '.nii'
                
            path_label = \
                    'path'
            print('=> & saving result to {}'.format(path_label))
            sitk.WriteImage(f_m_field, (path_label + file2))
            sitk.WriteImage(m_f_feild, (path_label + file1))
            feild1.append((path_label + file1))
            feild2.append((path_label + file2))
            #sitk.WriteImage(fix_file, (path_label + fix_file_name))
            #sitk.WriteImage(move_file, (path_label + move_file_name))
                


    df = pd.DataFrame({'forward':feild2,'backward':feild1})
    df.to_csv("deform_new_all.csv", index=False)
    print(np.array(dice_num).mean())

