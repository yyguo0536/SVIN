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

args = TestOptions().parse()

test_rows=[]

data = pd.read_csv(\
        'test data')
testdata_l = data['']
testlabel_l = data['']


for i in range(len(testdata_l)):
    test_rows.append(testdata_l[i])



deformed_name = []
moved_name = []
fix_name = []
move_initial = []

device = torch.device('cuda:0')
dir_deform = '.pth'


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

        #t = 0.5
        # create website
        #web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
        #webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, \
                        #Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
        # test
    for num in range(0,len(test_rows),5):
        image_list = test_rows[num:num+5]

        test_list = Slice3D_test_norm( \
                            image_list, \
                            classes = args.classes, job = 'seg',
                            )
        test_loader = DataLoader(
                    test_list, batch_size = args.batchSize, \
                    shuffle = None, num_workers=8, pin_memory=False)

        for i, (fix_image, move_image) in enumerate(test_loader):
            #if i >= opt.how_many:
            #break
            iter_start_time = time.time()
            img0_96 = fix_image.to(device)
            img1_96 = move_image.to(device)

            t = torch.tensor(0.0, dtype=torch.float)

            combined01_96 = torch.cat((img1_96, img0_96),1)
            combined10_96 = torch.cat((img0_96, img1_96),1)

            D01_24, D01_48, D01_96, D10_24, D10_48, D10_96 = deform_pred( \
                        combined01_96, combined10_96)

            img_4d_tmp = img0_96
            img_4d_tmp = img_4d_tmp[0,0,:,:,:]
            img_4d_tmp = torch.unsqueeze(img_4d_tmp, 3)

            for k in range(21):
                #t = t + k * 0.1
                if t == 0:
                    t = t + 0.02
                t = t.to(device)
                #t = t.type(torch.float32)
                print(t)

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
                

                model.set_input(input_combine, img0_96, img1_96, D0t_96, D1t_96, D0t_48, D1t_48, D0t_24, D1t_24)
                fake_image, D0t_96_test, D1t_96_test, V_t_0 = model.test()

                fake_image = fake_image[0,0,:,:,:]
                fake_image = torch.unsqueeze(fake_image, 3)

                img_4d_tmp = torch.cat((img_4d_tmp, fake_image), 3)
                if t == 0.02:
                    t = torch.tensor(0.0, dtype=torch.float)
                t = t + 0.05

                if t == 1:
                    t = torch.tensor(0.98, dtype=torch.float)

            img1_96 = img1_96[0,0,:,:,:]
            img1_96 = torch.unsqueeze(img1_96, 3)

            img_4d_tmp = torch.cat((img_4d_tmp, img1_96), 3)
            img_4d_tmp = img_4d_tmp.cpu().data.numpy()
            img_4d = sitk.GetImageFromArray(img_4d_tmp[:,:,:,1:-1])
            sitk.WriteImage(img_4d, 'test_4d.nii')
                    
            img1_name = './test_img/t' + str(index_l[0].numpy()[0]) + '.nii'
