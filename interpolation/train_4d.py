# - *- coding: utf- 8 - *-
import time
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import Visualizer
from niidata import *
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import SimpleITK as sitk
import numpy as np
import pandas as pd
from scipy.misc import imsave
from motion_model import Net_3d_ALL
from warp_layer import SpatialTransformer

args = TrainOptions().parse()

device = torch.device('cuda:0')

training_rows24=[]
trainning_label24=[]
training_rows48=[]
trainning_label48=[]
training_rows96=[]
trainning_label96=[]


data = pd.read_csv('your training data with scale/4')
training_l = data['training_file']
label_l = data['ground_truth']
for i in range(len(training_l)):
    training_rows24.append(training_l[i])
    trainning_label24.append(label_l[i])

data = pd.read_csv('your training data with scale/2')
training_l = data['training_file']
label_l = data['ground_truth']
for i in range(len(training_l)):
    training_rows48.append(training_l[i])
    trainning_label48.append(label_l[i])


data = pd.read_csv('your training data with scale/1')
training_l = data['training_file']
label_l = data['ground_truth']
for i in range(len(training_l)):
    training_rows96.append(training_l[i])
    trainning_label96.append(label_l[i])




STM_Net_Dir = 'netG-epoch500-all'

from scipy.misc import imsave



if __name__ == '__main__':
    opt = TrainOptions().parse() # Define the parameters

    dataset_size = len(training_rows)
    print('#training images = %d' % dataset_size)


    STM_Net = Net_3d_ALL()
    STM_Net.to(device)
    STM_Net = torch.nn.DataParallel(STM_Net, [0])
    tmp = torch.load(STM_Net_Dir)
    STM_Net.load_state_dict(tmp)
    STM_Net.eval()


    warp_layer96 = SpatialTransformer(96,96,96)
    warp_layer96 = warp_layer96.to(device)
    
    warp_layer48 = SpatialTransformer(48,48,48)
    warp_layer48 = warp_layer48.to(device)
    
    warp_layer24 = SpatialTransformer(24,24,24)
    warp_layer24 = warp_layer24.to(device)

    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    total_valid = 0
    trainl1_96_loss = []
    trainl1_48_loss = []
    trainl1_24_loss = []
    train_gradient = []
    train_R_img = []
    train_R_def = []
    train_loss1 = []


    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        epoch_iter_valid = 0

        for num in range(0,len(training_rows96),5):

            image_list96 = training_rows96[num:num+5]
            image_list48 = training_rows48[num:num+5]
            image_list24 = training_rows24[num:num+5]

            trainning_list = SlicesOfSubject_3d_norm(\
                image_list96, image_list48, image_list24, classes = args.classes, job='seg')

            train_loader = DataLoader(
                    trainning_list, batch_size=args.batchSize, \
                    shuffle=None, num_workers=8, pin_memory=False)



            for batch_idx, (ED96, ED48, ED24, ES96, ES48, ES24, \
                inter96, inter48, inter24, index_l) in enumerate(train_loader):

                iter_start_time = time.time()
                img0_96 = ED96.to(device)
                img1_96 = ES96.to(device)
                imgt_96 = inter96.to(device)

                img0_48 = ED48.to(device)
                img1_48 = ES48.to(device)
                imgt_48 = inter48.to(device)

                t = index_l/4
                t = t.to(device)
                t = t.type(torch.float32)

                img0_24 = ED24.to(device)
                img1_24 = ES24.to(device)
                imgt_24 = inter24.to(device)

                combined01_96 = torch.cat((img1_96, img0_96),1)
                combined10_96 = torch.cat((img0_96, img1_96),1)

                D01_24, D01_48, D01_96, D10_24, D10_48, D10_96 = STM_Net( \
                        combined01_96, combined10_96)
                
                ##############0t96###########
                D0t_01_96 = (1-t)*t*D01_96
                D0t_10_96 = - t * t * warp_layer96(\
                    tD10_96, D10_96[:,0,:,:,:], D10_96[:,2,:,:,:], D10_96[:,1,:,:,:])

                D0t_96 = D0t_01_96 + D0t_10_96
                #############1t96############
                D1t_10_96 = (1-t) * t * D10_96
                D1t_01_96 = - t * (1 - t) * warp_layer96(\
                    D01_96, D01_96[:,0,:,:,:], D01_96[:,2,:,:,:], D01_96[:,1,:,:,:])

                D1t_96 = D1t_01_96 + D1t_10_96
                
                ##############0t48###########
                D0t_01_48 = (1-t)*t*D01_48
                D0t_10_48 = - t * t * warp_layer48(\
                    tD10_48, D10_48[:,0,:,:,:], D10_48[:,2,:,:,:], D10_48[:,1,:,:,:])

                D0t_48 = D0t_01_48 + D0t_10_48
                #############1t48############
                D1t_10_48 = (1-t) * t * D10_48
                D1t_01_48 = - t * (1 - t) * warp_layer48(\
                    D01_48, D01_48[:,0,:,:,:], D01_48[:,2,:,:,:], D01_48[:,1,:,:,:])

                D1t_48 = D1t_01_48 + D1t_10_48
                
                ##############0t24###########
                D0t_01_24 = (1-t)*t*D01_24
                D0t_10_24 = - t * t * warp_layer24(\
                    tD10_24, D10_24[:,0,:,:,:], D10_24[:,2,:,:,:], D10_24[:,1,:,:,:])

                D0t_24 = D0t_01_24 + D0t_10_24
                #############1t24############
                D1t_10_24 = (1-t) * t * D10_24
                D1t_01_24 = - t * (1 - t) * warp_layer24(\
                    D01_24, D01_24[:,0,:,:,:], D01_24[:,2,:,:,:], D01_24[:,1,:,:,:])

                D1t_24 = D1t_01_24 + D1t_10_24

                #############generate linear interpolated intermediate image###############

                img_t_01_96 = warp_layer96(\
                    img0_96, D0t_96[:,0,:,:,:], D0t_96[:,2,:,:,:], D0t_96[:,1,:,:,:])

                img_t_10_96 = warp_layer96(\
                    img1_96, D1t_96[:,0,:,:,:], D1t_96[:,2,:,:,:], D1t_96[:,1,:,:,:])


                input_combine = torch.cat([img0_96, img_t_01_96, img_t_10_96, img1_96], 1)

                if total_steps % opt.print_freq == 0:

                    t_data = iter_start_time - iter_data_time

                visualizer.reset()
                total_steps += opt.batchSize
                epoch_iter += opt.batchSize
                model.set_input(input_combine, img0_96, img0_48, img0_24, \
                    img1_96, img1_48, img1_24, imgt_96, imgt_48, imgt_24, \
                    D0t_96, D1t_96, D0t_48, D1t_48, D0t_24, D1t_24, t)
                model.optimize_parameters()

                if total_steps % opt.display_freq == 0:

                    save_result = total_steps % opt.update_html_freq == 0
                        #visualizer.display_current_results(model.get_current_visuals(), epoch, epoch_iter, save_result)


                if total_steps % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    trainl1_96_loss.append(losses['G_L1_96'])
                    trainl1_48_loss.append(losses['G_L1_48'])
                    trainl1_24_loss.append(losses['G_L1_24'])
                    train_gradient.append(losses['G_gradient'])
                    train_R_img.append(losses['G_R_img'])
                    train_R_def.append(losses['G_R_def'])
                    #train_loss1.append(losses['D_seg'])
                    t = (time.time() - iter_start_time) / opt.batchSize
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

                if total_steps % opt.save_latest_freq == 0:
                    print('saving the latest model (epoch %d, total_steps %d)' %
                            (epoch, total_steps))
                    if args.which_model_netG == 'vnet' or args.which_model_netG == 'unet3d':
                        model.save_net('latest')
                    elif args.which_model_netG == 'unet3d_deep' or args.which_model_netG == 'vnet3d':
                        model.save_net(epoch)
                    else:
                        model.save_networks('latest')

                iter_data_time = time.time()

            plt.plot(trainl1_96_loss)
            plt.title("GAN_VNET_TRAIN_LOSS_epoch={}".format(epoch))
            plt.xlabel("Number of iterations")
            plt.ylabel("Average DICE loss per batch")
            plt.savefig("{}/trainloss_epoch={}.png".format(args.checkpoints_dir, epoch))

            np.save('{}/TrainLoss_epoch={}.npy'.format(args.checkpoints_dir, epoch),
                                                            np.asarray(trainl1_96_loss))

            plt.plot(trainl1_48_loss)
            plt.title("GAN_VNET_TRAIN_LOSS_epoch={}".format(epoch))
            plt.xlabel("Number of iterations")
            plt.ylabel("Average DICE loss per batch")
            plt.savefig("{}/trainloss_epoch={}.png".format(args.checkpoints_dir, epoch))

            np.save('{}/TrainLoss_epoch={}.npy'.format(args.checkpoints_dir, epoch),
                                                            np.asarray(trainl1_48_loss))


            plt.plot(trainl1_24_loss)
            plt.title("GAN_VNET_TRAIN_LOSS_epoch={}".format(epoch))
            plt.xlabel("Number of iterations")
            plt.ylabel("Average DICE loss per batch")
            plt.savefig("{}/trainloss_epoch={}.png".format(args.checkpoints_dir, epoch))

            np.save('{}/TrainLoss_epoch={}.npy'.format(args.checkpoints_dir, epoch),
                                                            np.asarray(trainl1_24_loss))

            plt.plot(train_gradient)
            plt.title("GAN_VNET_TRAIN_LOSS_epoch={}".format(epoch))
            plt.xlabel("Number of iterations")
            plt.ylabel("Average DICE loss per batch")
            plt.savefig("{}/trainloss_epoch={}.png".format(args.checkpoints_dir, epoch))

            np.save('{}/TrainLoss_epoch={}.npy'.format(args.checkpoints_dir, epoch),
                                                            np.asarray(train_gradient))

            plt.plot(train_R_img)
            plt.title("GAN_VNET_TRAIN_LOSS_epoch={}".format(epoch))
            plt.xlabel("Number of iterations")
            plt.ylabel("Average DICE loss per batch")
            plt.savefig("{}/trainloss_epoch={}.png".format(args.checkpoints_dir, epoch))

            np.save('{}/TrainLoss_epoch={}.npy'.format(args.checkpoints_dir, epoch),
                                                            np.asarray(train_R_img))

            plt.plot(train_R_def)
            plt.title("GAN_VNET_TRAIN_LOSS_epoch={}".format(epoch))
            plt.xlabel("Number of iterations")
            plt.ylabel("Average DICE loss per batch")
            plt.savefig("{}/trainloss_epoch={}.png".format(args.checkpoints_dir, epoch))

            np.save('{}/TrainLoss_epoch={}.npy'.format(args.checkpoints_dir, epoch),
                                                            np.asarray(train_R_def))

            


            if epoch % opt.save_epoch_freq == 0:

                print('saving the model at the end of epoch %d, iters %d' %
                    (epoch, total_steps))

                if args.which_model_netG == 'vnet' or args.which_model_netG == 'unet3d':
                    model.save_net(epoch)
                elif args.which_model_netG == 'unet3d_deep' or args.which_model_netG == 'vnet3d':
                    model.save_net(epoch)
                else:
                    model.save_networks('latest')
                    model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

            model.update_learning_rate()






