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


args = TrainOptions().parse()

device = torch.device('cuda:0')

training_rows24=[]
trainning_label24=[]
training_rows48=[]
trainning_label48=[]
training_rows96=[]
trainning_label96=[]


data = pd.read_csv('your training data with scale/4')
training_l = data['image']
label_l = data['annotation']
for i in range(len(training_l)):
    training_rows24.append(training_l[i])
    trainning_label24.append(label_l[i])

data = pd.read_csv('your training data with scale/2')
training_l = data['image']
label_l = data['annotation']
for i in range(len(training_l)):
    training_rows48.append(training_l[i])
    trainning_label48.append(label_l[i])


data = pd.read_csv('your training data with scale/1')
training_l = data['image']
label_l = data['annotation']
for i in range(len(training_l)):
    training_rows96.append(training_l[i])
    trainning_label96.append(label_l[i])




from scipy.misc import imsave

if __name__ == '__main__':
    opt = TrainOptions().parse() # Define the parameters
    dataset_size = len(training_rows)
    print('#training images = %d' % dataset_size)

    
    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    total_valid = 0
    train_similar_loss = []
    train_field_loss = []
    train_gradient_loss = []
    #train_dice_loss = []
    #train_distance_loss = []
    #train_angle_loss = []
    #train_simiangle_loss = []
    #train_sum_loss = []

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        epoch_iter_valid = 0

        for num in range(0,len(training_rows96),5):
            image_list96 = training_rows96[num:num+5]
            image_list48 = training_rows48[num:num+5]
            image_list24 = training_rows24[num:num+5]
            index_num = num / 5
            trainning_list = Paired_Subjects(\
                    image_list96, image_list48, image_list24, classes = args.classes, job='seg')

            train_loader = DataLoader(
                        trainning_list, batch_size=args.batchSize, \
                        shuffle=None, num_workers=8, pin_memory=False)



            for batch_idx, (image1_96, image1_48, image1_24, image2_96, image2_48, image2_24, index_l) in enumerate(train_loader):

                iter_start_time = time.time()
                combined_image12_96 = torch.cat((image1_96, image2_96),1)
                combined_image21_96 = torch.cat((image2_96, image1_96),1)

                combined_image12_48 = torch.cat((image1_48, image2_48),1)
                combined_image21_48 = torch.cat((image2_48, image1_48),1)

                combined_image12_24 = torch.cat((image1_24, image2_24),1)
                combined_image21_24 = torch.cat((image2_24, image1_24),1)

                if total_steps % opt.print_freq == 0:

                    t_data = iter_start_time - iter_data_time

                visualizer.reset()
                total_steps += opt.batchSize
                epoch_iter += opt.batchSize
                model.set_input(combined_image12_96, combined_image12_48, combined_image12_24, \
                        combined_image21_96, combined_image21_48, combined_image21_24, index_l)
                model.optimize_parameters()

                if total_steps % opt.display_freq == 0:

                    save_result = total_steps % opt.update_html_freq == 0
                    #visualizer.display_current_results(model.get_current_visuals(), epoch, epoch_iter, save_result)


                if total_steps % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    train_gradient_loss.append(losses['G24_gradient']+losses['G48_gradient']+losses['G96_gradient'])
                    train_field_loss.append(losses['G_field'])
                    #train_angle_loss.append(losses['G_angle'])
                    #train_distance_loss.append(losses['G_distance'])
                    train_similar_loss.append(losses['G24_L1']+losses['G48_L1']+losses['G96_L1'])
                    #train_simiangle_loss.append(losses['G_simiangle'])
                    #train_sum_loss.append(losses['G_sum'])
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

        plt.plot(train_gradient_loss)
        plt.title("gradient_loss_epoch={}".format(epoch))
        plt.xlabel("Number of iterations")
        plt.ylabel("Average DICE loss per batch")
        plt.savefig("{}/trainloss_epoch={}.png".format(args.checkpoints_dir, 'train_gradient_loss'))

        np.save('{}/TrainLoss_epoch={}.npy'.format(args.checkpoints_dir, 'train_gradient_loss'),
                                                            np.asarray(train_gradient_loss))

        plt.close('all')
        plt.plot(train_field_loss)
        plt.title("field_loss_epoch={}".format(epoch))
        plt.xlabel("Number of iterations")
        plt.ylabel("Average DICE loss per batch")
        plt.savefig("{}/trainloss_epoch={}.png".format(args.checkpoints_dir, 'train_field_loss'))

        np.save('{}/TrainLoss_epoch={}.npy'.format(args.checkpoints_dir, 'train_field_loss'),
                                                            np.asarray(train_field_loss))


        plt.close('all')
        plt.plot(train_similar_loss)
        plt.title("similar_loss_epoch={}".format(epoch))
        plt.xlabel("Number of iterations")
        plt.ylabel("Average DICE loss per batch")
        plt.savefig("{}/trainloss_epoch={}.png".format(args.checkpoints_dir, 'train_similar_loss'))

        np.save('{}/TrainLoss_epoch={}.npy'.format(args.checkpoints_dir, 'train_similar_loss'),
                                                            np.asarray(train_similar_loss))
        plt.close('all')

        plt.plot(train_simiangle_loss)
        plt.title("simiangle_loss_epoch={}".format(epoch))
        plt.xlabel("Number of iterations")
        plt.ylabel("Average DICE loss per batch")
        plt.savefig("{}/trainloss_epoch={}.png".format(args.checkpoints_dir, 'train_simiangle_loss'))

        np.save('{}/TrainLoss_epoch={}.npy'.format(args.checkpoints_dir, 'train_simiangle_loss'),
                                                            np.asarray(train_simiangle_loss))
        plt.close('all')


        if epoch % opt.save_epoch_freq == 0:

            print('saving the model at the end of epoch %d, iters %d' %
                    (epoch, total_steps))

            if args.which_model_netG == 'motion':
                model.save_net(epoch)
            elif args.which_model_netG == 'cnet3d' or args.which_model_netG == 'vnet3d':
                model.save_net(epoch)
            else:
                model.save_networks('latest')
                model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        model.update_learning_rate()






