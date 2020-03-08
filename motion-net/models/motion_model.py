import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from scipy.misc import imsave
from .vnet import VNet
from .net_3d import Net_3d_ALL, Net_3d
from .unet_3d import UNet3D, UNet3D_deep, PixelDiscriminator, UNet3D_FLOW, UNet3D_seg, UNet3D_FLOW_no
from .DiceLoss import DICELossMultiClass, DiceLoss
from torch.autograd import Variable
import SimpleITK as sitk
from .MotionLoss import simiangleLoss, gradientLoss, similarLoss, DICELossMultiClass, angleLoss, distanceLoss, fieldLoss, sumLoss
from .warp_layer import SpatialTransformer

class Motion_Model(BaseModel):
    def name(self):
        return 'Motion_Model'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(which_model_netG='unet_256')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=5000.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        #self.isTrain = True
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_mse', \
            'G24_gradient', 'G48_gradient', 'G96_gradient', \
            'G_angle', 'G_distance', 'G_field', 'G_simiangle', \
            'G24_L1', 'G48_L1', 'G96_L1']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_A', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        #self.warp = Dense3DSpatialTransformer(96,96,96)
        # load/define networks
        if opt.which_model_netG == 'vnet':
            self.netG = VNet()
            self.netG.to(self.device)
            self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)
        elif opt.which_model_netG == 'net3d':
            self.netG = Net_3d()
            self.netG.to(self.device)
            self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)
        elif opt.which_model_netG == 'motion':
            self.netG = Net_3d_ALL()
            self.netG.to(self.device)
            self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)
        elif opt.which_model_netG == 'unet3d':
            self.netG = UNet3D_seg(1,2)
            self.netG.to(self.device)
            self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)
        elif opt.which_model_netG == 'unet3d_deep':
            self.netG = UNet3D_deep(1,2)
            self.netG.to(self.device)
            self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)
        else:
            self.netG = networks.define_G(opt.input_nc, opt.output_nc*2, opt.ngf, opt.which_model_netG,
                                        opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if opt.which_model_netG == 'motion' or opt.which_model_netG == 'unet3d_deep' or opt.which_model_netG == 'vnet3d':
                self.netD = PixelDiscriminator(input_nc=1)
                self.netD.to(self.device)
                self.netD = torch.nn.DataParallel(self.netD, self.gpu_ids)
            else:
                self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            #self.criterionL1 = DICELossMultiClass().to(self.device)
            #self.criterionL1 = torch.nn.MSELoss().to(self.device)
            self.criterion_gradient = gradientLoss().to(self.device)
            self.criterion_similar = similarLoss().to(self.device)
            self.criterion_angle = angleLoss().to(self.device)
            self.criterion_distance = distanceLoss().to(self.device)
            self.criterion_field = fieldLoss().to(self.device)
            self.criterion_simiangle = simiangleLoss().to(self.device)
            self.criterion_sum = sumLoss().to(self.device)

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            if opt.which_model_netG == 'motion' or opt.which_model_netG == 'unet3d' or opt.which_model_netG == 'unet3d_deep' or opt.which_model_netG == 'vnet3d' or opt.which_model_netG == 'vnet3d1':
                self.netG.load_state_dict(torch.load(opt.net3d_dir_G))
                self.netD.load_state_dict(torch.load(opt.net3d_dir_D))
                print('Load Done!')
            else:
                self.load_networks(opt.which_epoch)

        self.print_networks(opt.verbose)

    def set_input(self, combined_image21_96, combined_image21_48, combined_image21_24, combined_image12_96, combined_image12_48, combined_image12_24, loss_w):
        #AtoB = self.opt.which_direction == 'AtoB'
        self.real96_A1 = torch.unsqueeze(combined_image21_96[:,0,:,:,:],0).to(self.device)
        self.real48_A1 = torch.unsqueeze(combined_image21_48[:,0,:,:,:],0).to(self.device)
        self.real24_A1 = torch.unsqueeze(combined_image21_24[:,0,:,:,:],0).to(self.device)

        self.real96_A2 = torch.unsqueeze(combined_image12_96[:,0,:,:,:],0).to(self.device)
        self.real48_A2 = torch.unsqueeze(combined_image12_48[:,0,:,:,:],0).to(self.device)
        self.real24_A2 = torch.unsqueeze(combined_image12_24[:,0,:,:,:],0).to(self.device)

        self.combined_image21_96 = combined_image21_96.to(self.device)
        self.combined_image21_48 = combined_image21_48.to(self.device)
        self.combined_image21_24 = combined_image21_24.to(self.device)

        self.combined_image12_96 = combined_image12_96.to(self.device)
        self.combined_image12_48 = combined_image12_48.to(self.device)
        self.combined_image12_24 = combined_image12_24.to(self.device)

        self.loss_w = loss_w.to(self.device).float()
        #self.real_mask = fix_mask.long().to(self.device)
        #self.real_A, self.real_B = Variable(self.real_A), Variable(self.real_B)
        #self.image_paths = input['A_paths' if AtoB else 'B_paths']
    def set_input_test(self, combined_image, fix_image, move_image):
        #AtoB = self.opt.which_direction == 'AtoB'
        self.combined_image = combined_image.to(self.device)
        self.real_A = fix_image.to(self.device)
        self.real_B = move_image.to(self.device)
        #self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def test(self):
        self.predict_A, self.predict_B = self.netG(self.real_A, self.real_B)
        self.field, self.fake_A = self.netD(self.combined_image, self.real_B)
        _, mask_a = torch.max(self.predict_A,1)
        #self.fake_B = self.predict
        #self.fake_B = self.fake_B.float()
        #self.fake_B = torch.unsqueeze(self.fake_B, dim=1)

        return mask_a, self.field, self.fake_A

    def forward(self):
        self.field12_24, self.field12_48, self.field12_96, \
        self.fake2_24, self.fake2_48, self.fake2_96, \
        self.field21_24, self.field21_48, self.field21_96, \
        self.fake1_24, self.fake1_48, self.fake1_96 = \
            self.netG(self.combined_image12_96, self.combined_image21_96, \
                self.real96_A1, self.real48_A1, self.real24_A1, \
                self.real96_A2, self.real48_A2, self.real24_A2)
        #self.predict_A, self.predict_fakeB = self.netG(self.field, self.real_A, self.real_B, None)
        #_, self.p_A = torch.max(self.predict_A,1)
        #_, self.p_B = torch.max(self.predict_fakeB,1)
        #p_B_tmp = torch.unsqueeze(self.p_B,1)
        #self.B_to_A = self.warp(p_B_tmp, self.field[:,0,:,:,:],self.field[:,1,:,:,:],self.field[:,2,:,:,:])
        #self.predict=self.predict.long()
        #self.fake_B = self.fake_B.float()
        #self.fake_B = torch.unsqueeze(self.fake_B, dim=1)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        #fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_A), 1))
        pred_fake = self.netD(self.fake2.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        pred_real = self.netD(self.real_A2)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        #self.loss_D_seg = self.criterion_dice(self.p_A, self.B_to_A) * 5000

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):

        #pred_fake = self.netD(self.fake1)
        #self.loss_D_fake = self.criterionGAN(pred_fake, True)
        
        self.loss_G24_L1 = (self.criterionL1(self.fake2_24, self.real24_A2) + \
                                    self.criterionL1(self.fake1_24, self.real24_A1)) * 50 * self.loss_w
        self.loss_G48_L1 = (self.criterionL1(self.fake2_48, self.real48_A2) + \
                                    self.criterionL1(self.fake1_48, self.real48_A1)) * 50 * self.loss_w
        self.loss_G96_L1 = (self.criterionL1(self.fake2_96, self.real96_A2) + \
                                    self.criterionL1(self.fake1_96, self.real96_A1)) * 50 * self.loss_w
        self.loss_G_mse = (self.criterion_similar(self.real96_A1, self.fake1_96) + \
                                    self.criterion_similar(self.real96_A2, self.fake2_96)) * 50
        self.loss_G24_gradient = (self.criterion_gradient(self.field12_24) + \
                                        self.criterion_gradient(self.field21_24)) * 50
        self.loss_G48_gradient = (self.criterion_gradient(self.field12_48) + \
                                        self.criterion_gradient(self.field21_48)) * 50
        self.loss_G96_gradient = (self.criterion_gradient(self.field12_96) + \
                                        self.criterion_gradient(self.field21_96)) * 50
        self.loss_G_angle = self.criterion_angle(self.field12_96) + \
                                self.criterion_angle(self.field21_96)
        self.loss_G_distance = self.criterion_distance(self.field12_96) + \
                                    self.criterion_distance(self.field21_96)
        self.loss_G_field = (self.criterion_field(self.field12_96) + \
                                        self.criterion_field(self.field21_96)) * 50
        self.loss_G_simiangle = (self.criterion_simiangle(self.field12_96) + \
                                            self.criterion_simiangle(self.field21_96)) * 100
        
        

        #self.loss_G_L1 = self.loss_G_L1.float()

        self.loss_G = \
            self.loss_G24_L1 + self.loss_G48_L1 + self.loss_G96_L1 + \
            self.loss_G24_gradient + self.loss_G48_gradient + self.loss_G96_gradient #+ \
            #self.loss_G_sum
            #self.loss_G_simiangle + self.loss_G_sum + self.loss_D_fake
        

        # First, G(A) should fake the discriminator
        '''pred_fake = self.netD(self.fake1)
        self.loss_D_fake = self.criterionGAN(pred_fake, True)
        
        self.loss_G_L1 = self.criterionL1(self.fake1, self.real_A1)
        self.loss_G_mse = self.criterion_similar(self.real_A1, self.fake1) * 200
        self.loss_G_gradient = self.criterion_gradient(self.field) * 200

        #self.loss_G_L1 = self.loss_G_L1.float()

        self.loss_G = self.loss_G_mse + self.loss_G_gradient + self.loss_D_fake'''
        

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        #self.set_requires_grad(self.netD, True)
        #self.optimizer_D.zero_grad()
        #self.backward_D()
        #self.optimizer_D.step()

        # update G
        #self.set_requires_grad(self.netG, True)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


    def save_net(self, epoch):
        torch.save(\
            self.netG.state_dict(), '{}/net{}-epoch{}'.format(self.opt.checkpoints_dir, 'G', epoch))
        #torch.save(\
            #self.netD.state_dict(), '{}/net{}-epoch{}'.format(self.opt.checkpoints_dir, 'D', epoch))

        fake_tmp96 = self.fake2_96[0,0,:,:,:].cpu().data.numpy()
        fake_tmp96 = sitk.GetImageFromArray(fake_tmp96)
        real2_tmp96 = self.real96_A2[0,0,:,:,:].cpu().data.numpy()
        real2_tmp96 = sitk.GetImageFromArray(real2_tmp96)
        real1_tmp96 = self.real96_A1[0,0,:,:,:].cpu().data.numpy()
        real1_tmp96 = sitk.GetImageFromArray(real1_tmp96)
        sitk.WriteImage(fake_tmp96,'test_fake96.nii')
        sitk.WriteImage(real2_tmp96,'test_real96.nii')
        sitk.WriteImage(real1_tmp96,'test_move96.nii')

        fake_tmp48 = self.fake2_48[0,0,:,:,:].cpu().data.numpy()
        fake_tmp48 = sitk.GetImageFromArray(fake_tmp48)
        real2_tmp48 = self.real48_A2[0,0,:,:,:].cpu().data.numpy()
        real2_tmp48 = sitk.GetImageFromArray(real2_tmp48)
        real1_tmp48 = self.real48_A1[0,0,:,:,:].cpu().data.numpy()
        real1_tmp48 = sitk.GetImageFromArray(real1_tmp48)
        sitk.WriteImage(fake_tmp48,'test_fake48.nii')
        sitk.WriteImage(real2_tmp48,'test_real48.nii')
        sitk.WriteImage(real1_tmp48,'test_move48.nii')

        fake_tmp24 = self.fake2_24[0,0,:,:,:].cpu().data.numpy()
        fake_tmp24 = sitk.GetImageFromArray(fake_tmp24)
        real2_tmp24 = self.real24_A2[0,0,:,:,:].cpu().data.numpy()
        real2_tmp24 = sitk.GetImageFromArray(real2_tmp24)
        real1_tmp24 = self.real24_A1[0,0,:,:,:].cpu().data.numpy()
        real1_tmp24 = sitk.GetImageFromArray(real1_tmp24)
        sitk.WriteImage(fake_tmp24,'test_fake24.nii')
        sitk.WriteImage(real2_tmp24,'test_real24.nii')
        sitk.WriteImage(real1_tmp24,'test_move24.nii')
