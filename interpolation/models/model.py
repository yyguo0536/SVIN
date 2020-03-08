import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from scipy.misc import imsave
from .interpolation import Net_3d_Scale
import SimpleITK as sitk
from .MotionLoss import gradientLoss, similarLoss, DICELossMultiClass
from .warp_layer import SpatialTransformer
import torch.nn.functional as F

class InterpolationModel(BaseModel):
    def name(self):
        return 'InterpolationModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        
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
        self.loss_names = ['G_L1_96', 'G_L1_48', 'G_L1_24', 'G_R_img', 'G_R_def', 'G_gradient']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_A', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        self.warp96 = SpatialTransformer(96,96,96)
        self.warp48 = SpatialTransformer(48,48,48)
        self.warp24 = SpatialTransformer(24,24,24)
        # load/define networks
        if opt.which_model_netG == 'vnet':
            self.netG = VNet()
            self.netG.to(self.device)
            self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)
        elif opt.which_model_netG == 'interpolation':
            self.netG = Net_3d_Scale(7)
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
            #if opt.which_model_netG == 'unet3d' or opt.which_model_netG == 'unet3d_deep' or opt.which_model_netG == 'vnet3d':
                #self.netD = NLayerDiscriminator3D(input_nc=1)
                #self.netD.to(self.device)
                #self.netD = torch.nn.DataParallel(self.netD, self.gpu_ids)
            #else:
                #self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                        #opt.which_model_netD,
                                        #opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterion_gradient = gradientLoss().to(self.device)
            self.criterion_similar = similarLoss().to(self.device)

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                #lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            #self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            if opt.which_model_netG == 'interpolation' or opt.which_model_netG == 'unet3d_deep' or opt.which_model_netG == 'vnet3d':
                self.netG.load_state_dict(torch.load(opt.net3d_dir_G))
                #self.netD.load_state_dict(torch.load(opt.net3d_dir_D))
                print('Load Done!')
            else:
                self.load_networks(opt.which_epoch)

        self.print_networks(opt.verbose)

    def set_input(self, combined_image, img0_96, img0_48, img0_24, \
        img1_96, img1_48, img1_24, imgt_96, imgt_48, imgt_24, D0t_96, D1t_96, \
        D0t_48, D1t_48, D0t_24, D1t_24, index_l):

        self.real96_A = imgt_96.to(self.device)
        self.real48_A = imgt_48.to(self.device)
        self.real24_A = imgt_24.to(self.device)
        self.combined_image = combined_image.to(self.device)
        self.img0_96 = img0_96.to(self.device)
        self.img1_96 = img1_96.to(self.device)
        self.img0_48 = img0_48.to(self.device)
        self.img1_48 = img1_48.to(self.device)
        self.img0_24 = img0_24.to(self.device)
        self.img1_24 = img1_24.to(self.device)
        self.D0t_96 = D0t_96.to(self.device)
        self.D1t_96 = D1t_96.to(self.device)
        self.D0t_48 = D0t_48.to(self.device)
        self.D1t_48 = D1t_48.to(self.device)
        self.D0t_24 = D0t_24.to(self.device)
        self.D1t_24 = D1t_24.to(self.device)
        self.D96_combine = torch.cat([self.D0t_96, self.D1t_96], 1)
        self.D48_combine = torch.cat([self.D0t_48, self.D1t_48], 1)
        self.D24_combine = torch.cat([self.D0t_24, self.D1t_24], 1)
        self.index_l = index_l.to(self.device)

    def set_input_test(self, combined_image, img0_96, img0_48, img0_24, \
        img1_96, img1_48, img1_24, imgt_96, D0t_96, D1t_96, D0t_48, D1t_48,D0t_24, D1t_24, t_t):
        #AtoB = self.opt.which_direction == 'AtoB'
        self.netG.eval()
        with torch.no_grad():
            combined_image = combined_image.to(self.device)
            img0_96 = img0_96.to(self.device)
            img1_96 = img1_96.to(self.device)
            D0t_96 = D0t_96.to(self.device)
            D1t_96 = D1t_96.to(self.device)
            D96_combine = torch.cat([D0t_96, D1t_96], 1)

            _, _, outputs_96, regNum= self.netG(combined_image, D96_combine)

            D0t_f96 = outputs_96[:, :3, :, :, :]
            D1t_f96 = outputs_96[:, 3:6, :, :, :]
            V_t_0_96   = F.sigmoid(outputs_96[:, 6:7, :, :, :])
            V_t_1_96   = 1 - V_t_0_96

            imgt0_f96 = self.warp96(img0_96, D0t_f96[:,0,:,:,:], D0t_f96[:,2,:,:,:], D0t_f96[:,1,:,:,:])
            imgt1_f96 = self.warp96(img1_96, D1t_f96[:,0,:,:,:], D1t_f96[:,2,:,:,:] ,D1t_f96[:,1,:,:,:])

            imgt_f96 = (0.5 * V_t_0_96 * imgt0_f96 + 0.5 * V_t_1_96 * imgt1_f96) / (0.5 * V_t_0_96 + 0.5 * V_t_1_96)

        fake_tmp = imgt_f96[0,0,:,:,:].cpu().data.numpy()
        fake_tmp = sitk.GetImageFromArray(fake_tmp)
        real2_tmp = img0_96[0,0,:,:,:].cpu().data.numpy()
        real2_tmp = sitk.GetImageFromArray(real2_tmp)
        real1_tmp = img1_96[0,0,:,:,:].cpu().data.numpy()
        real1_tmp = sitk.GetImageFromArray(real1_tmp)
        sitk.WriteImage(fake_tmp,'valid_fake.nii')
        sitk.WriteImage(real2_tmp,'valid_move1.nii')
        sitk.WriteImage(real1_tmp,'valid_move2.nii')
        real_tmp = imgt_96[0,0,:,:,:].data.numpy()
        real_tmp = sitk.GetImageFromArray(real_tmp)
        sitk.WriteImage(real_tmp,'valid_real.nii')
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
        self.outputs_24, self.outputs_48, self.outputs_96, self.regressionNum_img, self.regressionNum_def= \
            self.netG(self.combined_image, self.D96_combine, self.D48_combine, self.D24_combine)
        D0t_f24 = self.outputs_24[:, :3, :, :, :]
        D1t_f24 = self.outputs_24[:, 3:6, :, :, :]
        V_t_0_24   = F.sigmoid(self.outputs_24[:, 6:7, :, :, :])
        V_t_1_24   = 1 - V_t_0_24

        D0t_f48 = self.outputs_48[:, :3, :, :, :]
        D1t_f48 = self.outputs_48[:, 3:6, :, :, :]
        V_t_0_48   = F.sigmoid(self.outputs_48[:, 6:7, :, :, :])
        V_t_1_48   = 1 - V_t_0_48

        D0t_f96 = self.outputs_96[:, :3, :, :, :]
        D1t_f96 = self.outputs_96[:, 3:6, :, :, :]
        V_t_0_96   = F.sigmoid(self.outputs_96[:, 6:7, :, :, :])
        V_t_1_96   = 1 - V_t_0_96

        
        # Get intermediate frames from the intermediate flows
        imgt0_f24 = self.warp24(self.img0_24, D0t_f24[:,0,:,:,:], D0t_f24[:,2,:,:,:], D0t_f24[:,1,:,:,:])
        imgt1_f24 = self.warp24(self.img1_24, D1t_f24[:,0,:,:,:], D1t_f24[:,2,:,:,:] ,D1t_f24[:,1,:,:,:])

        imgt0_f48 = self.warp48(self.img0_48, D0t_f48[:,0,:,:,:], D0t_f48[:,2,:,:,:], D0t_f48[:,1,:,:,:])
        imgt1_f48 = self.warp48(self.img1_48, D1t_f48[:,0,:,:,:], D1t_f48[:,2,:,:,:] ,D1t_f48[:,1,:,:,:])

        imgt0_f96 = self.warp96(self.img0_96, D0t_f96[:,0,:,:,:], D0t_f96[:,2,:,:,:], D0t_f96[:,1,:,:,:])
        imgt1_f96 = self.warp96(self.img1_96, D1t_f96[:,0,:,:,:], D1t_f96[:,2,:,:,:] ,D1t_f96[:,1,:,:,:])

        # Calculate final intermediate frame 
        self.imgt0_f24 = imgt0_f24
        self.imgt1_f24 = imgt1_f24

        self.imgt_f24 = (0.5 * V_t_0_24 * imgt0_f24 + 0.5 * V_t_1_24 * imgt1_f24) / (0.5 * V_t_0_24 + 0.5 * V_t_1_24)

        self.imgt0_f48 = imgt0_f48
        self.imgt1_f48 = imgt1_f48
        self.imgt_f48 = (0.5 * V_t_0_48 * imgt0_f48 + 0.5 * V_t_1_48 * imgt1_f48) / (0.5 * V_t_0_48 + 0.5 * V_t_1_48)


        self.imgt0_f96 = imgt0_f96
        self.imgt1_f96 = imgt1_f96
        self.imgt_f96 = (0.5 * V_t_0_96 * imgt0_f96 + 0.5 * V_t_1_96 * imgt1_f96) / (0.5 * V_t_0_96 + 0.5 * V_t_1_96)
        

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        #fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_A), 1))
        pred_fake = self.netD(self.imgt_f.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        pred_real = self.netD(self.real_A)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        #self.loss_D_seg = self.criterion_dice(self.p_A, self.B_to_A) * 5000

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        #pred_fake = self.netD(self.imgt_f)
        #self.loss_D_fake = self.criterionGAN(pred_fake, True) * 5.0
        self.loss_G_L1_96 = self.criterionL1(self.imgt_f96, self.real96_A) * 3000.0
        self.loss_G_L1_48 = self.criterionL1(self.imgt_f48, self.real48_A) * 3000.0
        self.loss_G_L1_24 = self.criterionL1(self.imgt_f24, self.real24_A) * 3000.0
            
        #prcpLoss = MSE_LossFn(vgg16_conv_4_3(Ft_p), vgg16_conv_4_3(IFrame))
        
        warpLoss96 = (self.criterionL1(self.imgt0_f96, self.real96_A) + self.criterionL1(self.imgt1_f96, self.real96_A)) * 200.0
        warpLoss48 = (self.criterionL1(self.imgt0_f48, self.real48_A) + self.criterionL1(self.imgt1_f48, self.real48_A)) * 200.0
        warpLoss24 = (self.criterionL1(self.imgt0_f24, self.real24_A) + self.criterionL1(self.imgt1_f24, self.real24_A)) * 200.0

        
        self.loss_G_L1 = self.loss_G_L1_96 + self.loss_G_L1_48 + self.loss_G_L1_24
        self.loss_G_gradient = warpLoss96 + warpLoss48 + warpLoss24
        self.loss_G_R_img = self.criterionL1(self.regressionNum_img, torch.unsqueeze(self.index_l,0)) * 50
        self.loss_G_R_def = self.criterionL1(self.regressionNum_def, torch.unsqueeze(self.index_l,0)) * 200
        #self.loss_G_L1 = self.loss_G_L1.float()

        self.loss_G = self.loss_G_L1 + self.loss_G_gradient + self.loss_G_R_img + self.loss_G_R_def #+ self.loss_D_fake
        

        self.loss_G.backward()

    def optimize_parameters(self):
        self.netG.train()
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
        fake_tmp = self.imgt_f96[0,0,:,:,:].cpu().data.numpy()
        fake_tmp = sitk.GetImageFromArray(fake_tmp)
        real2_tmp = self.img0_96[0,0,:,:,:].cpu().data.numpy()
        real2_tmp = sitk.GetImageFromArray(real2_tmp)
        real1_tmp = self.img1_96[0,0,:,:,:].cpu().data.numpy()
        real1_tmp = sitk.GetImageFromArray(real1_tmp)
        sitk.WriteImage(fake_tmp,'test_fake.nii')
        sitk.WriteImage(real2_tmp,'test_move1.nii')
        sitk.WriteImage(real1_tmp,'test_move2.nii')
        real_tmp = self.real96_A[0,0,:,:,:].cpu().data.numpy()
        real_tmp = sitk.GetImageFromArray(real_tmp)
        sitk.WriteImage(real_tmp,'test_real.nii')
