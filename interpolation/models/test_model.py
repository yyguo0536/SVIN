from .base_model import BaseModel
from . import networks
from .interpolation import Net_3d_Scale
import torch
from .warp_layer import SpatialTransformer
import torch.nn.functional as F

class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(dataset_mode='single')
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')

        parser.add_argument('--model_suffix', type=str, default='',
                            help='In checkpoints_dir, [which_epoch]_net_G[model_suffix].pth will'
                            ' be loaded as the generator of TestModel')

        return parser

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        self.warp = SpatialTransformer(96,96,96)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G' + opt.model_suffix]

        if opt.which_model_netG == 'vnet':
            self.netG = VNet()
            self.netG.to(self.device)
            self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)
        elif opt.which_model_netG == 'interpolation':
            self.netG = Net_3d_Scale(7)
            self.netG.to(self.device)
            self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)
        else:
            self.netG = networks.define_G(opt.input_nc*3, opt.output_nc*2, opt.ngf, opt.which_model_netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see BaseModel.load_networks
        setattr(self, 'netG' + opt.model_suffix, self.netG)
        if opt.which_model_netG == 'vnet' or opt.which_model_netG == 'interpolation':
            self.netG.load_state_dict(torch.load(opt.net3d_dir_G))
        else:
            self.load_networks(opt.which_epoch)
            self.print_networks(opt.verbose)

    def set_input(self, combined_image, img0_96,  \
        img1_96, D0t_96, D1t_96, D0t_48, D1t_48, D0t_24, D1t_24):
        #AtoB = self.opt.which_direction == 'AtoB'
        #self.real96_A = imgt_96.to(self.device)
        self.combined_image = combined_image.to(self.device)
        self.img0_96 = img0_96.to(self.device)
        self.img1_96 = img1_96.to(self.device)
        self.D0t_96 = D0t_96.to(self.device)
        self.D1t_96 = D1t_96.to(self.device)
        self.D0t_48 = D0t_48.to(self.device)
        self.D1t_48 = D1t_48.to(self.device)
        self.D0t_24 = D0t_24.to(self.device)
        self.D1t_24 = D1t_24.to(self.device)
        #self.D96_combine = torch.cat([self.D0t_96, self.D1t_96], 1)
        self.D96_combine = torch.cat([self.D0t_96, self.D1t_96], 1)
        self.D48_combine = torch.cat([self.D0t_48, self.D1t_48], 1)
        self.D24_combine = torch.cat([self.D0t_24, self.D1t_24], 1)
        #self.index_l = index_l.to(self.device)
        #self.image_paths = input['A_paths']

    def test(self):
        with torch.no_grad():
            _, _, self.outputs_96, self.regressionNum, _ = self.netG(self.combined_image, self.D96_combine, self.D48_combine, self.D24_combine)
        D0t_f = self.outputs_96[:, :3, :, :, :]
        D1t_f = self.outputs_96[:, 3:6, :, :, :]
        V_t_0   = F.sigmoid(self.outputs_96[:, 6:7, :, :, :])
        V_t_1   = 1 - V_t_0
        
        # Get intermediate frames from the intermediate flows
        imgt0_f = self.warp(self.img0_96, D0t_f[:,0,:,:,:], D0t_f[:,2,:,:,:], D0t_f[:,1,:,:,:])
        imgt1_f = self.warp(self.img1_96, D1t_f[:,0,:,:,:], D1t_f[:,2,:,:,:] ,D1t_f[:,1,:,:,:])

        #self.imgt0_f = imgt0_f
        #self.imgt1_f = imgt1_f
        
        # Calculate final intermediate frame 
        self.imgt_f = (0.5 * V_t_0 * imgt0_f + 0.5 * V_t_1 * imgt1_f) / (0.5 * V_t_0 + 0.5 * V_t_1)
        return self.imgt_f, D0t_f, D1t_f, V_t_0
