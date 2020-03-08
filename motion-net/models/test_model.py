from .base_model import BaseModel
from . import networks
from .cycle_gan_model import CycleGANModel
from .vnet import VNet
from .net_3d import Net_3d, Net_3d_Test
from .unet_3d import UNet3D, UNet3D_deep, PixelDiscriminator, UNet3D_FLOW, UNet3D_seg, UNet3D_FLOW_no
import torch



class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
        parser = CycleGANModel.modify_commandline_options(parser, is_train=False)
        parser.set_defaults(dataset_mode='single')
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')

        parser.add_argument('--model_suffix', type=str, default='',
                            help='In checkpoints_dir, [which_epoch]_net_G[model_suffix].pth will'
                            ' be loaded as the generator of TestModel')

        return parser

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

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
        elif opt.which_model_netG == 'motion':
            self.netG = Net_3d_Test()
            self.netG.to(self.device)
            self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)
        elif opt.which_model_netG == 'unet3d':
            self.netG = UNet3D_FLOW_no(2,3)
            self.netG.to(self.device)
            self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)
        elif opt.which_model_netG == 'unet3d_deep':
            self.netG = UNet3D_deep(1,2)
            self.netG.to(self.device)
            self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)
        else:
            self.netG = networks.define_G(opt.input_nc*3, opt.output_nc*2, opt.ngf, opt.which_model_netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see BaseModel.load_networks
        setattr(self, 'netG' + opt.model_suffix, self.netG)
        if opt.which_model_netG == 'vnet' or opt.which_model_netG == 'unet3d_deep':
            self.netG.load_state_dict(torch.load(opt.net3d_dir_G))
        elif opt.which_model_netG == 'vnet3d' or opt.which_model_netG == 'motion':
            self.netG.load_state_dict(torch.load(opt.net3d_dir_G))
        else:
            self.load_networks(opt.which_epoch)
            self.print_networks(opt.verbose)

    def set_input(self, combined_image1, combined_image2, move_image, fix_image):
        # we need to use single_dataset mode
        self.combined_image1 = combined_image1.to(self.device)
        self.move_image = move_image.to(self.device)
        self.combined_image2 = combined_image2.to(self.device)
        self.fix_image = fix_image.to(self.device)
        #self.image_paths = input['A_paths']

    def test(self):
        field12, field21= self.netG(self.combined_image1, self.combined_image2, self.move_image, self.fix_image)
        return field12, field21