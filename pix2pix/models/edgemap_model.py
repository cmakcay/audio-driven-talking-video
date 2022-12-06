import torch
from .base_model import BaseModel
from . import networks
from torchvision import transforms
from .utils import erode_mask

class EdgemapModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.visual_names = ['fake', 'edge_map', 'face', 'background']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G','place']

        # define networks
        # edge map -> mouth  
        self.netG = networks.define_G(4, 3, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        # mouth + eroded background -> face
        self.netplace = networks.define_G(6, 3, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # define loss functions
        self.criterionL1 = torch.nn.L1Loss()
        
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_place = torch.optim.Adam(self.netplace.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_place)

    def set_input(self, input):
        self.edge_map = input["edge_map"].to(self.device)
        self.face = input["face_image"].to(self.device)
        self.mouth_mask = input["mask"][0]

    def forward(self):
        # create the mask bool
        self.mask = torch.where(self.mouth_mask==0, torch.zeros_like(self.mouth_mask, dtype=torch.bool), torch.ones_like(self.mouth_mask, dtype=torch.bool)).to(self.device)
        self.mask = self.mask.repeat(1,3,1,1)

        # erode the mask
        self.eroded_mask = erode_mask(self.mask)

        # get background
        self.background = torch.where(self.eroded_mask, torch.zeros_like(self.face), self.face)

        # forward pass of generators
        self.intermediate_fake = self.netG(torch.cat((self.edge_map, self.background),1))         
        self.fake = self.netplace(torch.cat((self.intermediate_fake, self.background),1))

    def backward_G(self):
        # weighting the loss according to the size of the mask, otherwised will be biased on size
        mask_true = len(self.mask[self.mask==True])
        num_pixels = self.mask.size(2) * self.mask.size(3)
        mask_weight = num_pixels / mask_true
                
        self.loss_G_L1 = self.criterionL1(self.intermediate_fake, self.face)
        self.loss_G_L1 += 5. * self.criterionL1(torch.where(self.mask, self.intermediate_fake, torch.zeros_like(self.intermediate_fake)), \
                                                torch.where(self.mask, self.face, torch.zeros_like(self.face))) * mask_weight
        self.loss_G_L1 += 10. * self.criterionL1(self.fake, self.face)
        self.loss_G = self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G        
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.optimizer_place.zero_grad()

        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        self.optimizer_place.step()