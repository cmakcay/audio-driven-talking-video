import torch
from .base_model import BaseModel
from . import networks
from torchvision import transforms
from .utils import erode_mask

class EdgemapinferenceModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=False):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.visual_names = ['fake_full_image']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G','place']

        # define networks
        # edge map -> mouth  
        self.netG = networks.define_G(4, 3, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        # mouth + eroded background -> face
        self.netplace = networks.define_G(6, 3, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

    def set_input(self, input):
        self.edge_map = input["edge_map"].to(self.device)
        self.face = input["face_image"].to(self.device)
        self.mouth_mask = input["mask"][0]
        self.face_orig_box = input["face_box"][0]
        self.full_image = input["full_image"].to(self.device)
        self.image_paths = str(float(input["frame_idx"])) # not used, should stay

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

        # place fake face image back to full image
        face_height = self.face_orig_box[3] - self.face_orig_box[1]
        face_width = self.face_orig_box[2] - self.face_orig_box[0]

        self.fake_full_image = self.full_image.clone()
        fake_face_orig_size = transforms.Resize([face_height, face_width])(self.fake)
        self.fake_full_image[:,:,self.face_orig_box[1]:self.face_orig_box[3], self.face_orig_box[0]:self.face_orig_box[2]] = fake_face_orig_size

    def optimize_parameters(self):
        pass

