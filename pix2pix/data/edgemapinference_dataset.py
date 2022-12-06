'''
Dataset to load inference data
'''
from .base_dataset import BaseDataset
import torchvision.transforms as transforms
import torch
import pandas as pd
from PIL import Image
import os

class EdgemapinferenceDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.is_train = self.opt.isTrain

        # paths to read data
        self.edge_maps_path = f"{self.root}/{self.opt.name}/targets/{self.opt.target_name}/edge_maps"
        self.masks_path = f"{self.root}/{self.opt.name}/targets/{self.opt.target_name}/masks"
        data_list_path = f"{self.root}/{self.opt.name}/source/frame_meta.csv"

        # load and read the data list
        self.data_list = pd.read_csv(data_list_path)

        # length of the source dataset
        self.source_length = len(self.data_list)

        # length of the target video
        self.target_length = len([f for f in os.listdir(self.edge_maps_path) if f.endswith('.jpg')])

        # dimensions to transform images
        self.transform_resize = transforms.Resize([256, 256])

    def __getitem__(self, index):

        # if target sequence is longer than source, loop the source
        index = index % self.source_length
    
        # get meta data
        meta = self.data_list.iloc[index, :]
        frame_idx = meta[0] # index should coincide with frame index always if not shuffled, still to be safe
        face_box = meta[1:5].to_numpy() # pixel bounding box of face crop

        # load the edge map
        edge_map = transforms.ToTensor()(Image.open(f"{self.edge_maps_path}/{frame_idx}.jpg"))
        edge_map = edge_map[0,:,:] # get one channel only
        edge_map = self.transform_resize(edge_map[None,...]) # transform to canocial size

        # load the mask
        mask = transforms.ToTensor()(Image.open(f"{self.masks_path}/{frame_idx}.jpg"))
        mask = mask[0,:,:][None,...] # get one channel only
        mask = torch.where(mask>0.5, torch.ones_like(mask), torch.zeros_like(mask)) #binary filter
        mask = transforms.Resize((256,256), transforms.InterpolationMode.NEAREST)(mask) # transform to canocial size
        
        # load the full image
        full_image = transforms.ToTensor()(Image.open(f"{self.root}/{self.opt.name}/source/frames/{frame_idx}.jpg"))        

        # face crop
        face_orig = full_image[:,face_box[1]:face_box[3], face_box[0]:face_box[2]]
        face_orig = self.transform_resize(face_orig)
        
        return {'frame_idx': frame_idx, 
                'face_image': face_orig, 
                'full_image': full_image, 
                'face_box': face_box, 
                'edge_map': edge_map,
                'mask': mask}

    def __len__(self):
        """Return the total number of images."""
        return self.target_length