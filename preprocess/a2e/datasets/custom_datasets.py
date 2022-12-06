import torch
from torch.utils.data import Dataset
import numpy as np
import os

class ds_demo(Dataset):
    def __init__(self, args):
        super().__init__()

        # Dataloader is used to load ds features only or ds features + flame params
        self.ds_only = args.ds_only

        self.ds_features_path = args.ds_features_path
        num_ds_features = len([f for f in os.listdir(self.ds_features_path) if f.endswith('.npy')])
        self.list_frames = list(range(num_ds_features))
        self.seq_length = len(self.list_frames)

        self.filter_size = args.filter_size

        if not self.ds_only:
            self.flame_params_path = args.flame_params_path        
            self.exp = torch.load(f"{self.flame_params_path}/exp.pt")
            self.pose = torch.load(f"{self.flame_params_path}/pose.pt")

    def __len__(self):
        return self.seq_length
    
    def __getitem__(self, frame_index):
        # get video name, which frame, and total length 
        T = self.filter_size
        
        ds_bf = torch.zeros(T, 16, 29) #Tx16x29       

        # store previous and future frames for filtering network
        for i in range(T):
            idx = frame_index + (i - T//2)
            
            # handle edges 
            if idx < 0 : idx = 0
            if idx > self.seq_length -1 : idx = self.seq_length -1
            # put into buffer
            ds_bf[i] = torch.from_numpy(np.load(f"{self.ds_features_path}/{idx}.npy"))

        ds_bf = torch.transpose(ds_bf, 1, 2) # 8x16x29 to 8x29x16

        if not self.ds_only:
            exp = self.exp[frame_index]
            pose = self.pose[frame_index]

            sample = {"exp": exp, #8x50
                    "pose": pose, #8x6
                    "ds_bf": ds_bf} #8x29x16
        else:
            sample = {"ds_bf": ds_bf} #8x29x16
        
        return sample