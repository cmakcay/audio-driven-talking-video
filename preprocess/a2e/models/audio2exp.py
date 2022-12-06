import torch
import torch.nn as nn

class Audio2Exp(nn.Module):
    def __init__(self, args):
        super(Audio2Exp, self).__init__()

        self.per_frame = PerFrameExp(args)
        self.filter = FilterWeight(args)

        self.filter_size = args.filter_size
        self.batch_size = args.batch_size

        # specialized expression basis to convert from Nx32 to Nx53
        self.exp_basis = nn.Parameter(torch.randn(size=(32, args.parameter_space), dtype=torch.float32))

    def forward(self, ds):
        # get per-frame coefficients
        pf = self.per_frame(ds) #128x32
        pf = pf.view(self.batch_size, self.filter_size,-1).transpose(1,2) #16x32x8

        # get filtering weights
        weights = self.filter(pf)
        weights = weights.view(weights.size(0), 1, -1)

        # filter per-frame coefficients
        filtered_ds = pf * weights # element-wise, 16x32x8 * 16x1x8 
        filtered_ds = filtered_ds.sum(dim=2) #16x32
        
        # get flame parameters by expression basis
        flame_params =  torch.matmul(filtered_ds, self.exp_basis)

        return flame_params, filtered_ds

class PerFrameExp(nn.Module):
    """
        Per frame audio to expression network
    """
    def __init__(self, args):
        super(PerFrameExp, self).__init__()

        # apply 1D convolution along time axis
        # each logit is a channel
        self.conv1 = nn.Conv2d(in_channels=29,out_channels=32, 
                            kernel_size=(3,1), stride=(2,1), padding=(1,0)) #29x16 to 32x8
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32, 
                            kernel_size=(3,1), stride=(2,1), padding=(1,0)) #32x8 to 32x4
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64, 
                            kernel_size=(3,1), stride=(2,1), padding=(1,0)) #32x4 to 64x2
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=64, 
                            kernel_size=(3,1), stride=(2,1), padding=(1,0)) #64x2 to 64x1
        
        # we have 1x64 vector, fully connected layer to extract expression parameters
        self.fc1 = nn.Linear(in_features=64, out_features=128) #64 to 128
        self.fc2 = nn.Linear(in_features=128, out_features=64) #128 to 64
        self.fc3 = nn.Linear(in_features=64, out_features=32) #64 to 32
        

        # activation functions
        self.leaky_relu_1 = nn.LeakyReLU(0.02, inplace=False)
        self.leaky_relu_2 = nn.LeakyReLU(0.2, inplace=False)
        self.tanh = nn.Tanh()                         
    
    def forward(self, ds):
        # deepspeech features are Nxfilter_sizex29x16
        ds = ds.reshape(ds.size(0)*ds.size(1), ds.size(2), ds.size(3))
        ds = ds[:,:,:,None] #128x29x16x1
        
        # conv
        ds = self.conv1(ds)
        ds = self.leaky_relu_1(ds)

        ds = self.conv2(ds)
        ds = self.leaky_relu_1(ds)

        ds = self.conv3(ds)
        ds = self.leaky_relu_2(ds)

        ds = self.conv4(ds)
        ds = self.leaky_relu_2(ds)
        ds = ds.squeeze()

        # fc
        ds = self.fc1(ds)
        ds = self.leaky_relu_1(ds)

        ds = self.fc2(ds)
        ds = self.leaky_relu_1(ds)

        ds = self.fc3(ds)
        pf = self.tanh(ds) # per-frame

        return pf

class FilterWeight(nn.Module):
    def __init__(self, args):
        super(FilterWeight, self).__init__()
        
        # T
        self.filter_size = args.filter_size

        # conv layers
        self.conv1 = nn.Conv1d(in_channels=32, out_channels=16, 
                            kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=8, 
                    kernel_size=3, stride=1, padding=1) 
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=4, 
                            kernel_size=3, stride=1, padding=1) 
        self.conv4 = nn.Conv1d(in_channels=4, out_channels=2, 
                            kernel_size=3, stride=1, padding=1) 
        self.conv5 = nn.Conv1d(in_channels=2, out_channels=1, 
                            kernel_size=3, stride=1, padding=1) 
        
        # fc layer
        self.fc = nn.Linear(in_features=self.filter_size, out_features=self.filter_size)

        # activation_functions
        self.leaky_relu = nn.LeakyReLU(0.02, inplace=False)
        self.soft_max = nn.Softmax(dim=1)
    
    def forward(self, pf):
        # conv
        pf = self.conv1(pf)
        pf = self.leaky_relu(pf)
        
        pf = self.conv2(pf)
        pf = self.leaky_relu(pf)
        
        pf = self.conv3(pf)
        pf = self.leaky_relu(pf)
        
        pf = self.conv4(pf)
        pf = self.leaky_relu(pf)

        pf = self.conv5(pf)
        pf = self.leaky_relu(pf)

        pf = pf.squeeze()

        #  fc
        pf = self.fc(pf)

        # softmax
        weights = self.soft_max(pf) # Tx1

        return weights
