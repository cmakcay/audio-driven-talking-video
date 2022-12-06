from a2e.datasets.base import build_dataloader
from a2e.models.audio2exp import Audio2Exp
import torch
import configargparse
import numpy as np
import pandas as pd

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--pipeline_files_path", type=str)
    parser.add_argument("--source_name", type=str)
    
    parser.add_argument("--filter_size", type=int, default=8, help="filter T")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size of audio2exp")
    parser.add_argument("--num_workers", type=int, default=2, help="number of workers to use during batch generation")
    parser.add_argument("--parameter_space", type=int, default=53, help="50 exp + 3 jaw")
 
    return parser

def load_model(network, args, device):
    checkpoint_path = f"{args.pipeline_files_path}/trained_models/a2e.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    network.load_state_dict(state_dict)
    print("-> loaded checkpoint %s (epoch: %d)" % (checkpoint_path, checkpoint['epoch']))

if __name__ == "__main__":

    parser = config_parser()
    args = parser.parse_args()

    args.ds_only = False
    
    # check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running code on", device)

    network = Audio2Exp(args).to(device)
    load_model(network, args, device)
    network.eval()
    torch.set_grad_enabled(False)
    
    args.test_num_frames = len(pd.read_csv(f"{args.dataset_path}/{args.source_name}/source/frame_meta.csv"))
    args.flame_params_path = f"{args.dataset_path}/{args.source_name}/source/flame_params"
    args.ds_features_path = f"{args.dataset_path}/{args.source_name}/source/ds_features"
    demo_loader = build_dataloader(args, drop_last=True)    
    
    per_frame_buffer = []
    gt_buffer = []

    for batch_idx, data in enumerate(demo_loader):
        ds = data["ds_bf"].to(device)
        pose = data["pose"].to(device)
        exp = data["exp"].to(device)

        #flame params: batch size x 53
        #per frame filtered ds: batch size x 32
        _, filtered_ds = network(ds)
        per_frame_buffer.append(filtered_ds.detach().cpu().numpy())       
        
        gt_tensor = torch.zeros(args.batch_size, 53)
        gt_tensor[:,:50], gt_tensor[:,50:] = exp, pose[:,3:]
        gt_buffer.append(gt_tensor.detach().numpy())

    per_frame_buffer_tensor = torch.tensor(per_frame_buffer).view(-1,32)

    gt_buffer_tensor = torch.tensor(gt_buffer).view(-1,53)

    person_specific = torch.linalg.lstsq(per_frame_buffer_tensor, gt_buffer_tensor).solution

    torch.save(person_specific, f"{args.dataset_path}/{args.source_name}/source/person_specific.pt")