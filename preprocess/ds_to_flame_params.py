'''
Get FLAME parameters from deepspeech features.
'''
from a2e.datasets.base import build_dataloader
import torch
from a2e.models.audio2exp import Audio2Exp
import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()

    # jaw tuning
    parser.add_argument("--jaw_gain", type=float, default=0.4)
    parser.add_argument("--jaw_closure", type=float, default=0.15)
    #  

    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--pipeline_files_path", type=str)
    parser.add_argument("--source_name", type=str)
    parser.add_argument("--target_name", type=str)

    parser.add_argument("--filter_size", type=int, default=8, help="filter T")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size of audio2exp")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers to use during batch generation")
    parser.add_argument("--parameter_space", type=int, default=53, help="50 exp + 3 jaw")

    return parser

def load_model(network, args, device):
    checkpoint = torch.load(args.checkpoint_model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    network.load_state_dict(state_dict)
    print("-> loaded checkpoint %s (epoch: %d)" % (args.checkpoint_model_path, checkpoint['epoch']))

if __name__ == "__main__":

    # check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running code on", device)

    parser = config_parser()
    args = parser.parse_args()

    args.ds_features_path = f"{args.dataset_path}/{args.source_name}/targets/{args.target_name}/ds_features"
    args.checkpoint_model_path = f"{args.pipeline_files_path}/trained_models/a2e.pt"
    args.ds_only = True

    network = Audio2Exp(args).to(device)
    load_model(network, args, device)
    network.eval()
    torch.set_grad_enabled(False)
    
    demo_loader = build_dataloader(args)    
    flame_params_buf = []

    person_specific = torch.load(f"{args.dataset_path}/{args.source_name}/source/person_specific.pt").to(device)

    for batch_idx, data in enumerate(demo_loader):
        ds = data["ds_bf"].to(device)
        
        # filtering net needs all batches to be 16
        current_batch = ds.shape[0]
        if current_batch != 16:
            ds = torch.cat((ds, ds[-1:].repeat(16-current_batch,1,1,1)), 0)
        _, filtered_ds = network(ds) #batch size x 53
        
        # back to original size
        filtered_ds = filtered_ds[:current_batch]
        flame_params =  torch.matmul(filtered_ds, person_specific)
        flame_params_buf.append(flame_params)

    flame_params_buf_tensor = torch.cat(flame_params_buf)
    
    # get jaw opening corresponding to "blank" character (i.e. silence).
    ds = torch.zeros(16,8,29,16).to(device)
    ds-=5
    ds[:,:,-1,:] = 15 # logit 29 is "blank" character
    _, filtered_ds = network(ds)
    blank_flame_params = torch.matmul(filtered_ds, person_specific)

    # scale FLAME parameters
    jaw = flame_params_buf_tensor[:,50:51].clone()
    jaw_min = torch.min(jaw, dim=0)[0]
    jaw_max = torch.max(jaw, dim=0)[0]
    jaw_std = (jaw - jaw_min)/(jaw_max-jaw_min)
    jaw = args.jaw_gain * jaw_std

    # subtract blank FLAME parameters
    blank_jaw = args.jaw_closure * (blank_flame_params[:,50:51] - jaw_min)/(jaw_max-jaw_min)
    jaw -= blank_jaw[0:1,:].repeat(jaw.shape[0], 1)
    jaw[jaw<0]=0 # should not be negative
    flame_params_buf_tensor[:,50:51] = jaw

    torch.save(flame_params_buf_tensor, f"{args.dataset_path}/{args.source_name}/targets/{args.target_name}/a2e_params.pt")

