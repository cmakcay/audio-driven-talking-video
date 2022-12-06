import configargparse
from deepspeech.ds_features import get_ds
from pathlib import Path
import os
import torch
"""
    Given the video directory, extract DeepSpeech features
"""

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--pipeline_files_path", type=str)
    parser.add_argument("--source_name", type=str)
    parser.add_argument("--target_name", type=str)
    parser.add_argument("--video_folder", type=str)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--target_fps", type=float, default=25.0)
    
    return parser


if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    deepspeech_model_path = f"{args.pipeline_files_path}/trained_models/output_graph.pb"
    if args.inference: 
        ds_features_save_path = f"{args.dataset_path}/{args.source_name}/targets/{args.target_name}/ds_features"
        
        args.num_frames=None # calculate num frames based on target fps and audio length for inference
    else: 
        ds_features_save_path = f"{args.dataset_path}/{args.source_name}/source/ds_features"

        # for training, we need #flame params = #deepspeech features
        temp_exp = torch.load(f"{args.dataset_path}/{args.source_name}/source/flame_params/exp.pt", map_location="cpu")
        args.num_frames = int(float(temp_exp.shape[0]))
    

    ds_output_path = Path(ds_features_save_path)
    if not os.path.exists(ds_output_path):
        os.makedirs(ds_output_path)    
    
    args.deepspeech_model_path = deepspeech_model_path
    args.ds_output_path = ds_output_path
    get_ds(args)