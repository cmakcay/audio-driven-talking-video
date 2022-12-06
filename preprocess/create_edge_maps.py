'''
Create edge maps, the source can be either from tracked source video, or tracked source video + output of a2e network
'''

import os
import cv2
import torch
import pandas as pd
import configargparse
from PIL import Image, ImageFilter
from utils import draw_landmarks
from tracker.models import net
from tracker.utils.utils import transform_landmarks, orthographic_projection, backproject_vertices

def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--pipeline_files_path", type=str)
    parser.add_argument("--source_name", type=str)
    parser.add_argument("--target_name", type=str)


    parser.add_argument("--mouth_mask_path", type=str, default="./tracker/data/face_mask_small.npy")
    parser.add_argument("--mouth_mask_faces_path", type=str, default="./tracker/data/face_mask_small_faces.pt")
    parser.add_argument("--width", type=int, default=224, help="width")
    parser.add_argument("--height", type=int, default=224, help="height")

    # FLAME params
    parser.add_argument('--shape_params', type=int, default=100, help='the number of shape parameters')
    parser.add_argument('--expression_params', type=int, default=50, help='the number of expression parameters')
    parser.add_argument('--pose_params', type=int, default=6, help='the number of pose parameters')
    parser.add_argument("--num_classes", type=int, default=159, help="number of encoder parameters to predict")
    
    return parser
    

if __name__ == '__main__':

    # load the configs
    parser = config_parser()
    args = parser.parse_args()

    args.flame_lmk_embedding_path = f"{args.pipeline_files_path}/flame_files/landmark_embedding.npy"
    args.flame_model_path = f"{args.pipeline_files_path}/flame_files/generic_model.pkl"
    
    # set device
    device = "cpu" # cpu is faster than cuda, as not so computationally heavy

    # initialize the network only to use FLAME layer
    network = net.Encoder(args, device).to(device)
    network.eval()
    torch.set_grad_enabled(False)

    data_list = pd.read_csv(f"{args.dataset_path}/{args.source_name}/source/frame_meta.csv")
    shape_params = torch.load(f"{args.dataset_path}/{args.source_name}/source/flame_params/shape.pt", map_location=device).to(device)
    pose_params = torch.load(f"{args.dataset_path}/{args.source_name}/source/flame_params/pose.pt", map_location=device).to(device)
    cam_params = torch.load(f"{args.dataset_path}/{args.source_name}/source/flame_params/cam.pt", map_location=device).to(device)

    if args.inference:
        save_path = f"{args.dataset_path}/{args.source_name}/targets/{args.target_name}"
        a2e_params  = torch.load(f"{args.dataset_path}/{args.source_name}/targets/{args.target_name}/a2e_params.pt", map_location=device).to(device)
        num_frames = a2e_params.shape[0]
    else:
        save_path = f"{args.dataset_path}/{args.source_name}/source"
        exp_params = torch.load(f"{args.dataset_path}/{args.source_name}/source/flame_params/exp.pt", map_location=device).to(device)
        num_frames = exp_params.shape[0]
     

    edge_map_save_path = f"{save_path}/edge_maps"
    if not os.path.exists(edge_map_save_path):
        os.makedirs(edge_map_save_path)

    masks_save_path = f"{save_path}/masks"
    if not os.path.exists(masks_save_path):
        os.makedirs(masks_save_path)
        
    for index in range(num_frames):
        # if inference frames are more than source frames, start from beginning
        index = index % len(data_list)
        
        meta = data_list.iloc[index, :]

        frame_idx = meta[0]
        face_box = meta[1:5].to_numpy()
        
        full_image = cv2.imread(f"{args.dataset_path}/{args.source_name}/source/frames/{frame_idx}.jpg")
        full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)

        face_crop = full_image[face_box[1]:face_box[3], face_box[0]:face_box[2]]

        orig_height, orig_width = face_box[2] - face_box[0], face_box[3] - face_box[1]        
        mod_height, mod_width = args.height, args.width
 
        # calculate dimension transforms
        tf_x, tf_y = torch.tensor(mod_height/orig_height)[None,...].to(device), torch.tensor(mod_width/orig_width)[None,...].to(device) 

        # Acquire vertices and landmarks from the flame layer with encoded features
        if args.inference:
            pose = pose_params[frame_idx]
            pose[3:] = a2e_params[frame_idx,50:]
            exp = a2e_params[frame_idx,:50]
        else:
            pose = pose_params[frame_idx]
            exp = exp_params[frame_idx]
        
        vertices, landmarks, _ = network.flame_layer(shape_params[None,...], exp[None,...], pose[None,...])
        scale, tran = cam_params[frame_idx,0][None,...], cam_params[frame_idx,1:][None,...]

        # Transform FLAME landmarks same ratio as original image landmarks
        transformed_landmarks = transform_landmarks(landmarks, (tf_x, tf_y))

        # Project landmarks on 2D with scaling and translation parameters
        projected_landmarks = orthographic_projection(transformed_landmarks, scale, tran)
        _, _, landmarks = backproject_vertices(projected_landmarks, vertices, (tf_x, tf_y), scale)
        
        face_grey = Image.fromarray(face_crop).convert("L")
        face_edges = face_grey.filter(ImageFilter.FIND_EDGES)
        
        draw_landmarks(landmarks, face_edges, (orig_height, orig_width), save_path, frame_idx)

            

