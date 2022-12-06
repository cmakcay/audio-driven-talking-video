import os
from os import path
import cv2
import csv
import torch
import numpy as np
import configargparse
import mediapipe as mp
from torchvision import transforms

from tracker.models import net
from tracker.utils.utils import filter_headpose


def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--pipeline_files_path", type=str)
    parser.add_argument("--source_name", type=str)    
    parser.add_argument("--video_folder", type=str)

    # defaults
    parser.add_argument("--detection_threshold", type=float, default=0.5, help="detection threshold for MediaPipe face detection")
    parser.add_argument("--batch_size", type=int, default=1, help="process frames sequentially")
    parser.add_argument("--num_classes", type=int, default=159, help="number of encoder parameters to predict")
    parser.add_argument("--width", type=int, default=224, help="width")
    parser.add_argument("--height", type=int, default=224, help="height")

    # FLAME params
    parser.add_argument('--shape_params', type=int, default=100, help='the number of shape parameters')
    parser.add_argument('--expression_params', type=int, default=50, help='the number of expression parameters')
    parser.add_argument('--pose_params', type=int, default=6, help='the number of pose parameters')
    
    return parser


def load_model(network, args, device):
    checkpoint_path = f"{args.pipeline_files_path}/trained_models/tracker.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    network.load_state_dict(state_dict)
    print("-> loaded checkpoint %s (epoch: %d)" % (checkpoint_path, checkpoint['epoch']))


if __name__ == '__main__':

    # load the configs
    parser = config_parser()
    args = parser.parse_args()

    args.flame_lmk_embedding_path = f"{args.pipeline_files_path}/flame_files/landmark_embedding.npy"
    args.flame_model_path = f"{args.pipeline_files_path}/flame_files/generic_model.pkl"

    # check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running code on", device)

    # # load mouth faces
    # mouth_region_mask = np.load(args.mouth_mask_path)
    # mouth_faces = torch.load(args.mouth_mask_faces_path)[None,...].to(device)

    # load the trained model
    network = net.Encoder(args, device).to(device)
    load_model(network, args, device)
    network.eval()
    torch.set_grad_enabled(False)
    faces = network.flame_layer.faces_tensor[None,...]

    # transform the image same as training data
    transform_image = transforms.Compose([transforms.ToTensor(), transforms.Resize([args.height, args.width]),
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # MediaPipe's Face Detection
    mp_face_detection = mp.solutions.face_detection

    if path.isfile(f"{args.video_folder}/{args.source_name}.mp4"): video_path = f"{args.video_folder}/{args.source_name}.mp4"
    elif path.isfile(f"{args.video_folder}/{args.source_name}.avi"): video_path = f"{args.video_folder}/{args.source_name}.avi"
    else: raise NotImplementedError("Source can only be a video with .mp4 or .avi extension")

    cap = cv2.VideoCapture(video_path)

    idx = 0

    # initialize vertices as none
    vertices = None

    # initialize the face detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=args.detection_threshold)

    # initialize FLAME parameters
    dataset_save_path = f"{args.dataset_path}/{args.source_name}/source"
    exp_stored, shape_stored, pose_stored, cam_stored = [],[],[],[]

    flame_params_save_path = f"{dataset_save_path}/flame_params"
    if not os.path.exists(flame_params_save_path):
        os.makedirs(flame_params_save_path)

    frames_save_path = f"{dataset_save_path}/frames"
    if not os.path.exists(frames_save_path):
        os.makedirs(frames_save_path)

    list_of_dicts = []

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            cv2.destroyAllWindows()
            print("Done!")
            break

        # should make it faster
        image.flags.writeable = False

        # copy RGB order image before normalizations etc.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = image.copy() 
    
        # get image dimensions
        img_shape = image.shape
        height, width = img_shape[0], img_shape[1]

        #get face detection results
        results = face_detection.process(image) 
        
        #change back to BGR for cv2 visualization
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        
        # check if there are detections
        if results.detections:
            # if there are multiple faces detected in the video, ignore the frame
            # print("Warning: Multiple faces detected, getting the one with highest confidence")
            scores = []
            for detection in results.detections:
                scores.append(detection.score[0])
            max_detection = results.detections[scores.index(max(scores))]

            # get and enlarge the bounding box
            bb = max_detection.location_data.relative_bounding_box
            x, y = bb.xmin*width, bb.ymin*height
            w, h = bb.width*width, bb.height*height
            x_e, y_e = x-0.15*w, y-0.2*h
            w_e, h_e = 1.2*w, 1.35*h

            # if the bounding box is out of the frame, ignore the frame
            if x_e > 0 and y_e > 0 and x_e+w_e<width and y_e+h_e<height:
                # get the enlarged face crop
                face_crop = image_rgb[round(y_e):round(y_e+h_e), round(x_e):round(x_e+w_e)]
                
                # original dimensions before transforming
                orig_height, orig_width = face_crop.shape[1], face_crop.shape[0]

                # transform the image
                input_img = transform_image(face_crop)[None,...].to(device)

                # get dimensions after transforming
                mod_height, mod_width = input_img.shape[2], input_img.shape[3]
                
                # calculate dimension transforms
                tf_x, tf_y = torch.tensor(mod_height/orig_height)[None,...].to(device), torch.tensor(mod_width/orig_width)[None,...].to(device) 

                # get the prediction
                prediction, vertices = network(input_img, (tf_x, tf_y))

            else: 
                print("Warning: All face is not visible, ignoring the frame")

        else:
            # cannot find a face in current frame, ignore the frame
            print("Warning: No faces detected, ignoring the frame")
        
        # if the frame is not ignored, use the current prediction, otherwise use the latest
        # append FLAME parameters
        scale = network.scale.detach().cpu().numpy()[None,...]
        tran = network.rot.detach().cpu().numpy()

        exp_stored.append(network.expression_params.detach().cpu().numpy()[0])
        shape_stored.append(network.shape_params.detach().cpu().numpy()[0])
        pose_stored.append(network.pose_params.detach().cpu().numpy()[0])
        cam_stored.append(np.concatenate((scale,tran), axis=1)[0])
        
        # backproject vertices to the original image to display
        if vertices is not None:
            list_of_dicts.append({"index": idx,
                                "face_min_x": round(x_e), "face_min_y": round(y_e), 
                                "face_max_x": round(x_e+w_e), "face_max_y":round(y_e+h_e)})

            cv2.imwrite(f"{frames_save_path}/{idx}.jpg", image)
        
            idx += 1
    cap.release()
    
    # save FLAME parameters 
    exp_stored = torch.tensor(exp_stored)          
    shape_stored = torch.tensor(shape_stored).mean(dim=0)
    pose_stored = filter_headpose(torch.tensor(pose_stored))
    cam_stored = torch.tensor(cam_stored)
    
    torch.save(exp_stored, f"{flame_params_save_path}/exp.pt")
    torch.save(shape_stored, f"{flame_params_save_path}/shape.pt")
    torch.save(pose_stored, f"{flame_params_save_path}/pose.pt")
    torch.save(cam_stored, f"{flame_params_save_path}/cam.pt")

    keys = list_of_dicts[0].keys()
    with open(f"{dataset_save_path}/frame_meta.csv", 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(list_of_dicts)
