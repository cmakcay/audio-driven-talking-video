import torch
import numpy as np
from PIL import Image, ImageDraw

def draw_landmarks(landmarks, face_edges, orig_size, save_path, idx):

    landmarks = torch.squeeze(landmarks).detach()
    assert landmarks.shape == (68,2)

    face_list = [
                    range(0, 17), # face
                    range(17, 22), # left eyebrow
                    range(22, 27), # right eyebrow
                    range(27, 31), range(31, 36), # nose
                    [36,37,38,39], [39,40,41,36], # left eye
                    [42,43,44,45], [45,46,47,42], # right eye
                    range(48, 55), [54,55,56,57,58,59,48], range(60,68), [67, 60] # mouth
                ]
    mask_list = list(range(0, 17))  + list(range(26, 16,-1)) +  [0]

    mouth_mask_list = list(range(2, 15))  + [29, 2]

    draw = ImageDraw.Draw(face_edges)

    mask_x = landmarks[mask_list,0]
    mask_y = landmarks[mask_list,1]
    mask_combined = []
    for i in range(mask_x.shape[0]):
        mask_combined.append((mask_x[i],mask_y[i]))

    draw.polygon(mask_combined, fill="black")    

    for p in face_list:
        p = list(p)
        for i, _ in enumerate(p):
            if i < len(p)-1:
                segment = [p[i], p[i+1]]
                x1 = landmarks[segment[0],0]
                x2 = landmarks[segment[1],0]
                
                y1 = landmarks[segment[0],1]
                y2 = landmarks[segment[1],1]
                draw.line([(x1,y1), (x2, y2)], width=5, fill="white")
    face_edges.save(f"{save_path}/edge_maps/{idx}.jpg")

    black_image = np.zeros((int(orig_size[1]),int(orig_size[0]), 3), dtype=np.uint8)
    black_image = Image.fromarray(black_image)
    draw = ImageDraw.Draw(black_image)

    mask_x = landmarks[mouth_mask_list,0]
    mask_y = landmarks[mouth_mask_list,1]
    mask_combined = []
    for i in range(mask_x.shape[0]):
        mask_combined.append((mask_x[i],mask_y[i]))

    draw.polygon(mask_combined, fill="white")    
    black_image.save(f"{save_path}/masks/{idx}.jpg")

