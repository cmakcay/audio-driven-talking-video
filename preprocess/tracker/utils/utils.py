import torch
from scipy.ndimage import gaussian_filter1d
import numpy as np

def scale_and_translate(landmark, s, t):
    """
    Scale and translate the landmarks

    Parameters
    ----------
    landmark
        Tensor with size [N, 68, 2].
    s
        Scaling parameter [N].
    t
        Translation vector [N, 2].
    """
    # add singleton dimensions
    tran = torch.transpose(t[None,:,:], 0, 1).repeat(1, 68, 1)
    scale = s[:, None, None].repeat(1, 68, 2)

    # scaling and translation
    return torch.mul(scale, landmark) + tran


def orthographic_projection(landmarks, s, t):
    """
    Perform orthographic projection

    Parameters
    ----------
    landmarks
        Tensor with size [N, 68, 3].
    s
        Scaling parameter [N].
    t
        Translation vector [N, 2].
    """

    # first remove 3d dimension (depth)
    # orthographic projection by simply ignoring z component
    proj_lmk = landmarks[:,:,[0, 1]]

    # Flip y to transform to image axes (xy -> uv)
    y = proj_lmk[:,:,1]
    max_y, min_y = torch.max(y,1)[0], torch.min(y,1)[0]
    mid_y = ((max_y+min_y)/2)[:,None].repeat(1,68)
    y = 2*mid_y-y
    proj_lmk[:,:,1] = y

    scaled_and_translated_lmk = scale_and_translate(proj_lmk, s, t)
    return scaled_and_translated_lmk

def transform_landmarks(landmarks, transform_params):
    """
    Transform the landmarks of FLAME according to the ratio changes in x and y dimensions of the image

    Parameters
    ----------
    landmarks
        Tensor with size [N, 68, 3].
    transform_params
        Transforms in x and y dimensions [N, 2].
    """
    
    tf_landmarks = landmarks.clone()
    
    # move the landmarks to 0 in both dimensions by subtracting the minimum
    min_x = torch.min(tf_landmarks[:,:,0], dim=1)[0]
    min_y = torch.min(tf_landmarks[:,:,1], dim=1)[0]
    tf_landmarks[:,:,0] = landmarks[:,:,0] - min_x[:,None].repeat(1,68)
    tf_landmarks[:,:,1] = landmarks[:,:,1] - min_y[:,None].repeat(1,68)
    
    # x axis is multiplied by a constant
    tf_landmarks[:,:,0] = tf_landmarks[:,:,0] *  1000.0
    
    # y axis is scaled to keep the modified ratio
    # tf_landmarks[:,:,1] = tf_landmarks[:,:,1] * (transform_params[1][:, None].repeat(1, 68) * (1000.0/transform_params[0][:, None].repeat(1, 68))).cuda()
    tf_landmarks[:,:,1] = tf_landmarks[:,:,1] * (transform_params[1][:, None].repeat(1, 68) * (1000.0/transform_params[0][:, None].repeat(1, 68)))
    
    return tf_landmarks


def squared_norm(vector):
    """
    Squared norm of the vector

    Parameters
    ----------
    vector
        Shape or expression parameters.
    """
    return torch.mean(torch.sum(torch.square(vector), 1))


def backproject_vertices(landmarks, vertices, tf_params, scale):
    """
    Backproject vertices to the original image 

    Parameters
    ----------
    landmarks
        Landmarks [N, 68, 3].
    vertices
        Vertices of FLAME model [N, 5023, 3].
    tf_params
        Transform ratios [N, 2].
    scale
        Scaling parameter [N].
    """
    
    # Move landmarks to 0 by subtracting the minimum
    orig_landmarks = landmarks.clone()
    min_x_ = torch.min(orig_landmarks[:,:,0], dim=1)[0]
    min_y_ = torch.min(orig_landmarks[:,:,1], dim=1)[0]

    orig_landmarks[:,:,0] = orig_landmarks[:,:,0] - min_x_[:,None].repeat(1,68)
    orig_landmarks[:,:,1] = orig_landmarks[:,:,1] - min_y_[:,None].repeat(1,68)

    # Scale landmarks by transform ratios and move back to the original place
    orig_landmarks[:,:,0] = orig_landmarks[:,:,0] * (1./tf_params[0][:, None].repeat(1, 68))
    orig_landmarks[:,:,1] = orig_landmarks[:,:,1] * (1./tf_params[1][:, None].repeat(1, 68))
    orig_landmarks[:,:, 0] = orig_landmarks[:,:,0] + min_x_[:,None].repeat(1,68)*(1./tf_params[0][:, None].repeat(1, 68))
    orig_landmarks[:, :,1] = orig_landmarks[:,:,1] + min_y_[:,None].repeat(1,68)*(1./tf_params[1][:, None].repeat(1, 68))
    
    # Now we have landmarks backprojected
    landmarks=orig_landmarks

    # Move vertices to 0 in 3 dimensions by subtracting the minimum
    scaled_vertices = vertices.clone() #Nx5023x3
    min_x = torch.min(scaled_vertices[:,:,0], dim=1)[0]
    min_y = torch.min(scaled_vertices[:,:,1], dim=1)[0]
    min_z = torch.min(scaled_vertices[:,:,2], dim=1)[0]

    scaled_vertices[:,:,0] = vertices[:,:,0] - min_x[:, None].repeat(1, 5023)
    scaled_vertices[:,:,1] = vertices[:,:,1] - min_y[:, None].repeat(1, 5023)
    scaled_vertices[:,:,2] = vertices[:,:,2] - min_z[:, None].repeat(1, 5023)
    
    # Vertices were not rescaled before, thus scaling by the same value
    scaled_vertices[:,:,0] = scaled_vertices[:,:,0] * 1000.0 * (scale)[:, None].repeat(1, 5023) * (1./tf_params[0][:, None].repeat(1, 5023))
    scaled_vertices[:,:,1] = scaled_vertices[:,:,1] * 1000.0 * (scale)[:, None].repeat(1, 5023) * (1./tf_params[0][:, None].repeat(1, 5023))
    scaled_vertices[:,:,2] = scaled_vertices[:,:,2] * 1000.0 * (scale)[:, None].repeat(1, 5023) * (1./tf_params[0][:, None].repeat(1, 5023))

    # Flip vertices to be comparable with landmarks
    scaled_vertices = flip_vertices(scaled_vertices) 

    # Use the position of a landmark as a reference to move vertices to the original location
    # Face ind 7493 corresponds to 68th landmark, vertices are [2938, 2937, 2928]
    lmk_ref_x, lmk_ref_y = landmarks[:, -1, 0], landmarks[:, -1, 1]
    
    # Average of 3 attached vertices 
    vert_ref = (scaled_vertices[:,2937,:] + scaled_vertices[:,2938,:]+ scaled_vertices[:,2928,:])/3.
    vert_ref_x, vert_ref_y = vert_ref[:,0], vert_ref[:,1]

    # Move vertices to the actual location
    scaled_x_tran = lmk_ref_x - vert_ref_x
    scaled_y_tran = lmk_ref_y - vert_ref_y
    translated_vertices = scaled_vertices.clone()
    translated_vertices[:,:,0] += scaled_x_tran[:, None].repeat(1, 5023)
    translated_vertices[:,:,1] += scaled_y_tran[:, None].repeat(1, 5023)
    translated_vertices[:,:,2] = translated_vertices[:,:,2] - torch.max(translated_vertices[:,:,2], dim=1)[0][:, None].repeat(1, 5023)

    # Flip the vertices again to be upright
    flipped_translated_vertices = flip_vertices(translated_vertices)

    return translated_vertices, flipped_translated_vertices, landmarks

def flip_vertices(vertices):
    """
    Flip vertices to switch between image coordinate system (u, v) and (x, y) 

    Parameters
    ----------
    vertices
        Vertices of FLAME model [N, 5023, 3].
    """
    flipped_vertices = vertices.clone()
    y_flip = flipped_vertices[:,:,1]
    max_y, min_y = torch.max(y_flip, 1)[0], torch.min(y_flip, 1)[0]
    mid_y = ((max_y+min_y)/2)[:, None].repeat(1, vertices.size(1))
    y_flip = 2*mid_y-y_flip
    flipped_vertices[:,:,1]=y_flip
    return flipped_vertices

def filter_headpose(headpose, smooth_sigmas=[1,1], method='gaussian'):
    rot_sigma, trans_sigma = smooth_sigmas
    
    # not filtering jaw
    rot = gaussian_filter1d(headpose.reshape(-1, 6)[:,:3], rot_sigma, axis=0).reshape(-1, 3)
    trans = headpose.reshape(-1, 6)[:, 3:].reshape(-1, 3)

    headpose_smooth = np.concatenate([rot, trans], axis=1)
    return torch.tensor(headpose_smooth)