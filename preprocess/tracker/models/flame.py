'''
    FLAME model borrowed from https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
    Given shape, pose and expression parameters, this FLAME layer returns vertices and landmarks.
'''

import torch
import torch.nn as nn
import numpy as np
import pickle
from ..utils.flame_utils import linear_blend_skinning, rodrigues_rotation, rotation_matrix_to_euler_angle, vertices_to_landmarks, Struct


class FLAME(nn.Module):

    def __init__(self, args):
        super(FLAME, self).__init__()

        # Load FLAME model
        with open(args.flame_model_path, 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)

        # dtype used in FLAME layer is float32
        self.dtype = torch.float32

        # Faces of the FLAME model
        self.register_buffer('faces_tensor', torch.tensor(np.array(flame_model.f, dtype=np.int64), dtype=torch.long))
        
        # Vertices of the FLAME model
        self.register_buffer('v_template', torch.tensor(np.array(flame_model.v_template, dtype=np.float32), dtype=self.dtype))
        
        # Shape and expression components
        shapedirs = torch.tensor(np.array(flame_model.shapedirs, dtype=np.float32), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:, :, :args.shape_params], shapedirs[:, :, 300:300 + args.expression_params]], 2)
        self.register_buffer('shapedirs', shapedirs)
        
        # Pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.tensor(np.array(posedirs, dtype=np.float32), dtype=self.dtype))
        
        # Joint regresor
        self.register_buffer('J_regressor', torch.tensor(np.array(flame_model.J_regressor.todense(), dtype=np.float32), dtype=self.dtype))
        parents = torch.tensor(np.array(flame_model.kintree_table[0], dtype=np.float32), dtype=torch.long)
        parents[0] = -1
        self.register_buffer('parents', parents)

        # LBS weights
        self.register_buffer('lbs_weights', torch.tensor(np.array(flame_model.weights, dtype=np.float32), dtype=self.dtype))

        # Eye pose and neck pose are constant, set default to zero
        default_eyeball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyeball_pose,
                                                         requires_grad=False))
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose,
                                                          requires_grad=False))

        # Static and Dynamic Landmark embeddings for FLAME
        lmk_embeddings = np.load(args.flame_lmk_embedding_path, allow_pickle=True, encoding='latin1')
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer('lmk_faces_idx', torch.from_numpy(lmk_embeddings['static_lmk_faces_idx']).long())
        self.register_buffer('lmk_bary_coords',
                             torch.from_numpy(lmk_embeddings['static_lmk_bary_coords']).to(self.dtype))
        self.register_buffer('dynamic_lmk_faces_idx', lmk_embeddings['dynamic_lmk_faces_idx'].long())
        self.register_buffer('dynamic_lmk_bary_coords', lmk_embeddings['dynamic_lmk_bary_coords'].to(self.dtype))
        self.register_buffer('full_lmk_faces_idx', torch.from_numpy(lmk_embeddings['full_lmk_faces_idx']).long())
        self.register_buffer('full_lmk_bary_coords',
                             torch.from_numpy(lmk_embeddings['full_lmk_bary_coords']).to(self.dtype))

        neck_kin_chain = [];
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))


    def _find_dynamic_lmk_idx_and_bcoords(self, pose, dynamic_lmk_faces_idx,
                                          dynamic_lmk_b_coords,
                                          neck_kin_chain, dtype=torch.float32):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
        """

        batch_size = pose.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                     neck_kin_chain)
        rot_mats = rodrigues_rotation(
            aa_pose.view(-1, 3)).view(batch_size, -1, 3, 3)

        rel_rot_mat = torch.eye(3, device=pose.device,
                                dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(rotation_matrix_to_euler_angle(rel_rot_mat) * 180.0 / np.pi,
                        max=39)).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals +
                       (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                               0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                              0, y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_b_coords



    def seletec_3d68(self, vertices):
        landmarks3d = vertices_to_landmarks(vertices, self.faces_tensor,
                                         self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
                                         self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1))
        return landmarks3d

    def forward(self, shape_params=None, expression_params=None, pose_params=None, eye_pose_params=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters (6)
            return:d
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        batch_size = shape_params.shape[0]

        if pose_params is None:
            pose_params = self.eye_pose.expand(batch_size, -1)
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1)
        
        betas = torch.cat([shape_params, expression_params], dim=1)
        full_pose = torch.cat(
            [pose_params[:, :3], self.neck_pose.expand(batch_size, -1), pose_params[:, 3:], eye_pose_params], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        vertices, _ = linear_blend_skinning(betas, full_pose, template_vertices,
                          self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights, dtype=self.dtype)

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)

        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
            full_pose, self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.neck_kin_chain, dtype=self.dtype)
        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks2d = vertices_to_landmarks(vertices, self.faces_tensor,
                                         lmk_faces_idx,
                                         lmk_bary_coords)
        bz = vertices.shape[0]
        landmarks3d = vertices_to_landmarks(vertices, self.faces_tensor,
                                         self.full_lmk_faces_idx.repeat(bz, 1),
                                         self.full_lmk_bary_coords.repeat(bz, 1, 1))
        return vertices, landmarks2d, landmarks3d
