import torch

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def rotation_matrix_to_euler_angle(R):
    # rotation around y axis (pitch)
    # [[cosb 0 sinb];[0 1 0];[-sinb 0 cosb]]
    r31, r32, r33 = R[:,2,0], R[:,2,1], R[:,2,2]
    return torch.atan2(-r31, torch.sqrt(r32**2 + r33**2))


def transformation_matrix(R, t):
    # Given R(3x3) and t(3x1), return  transformation matrix(4x4)
    transformation_matrix = torch.zeros((R.shape[0],4,4), dtype=R.dtype, device=R.device)
    transformation_matrix[:,:3,:3], transformation_matrix[:,:3,3:], transformation_matrix[:,3:,3:] = R, t, 1
    return transformation_matrix


def vertices_to_joints(J_regressor, vertices):
    # Get joint locations from vertices
    regressor_temp = J_regressor[...,None].repeat(1,1,vertices.shape[0]).permute(2,0,1)
    joints = torch.bmm(regressor_temp, vertices)
    return joints


def rodrigues_rotation(rotation_vectors):
    # Compute rotation matrices from rotation vectors using Rodrigues' formula

    theta = torch.linalg.vector_norm(rotation_vectors + 1e-8, dim=1)
    rotation_vectors_ = rotation_vectors / theta[...,None]
    wx, wy, wz = rotation_vectors_[:,0], rotation_vectors_[:,1], rotation_vectors_[:,2]
    skew_symmetric_w = torch.zeros((rotation_vectors.shape[0], 3, 3),device=rotation_vectors.device, dtype=torch.float32)
    skew_symmetric_w[:,0,1], skew_symmetric_w[:,1,0] = -wz, wz
    skew_symmetric_w[:,0,2], skew_symmetric_w[:,2,0] = wy, -wy
    skew_symmetric_w[:,1,2], skew_symmetric_w[:,2,1] = -wx, wx 
    
    I = torch.eye(3,device=rotation_vectors.device, dtype=torch.float32)[None,...].repeat(rotation_vectors.shape[0],1,1)
    sin_theta = torch.sin(theta)[...,None,None].repeat(1,3,3)
    cos_theta = torch.cos(theta)[...,None,None].repeat(1,3,3)
    R = I + sin_theta*skew_symmetric_w + (1-cos_theta) * torch.bmm(skew_symmetric_w, skew_symmetric_w)
    
    return R


def blend_shape_deform(betas, shapedirs):
    # Deform the template model by calculating vertex locations
    betas_temp = betas[...,None,None].repeat(1,1,shapedirs.shape[0], shapedirs.shape[1]) 
    shapes_temp = torch.permute(shapedirs[None,...], (0,3,1,2))
    blend_shape = (betas_temp*shapes_temp).sum(1)
    return blend_shape


def rigid_body_transform(rotation_matrices, joints, parents):

    # Expand the dimension
    joints = joints[...,None]

    # Get relative joint positions by subtracting the parent
    joints_relative = joints.clone()
    joints_relative[:, 1:] -= joints[:, parents[1:]]

    # Given the rotation and position, get the transformation matrix
    # B,J,4,4 to B*J,4,4 to B,J,4,4 
    transformation_matrices = transformation_matrix(rotation_matrices.view(-1, 3, 3), 
                            joints_relative.view(-1, 3, 1)).view(-1, joints.shape[1], 4, 4)


    list_of_transforms = [transformation_matrices[:, 0]] # start with first joint (global rotation) 
    # For each joint, get the rest pose and add to the list, i.e. chain
    for joint_ in range(1, parents.shape[0]):
        rest_pose = torch.matmul(list_of_transforms[parents[joint_]], transformation_matrices[:, joint_])
        list_of_transforms.append(rest_pose)

    # Stack the list of transforms
    transforms = torch.stack(list_of_transforms, dim=1)

    # Return translation vector (t) from the transformation matrices, which is position of the joints
    joint_positions = transforms[:, :, :3, 3]

    # pad joint positions with a zero for applying transformations
    joints_padded = torch.zeros((joints.shape[0],joints.shape[1],4,1), dtype=joints.dtype, device=joints.device)
    joints_padded[:,:,:3,:] = joints
    
    # 4x1 to 4x4
    transformed_joints = torch.matmul(transforms, joints_padded)
    transformed_joints_padded = torch.zeros((transformed_joints.shape[0],transformed_joints.shape[1],4,4), dtype=transformed_joints.dtype, device=transformed_joints.device)
    transformed_joints_padded[:,:,:,3:] = transformed_joints

    transformations_relative = transforms - transformed_joints_padded 
    return joint_positions, transformations_relative


def vertices_to_landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):

    batch_size, num_vertices = vertices.shape[0], vertices.shape[1]

    landmarks_faces = faces[torch.flatten(lmk_faces_idx)].view(batch_size, -1, 3)
    landmarks_faces += torch.arange(batch_size, dtype=torch.long)[...,None,None].to(vertices.device) * num_vertices

    lmk_vertices = vertices.view(-1, 3)[landmarks_faces].view(batch_size, -1, 3, 3)
    landmarks = (lmk_vertices * lmk_bary_coords[...,None].repeat(1,1,1,3)).sum(2)
    return landmarks


def linear_blend_skinning(betas, pose, template_mesh, shapedirs, posedirs, J_regressor, parents, lbs_weights, dtype=torch.float32):

    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    # Apply changes due to shape parameters
    shaped_mesh = template_mesh + blend_shape_deform(betas, shapedirs)

    # Get joints from vertices
    joints = vertices_to_joints(J_regressor, shaped_mesh)
    
    I = torch.eye(3, dtype=dtype, device=device)
    rotation_matrices = rodrigues_rotation(pose.view(-1, 3)).view([batch_size, -1, 3, 3])
    pose_feature = (rotation_matrices[:, 1:, :, :] - I).view([batch_size, -1])
    pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)

    shaped_pose = pose_offsets + shaped_mesh
    
    joints_transformed, A = rigid_body_transform(rotation_matrices, joints, parents)

    # Skinning
    W = lbs_weights[None,...].repeat(batch_size, 1, 1)
    T = torch.matmul(W, A.view(batch_size, J_regressor.shape[0], 16)).view(batch_size, -1, 4, 4)

    homo_coordinates = torch.ones([batch_size, shaped_pose.shape[1], 1], dtype=dtype, device=device)
    posed_mesh = torch.cat([shaped_pose, homo_coordinates], dim=2)
    homogen_mesh = torch.matmul(T, torch.unsqueeze(posed_mesh, dim=-1))

    vertices = homogen_mesh[:, :, :3, 0]

    return vertices, joints_transformed
