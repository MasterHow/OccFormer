import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES
from .loading_nusc_occ import custom_rotate_3d
import pdb

@PIPELINES.register_module()
class LoadH3OAnnotation():
    def __init__(self, bda_aug_conf, is_train=True, 
                 point_cloud_range=[-12.8, -12.8, -2.4, 12.8, 12.8, 0.8]):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.point_cloud_range = torch.tensor(point_cloud_range)
        self.transform_center = (self.point_cloud_range[:3] + self.point_cloud_range[3:]) / 2

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""

        rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
        scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
        flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
        flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        flip_dz = np.random.uniform() < self.bda_aug_conf['flip_dz_ratio']
        
        return rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz

    def forward_test(self, results):
        bda_rot = torch.eye(4).float()
        imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors = results['img_inputs']
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, gt_depths, sensor2sensors)
        
        return results

    def __call__(self, results):
        if results['gt_occ'] is None:
            return self.forward_test(results)
        
        if type(results['gt_occ']) is list:
            gt_occ = [torch.tensor(x) for x in results['gt_occ']]
        else:
            gt_occ = torch.tensor(results['gt_occ'])
        
        if self.is_train:
            rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz = self.sample_bda_augmentation()
            gt_occ, bda_rot = voxel_transform(gt_occ, rotate_bda, scale_bda, 
                        flip_dx, flip_dy, flip_dz, self.transform_center)
        else:
            bda_rot = torch.eye(4).float()
        
        imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors = results['img_inputs']
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, gt_depths, sensor2sensors)
        results['gt_occ'] = gt_occ.long()
        
        return results

def voxel_transform(voxel_labels, rotate_angle, scale_ratio, flip_dx, flip_dy, flip_dz, transform_center=None):
    # for H3O dataset, the transform origin is the center of the point cloud range
    
    if transform_center is None:
        transform_center = torch.zeros(3)
    
    # Create rotation matrix
    rotate_angle = rotate_angle * np.pi / 180
    cos_angle = np.cos(rotate_angle)
    sin_angle = np.sin(rotate_angle)
    
    # Rotation around Z axis (yaw)
    rotation_matrix = torch.tensor([
        [cos_angle, -sin_angle, 0, 0],
        [sin_angle, cos_angle, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)
    
    # Create scale matrix
    scale_matrix = torch.tensor([
        [scale_ratio, 0, 0, 0],
        [0, scale_ratio, 0, 0],
        [0, 0, scale_ratio, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)
    
    # Create flip matrices
    flip_x_matrix = torch.tensor([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32) if flip_dx else torch.eye(4)
    
    flip_y_matrix = torch.tensor([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32) if flip_dy else torch.eye(4)
    
    flip_z_matrix = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32) if flip_dz else torch.eye(4)
    
    # Combine all transformations
    transform_matrix = flip_z_matrix @ flip_y_matrix @ flip_x_matrix @ scale_matrix @ rotation_matrix
    
    # Apply transformation to voxel labels
    # Note: This is a simplified version. For proper 3D voxel transformation,
    # you would need to implement proper 3D interpolation
    # Extract rotation angle from transform matrix
    rotation_angle = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0]) * 180 / np.pi
    transformed_labels = custom_rotate_3d(voxel_labels, rotation_angle)
    
    return transformed_labels, transform_matrix
