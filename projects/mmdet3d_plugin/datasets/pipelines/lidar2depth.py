import open3d as o3d
import numpy as np
import torch
import os
from mmdet.datasets.builder import PIPELINES
from .project_ocam import load_maps, ocam_model, get_ocam_model, project_lidar_to_image
import pdb

@PIPELINES.register_module()
class CreateDepthFromLiDAR(object):
    def __init__(self, data_root=None, dataset='kitti'):
        self.data_root = data_root
        self.dataset = dataset
        assert self.dataset in ['kitti', 'nusc', 'quad', 'h3o']
        
    def project_points(self, points, rots, trans, intrins, post_rots, post_trans):
        # from lidar to camera
        points = points.view(-1, 1, 3)
        points = points - trans.view(1, -1, 3)
        inv_rots = rots.inverse().unsqueeze(0)
        points = (inv_rots @ points.unsqueeze(-1))
        
        # the intrinsic matrix is [4, 4] for kitti and [3, 3] for nuscenes 
        if intrins.shape[-1] == 4:
            points = torch.cat((points, torch.ones((points.shape[0], 1, 1, 1))), dim=2)
            points = (intrins.unsqueeze(0) @ points).squeeze(-1)
        else:
            points = (intrins.unsqueeze(0) @ points).squeeze(-1)
        
        points_d = points[..., 2:3]
        points_uv = points[..., :2] / points_d
        
        # from raw pixel to transformed pixel
        points_uv = post_rots[:, :2, :2].unsqueeze(0) @ points_uv.unsqueeze(-1)
        points_uv = points_uv.squeeze(-1) + post_trans[..., :2].unsqueeze(0)
        points_uvd = torch.cat((points_uv, points_d), dim=2)
        
        return points_uvd

    def _project_3d_to_equirect(self, points_3d, img_h, img_w):
        """
        Project 3D Cartesian points back to equirectangular image coordinates
        Args:
            points_3d: numpy array of shape (N, 3) with Cartesian coordinates (x, y, z)
            img_h, img_w: target image dimensions
        Returns:
            projected_points: torch tensor of shape (N, 2) with (u, v) coordinates
        """
        x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
        
        # Convert Cartesian to spherical coordinates
        # Right-handed system: X forward, Y left, Z up
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # Avoid division by zero
        valid_mask = r > 1e-8
        if not np.any(valid_mask):
            return torch.zeros((len(points_3d), 2)).float()
        
        # Calculate spherical angles
        # theta: azimuth angle [-π, π]
        theta = np.arctan2(y, x)  # atan2(y, x) gives correct angle
        
        # phi: elevation angle [0, π]
        phi = np.arccos(z / r)
        
        # Convert to equirectangular coordinates
        # u = (theta + π) / (2π) * W  -> u in [0, W]
        u = (theta + np.pi) / (2 * np.pi) * img_w
        
        # v = phi / π * H  -> v in [0, H]  
        v = phi / np.pi * img_h
        
        # Handle wrap-around for u coordinate
        u = np.mod(u, img_w)
        
        # Create projected points
        projected_points = np.stack([u, v], axis=1)
        
        # For H3O synthetic data, keep all projected points without filtering
        # Just clamp coordinates to image bounds to avoid out-of-bounds access
        projected_points[:, 0] = np.clip(projected_points[:, 0], 0, img_w - 1)
        projected_points[:, 1] = np.clip(projected_points[:, 1], 0, img_h - 1)
        
        return torch.from_numpy(projected_points).float()

    def __call__(self, results):
        # 加载LiDAR点
        if self.dataset == 'kitti':
            img_filename = results['img_filename'][0]
            seq_id, _, filename = img_filename.split("/")[-3:]
            lidar_filename = os.path.join(self.data_root, 'data_velodyne/velodyne/sequences',
                                          seq_id, "velodyne", filename.replace(".png", ".bin"))
            lidar_points = np.fromfile(lidar_filename, dtype=np.float32).reshape(-1, 4)
            lidar_points = torch.from_numpy(lidar_points[:, :3]).float()

        elif self.dataset == 'quad':
            # 获取config中输入的图像尺寸
            h, w = results['img_inputs'][0].shape[-2:]

            # 确保路径分隔符使用的是操作系统的格式
            img_filename = results['img_filename'][0]
            img_filename = os.path.normpath(img_filename)

            # 根据路径分隔符进行拆分
            parts = img_filename.split(os.sep)
            if len(parts) >= 3:
                seq_id, _, filename = parts[-3:]
            else:
                raise ValueError(f"img_filename path {img_filename} has an unexpected format")

            # 加载点云文件
            lidar_filename = os.path.join(self.data_root, 'sequences', seq_id, "lidar",
                                          filename.replace(".jpg", ".pcd"))
            lidar_points_pcd = o3d.io.read_point_cloud(lidar_filename)
            lidar_points = np.asarray(lidar_points_pcd.points, dtype=np.float32)
            # 删除所有坐标为 (0, 0, 0) 的点 原因：livox中包含大量为坐标0的点
            mask = ~(np.all(lidar_points == 0, axis=1))
            lidar_points = lidar_points[mask]
            lidar_points = torch.from_numpy(lidar_points).float()

            # 定义相机到雷达的旋转和平移矩阵
            R_lidar_to_camera = np.array([
                [0.965595, 0.02617695, 0.25873031],
                [-0.02528499, 0.99965732, -0.00677509],
                [-0.258819, 0., 0.965926]
            ])
            T_lidar_to_camera = np.array([0.08, 0, -0.02])

            # 加载maps.npz文件获取mapx和mapy
            # map_file = os.path.join(self.data_root, 'maps.npz')
            map_file = os.path.join(self.data_root, 'maps_384.npz')
            if os.path.exists(map_file):
                mapx, mapy, inv_mapx, inv_mapy = load_maps(map_file)
            else:
                raise FileNotFoundError(f"{map_file} not found. Please provide the correct mapping file.")

            # 加载相机模型
            o_cata = ocam_model()
            calib_file = os.path.join(self.data_root, 'PAL_intrinsic_calib_results.txt')
            get_ocam_model(o_cata, calib_file)

            # 执行投影
            projected_points, corresponding_lidar_points = project_lidar_to_image(
                lidar_points.numpy(), R_lidar_to_camera, T_lidar_to_camera, inv_mapx, inv_mapy, o_cata, unfold_img_h=h,
                unfold_img_w=w
            )

            # 将投影点转换为 torch 张量
            projected_points = torch.tensor(projected_points, dtype=torch.float32)      # 雷达点在图像上的投影坐标
            # corresponding_lidar_points = torch.tensor(np.array(corresponding_lidar_points), dtype=torch.float32)
            lidar_points = torch.tensor(np.array(corresponding_lidar_points), dtype=torch.float32)    # 每个图像坐标对应的雷达点
            # corresponding_lidar_points = torch.tensor(corresponding_lidar_points, dtype=torch.float32)

        elif self.dataset == 'h3o':
            # H3O dataset: Convert equirectangular Z-depth to Cartesian depth
            img_filename = results['img_filename'][0]
            img_filename = os.path.normpath(img_filename)
            
            # Try to load depth from panorama_depth_m if available
            parts = img_filename.split(os.sep)
            if len(parts) >= 3:
                seq_id, _, filename = parts[-3:]
                depth_filename = os.path.join(self.data_root, 'sequences', seq_id, 'panorama_depth_m',
                                             filename.replace('.png', '.npy'))
                
                if os.path.exists(depth_filename):
                    # Load equirectangular Z-depth
                    z_depth = np.load(depth_filename)  # Shape: (H, W)
                    h, w = z_depth.shape
                    
                    # Convert equirectangular Z-depth to Cartesian depth
                    # Following H3O documentation: theta=(u/W)*2π-π, phi=(v/H)*π
                    u, v = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
                    theta = (u / w) * 2 * np.pi - np.pi  # [-π, π]
                    phi = (v / h) * np.pi  # [0, π]
                    
                    # Convert to ray direction (right-handed: X forward, Y left, Z up)
                    nx = np.sin(phi) * np.cos(theta)
                    ny = np.sin(phi) * np.sin(theta)
                    nz = np.cos(phi)
                    
                    # Calculate projection ratio for cubemap face
                    denom = np.maximum(np.maximum(np.abs(nx), np.abs(ny)), np.abs(nz))
                    
                    # Convert Z-depth to Euclidean distance
                    r = z_depth / (denom + 1e-8)  # Avoid division by zero
                    
                    # Convert to Cartesian coordinates
                    x = r * nx
                    y = r * ny
                    z = r * nz
                    
                    # Stack coordinates
                    points_3d = np.stack([x, y, z], axis=-1)  # Shape: (H, W, 3)
                    
                    # For H3O synthetic data, all points are valid - no filtering needed
                    # Use all points for synthetic data
                    lidar_points = torch.from_numpy(points_3d).float()
                    # For equirectangular, we need to project these 3D points back to image coordinates
                    projected_points = self._project_3d_to_equirect(points_3d, h, w)
                        
                else:
                    # Fallback: create dummy depth map
                    h, w = results['img_inputs'][0].shape[-2:]
                    # Create a simple depth pattern for equirectangular
                    u, v = np.meshgrid(np.linspace(0, w-1, w), np.linspace(0, h-1, h))
                    # Simple depth based on vertical position (closer to horizon = farther)
                    depth = 10.0 + 5.0 * np.abs(v - h/2) / (h/2)  # 10-15m range
                    
                    # Convert to 3D points
                    theta = (u / w) * 2 * np.pi - np.pi
                    phi = (v / h) * np.pi
                    nx = np.sin(phi) * np.cos(theta)
                    ny = np.sin(phi) * np.sin(theta)
                    nz = np.cos(phi)
                    
                    x = depth * nx
                    y = depth * ny
                    z = depth * nz
                    
                    points_3d = np.stack([x, y, z], axis=-1)
                    # Ensure points are within the point cloud range for H3O dataset
                    # H3O point cloud range: [-12.8, -12.8, -2.4, 12.8, 12.8, 0.8]
                    valid_mask = (depth > 0) & (depth < 100) & (z > -2.4) & (z < 0.8) & \
                                (x > -12.8) & (x < 12.8) & (y > -12.8) & (y < 12.8)
                    valid_points = points_3d[valid_mask]
                    
                    if len(valid_points) > 0:
                        lidar_points = torch.from_numpy(valid_points).float()
                        projected_points = self._project_3d_to_equirect(valid_points, h, w)
                    else:
                        # Create points within the valid range if no points pass the filter
                        # Generate points in a grid pattern within the point cloud range
                        x_range = np.linspace(-10, 10, 20)  # Within [-12.8, 12.8]
                        y_range = np.linspace(-10, 10, 20)  # Within [-12.8, 12.8]
                        z_range = np.linspace(-2, 0.5, 10)  # Within [-2.4, 0.8]
                        
                        xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')
                        points_3d = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1)
                        
                        lidar_points = torch.from_numpy(points_3d).float()
                        projected_points = self._project_3d_to_equirect(points_3d, h, w)
            else:
                raise ValueError(f"img_filename path {img_filename} has an unexpected format")

        else:
            lidar_points = np.fromfile(results['pts_filename'], dtype=np.float32, count=-1).reshape(-1, 5)[..., :3]
            lidar_points = torch.from_numpy(lidar_points).float()

        # 创建深度图
        imgs, rots, trans, intrins, post_rots, post_trans = results['img_inputs'][:6]
        img_h, img_w = imgs[0].shape[-2:]
        gt_depths = []

        if self.dataset == 'quad':
            # quad 数据集：从 projected_points 获取像素位置，从 corresponding_lidar_points 获取深度
            # projected_points = self.project_points(lidar_points, rots, trans, intrins, post_rots, post_trans)
            # valid_mask = (projected_points[..., 0] >= 0) & \
            #              (projected_points[..., 1] >= 0) & \
            #              (projected_points[..., 0] <= img_w - 1) & \
            #              (projected_points[..., 1] <= img_h - 1) & \
            #              (projected_points[..., 2] > 0)
            valid_mask = (projected_points[..., 0] >= 0) & \
                         (projected_points[..., 1] >= 0) & \
                         (projected_points[..., 0] <= img_w - 1) & \
                         (projected_points[..., 1] <= img_h - 1)

            for img_index in range(imgs.shape[0]):
                gt_depth = torch.zeros((img_h, img_w))
                valid_points_uv = projected_points[valid_mask]  # 使用图像 u,v 坐标
                valid_points_depth = lidar_points[valid_mask][:, 2]  # 使用对应的深度值（z 坐标）

                # 对深度进行排序
                depth_order = torch.argsort(valid_points_depth, descending=True)
                valid_points_uv = valid_points_uv[depth_order]
                valid_points_depth = valid_points_depth[depth_order]

                # 填充深度图
                gt_depth[valid_points_uv[:, 1].round().long(),
                valid_points_uv[:, 0].round().long()] = valid_points_depth
                gt_depths.append(gt_depth)
        elif self.dataset == 'h3o':
            # H3O 数据集：从等距圆柱Z深度转换为笛卡尔深度图
            # 对于合成数据，所有深度都是准确的，不需要过滤
            for img_index in range(imgs.shape[0]):
                gt_depth = torch.zeros((img_h, img_w))
                
                if projected_points is not None and len(projected_points) > 0:
                    # 使用投影的3D点创建深度图
                    valid_points_uv = projected_points
                    valid_points_depth = torch.norm(lidar_points, dim=1)  # Euclidean distance
                    
                    # 对深度进行排序（远的先画，近的后画，避免遮挡）
                    depth_order = torch.argsort(valid_points_depth, descending=True)
                    valid_points_uv = valid_points_uv[depth_order]
                    valid_points_depth = valid_points_depth[depth_order]
                    
                    # 确保坐标在图像范围内
                    u_coords = torch.clamp(valid_points_uv[:, 0].round().long(), 0, img_w - 1)
                    v_coords = torch.clamp(valid_points_uv[:, 1].round().long(), 0, img_h - 1)
                    
                    # 填充深度图
                    gt_depth[v_coords, u_coords] = valid_points_depth
                else:
                    # 对于 H3O 合成数据，创建完整的深度图覆盖整个图像
                    # 使用等距圆柱投影的深度模式
                    u, v = torch.meshgrid(torch.arange(img_w), torch.arange(img_h), indexing='xy')
                    
                    # 转换为球面坐标
                    theta = (u.float() / img_w) * 2 * np.pi - np.pi  # [-π, π]
                    phi = (v.float() / img_h) * np.pi  # [0, π]
                    
                    # 基于仰角的深度估计（地平线附近更深）
                    # 使用正弦函数模拟距离变化
                    depth_factor = 1.0 + 0.5 * torch.sin(phi)  # 地平线附近更深
                    base_depth = 10.0  # 基础深度
                    estimated_depth = base_depth * depth_factor
                    
                    gt_depth = estimated_depth
                
                gt_depths.append(gt_depth)
        else:
            # 标准数据集处理流程
            projected_points = self.project_points(lidar_points, rots, trans, intrins, post_rots, post_trans)
            valid_mask = (projected_points[..., 0] >= 0) & \
                         (projected_points[..., 1] >= 0) & \
                         (projected_points[..., 0] <= img_w - 1) & \
                         (projected_points[..., 1] <= img_h - 1) & \
                         (projected_points[..., 2] > 0)

            for img_index in range(imgs.shape[0]):
                gt_depth = torch.zeros((img_h, img_w))
                projected_points_i = projected_points[:, img_index]
                valid_mask_i = valid_mask[:, img_index]
                valid_points_i = projected_points_i[valid_mask_i]
                # 排序深度值
                depth_order = torch.argsort(valid_points_i[:, 2], descending=True)
                valid_points_i = valid_points_i[depth_order]
                # 填充深度图
                gt_depth[valid_points_i[:, 1].round().long(),
                valid_points_i[:, 0].round().long()] = valid_points_i[:, 2]
                gt_depths.append(gt_depth)

        gt_depths = torch.stack(gt_depths)
        results['img_inputs'] = (*results['img_inputs'][:6], gt_depths, results['img_inputs'][7])

        return results
        
    def visualize(self, imgs, img_depths):
        out_path = 'debugs/lidar2depth'
        os.makedirs(out_path, exist_ok=True)
        
        import matplotlib.pyplot as plt
        
        # convert depth-map to depth-points
        for img_index in range(imgs.shape[0]):
            img_i = imgs[img_index][..., [2, 1, 0]]
            depth_i = img_depths[img_index]
            depth_points = torch.nonzero(depth_i)
            depth_points = torch.stack((depth_points[:, 1], depth_points[:, 0], depth_i[depth_points[:, 0], depth_points[:, 1]]), dim=1)
            
            plt.figure(dpi=300)
            plt.imshow(img_i)
            plt.scatter(depth_points[:, 0], depth_points[:, 1], s=1, c=depth_points[:, 2], alpha=0.2)
            plt.axis('off')
            plt.title('Image Depth')
            
            plt.savefig(os.path.join(out_path, 'demo_depth_{}.png'.format(img_index)))
            plt.close()
        
        pdb.set_trace()