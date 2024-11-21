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
        assert self.dataset in ['kitti', 'nusc', 'quad']
        
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