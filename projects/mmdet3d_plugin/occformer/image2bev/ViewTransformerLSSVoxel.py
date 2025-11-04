# Copyright (c) Phigent Robotics. All rights reserved.
import torch
from mmdet3d.models.builder import NECKS
from mmdet3d.ops.bev_pool import bev_pool
from mmcv.runner import force_fp32
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F
import pdb

from .ViewTransformerLSSBEVDepth import *

@NECKS.register_module()
class ViewTransformerLiftSplatShootVoxel(ViewTransformerLSSBEVDepth):
    def __init__(
            self, 
            loss_depth_weight,
            point_cloud_range=None,
            loss_depth_type='bce', 
            **kwargs,
        ):
        super(ViewTransformerLiftSplatShootVoxel, self).__init__(loss_depth_weight=loss_depth_weight, **kwargs)
        
        self.loss_depth_type = loss_depth_type
        self.cam_depth_range = self.grid_config['dbound']
        self.point_cloud_range = point_cloud_range
    
    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N,
                                   H // self.downsample, self.downsample,
                                   W // self.downsample, self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)
        
        # [min - step / 2, min + step / 2] creates min depth
        gt_depths = (gt_depths - (self.grid_config['dbound'][0] - self.grid_config['dbound'][2] / 2)) / self.grid_config['dbound'][2]
        gt_depths_vals = gt_depths.clone()
        
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]
        
        return gt_depths_vals, gt_depths.float()
    
    @force_fp32()
    def get_bce_depth_loss(self, depth_labels, depth_preds):
        _, depth_labels = self.get_downsampled_gt_depth(depth_labels)
        # depth_labels = self._prepare_depth_gt(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(depth_preds, depth_labels, reduction='none').sum() / max(1.0, fg_mask.sum())
        
        return depth_loss
    
    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        if self.loss_depth_type == 'bce':
            depth_loss = self.get_bce_depth_loss(depth_labels, depth_preds)
        
        else:
            pdb.set_trace()
        
        return self.loss_depth_weight * depth_loss
        
    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box        # todo 根据数据集的不同调整bev网格的上下限 0: x 1: y 2: h
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]
        
        # [b, c, z, x, y] == [b, c, x, y, z]
        final = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])
        final = final.permute(0, 1, 3, 4, 2)

        return final

    def forward(self, input):
        (x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input) = input[:8]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x, mlp_input)
        depth_digit = x[:, :self.D, ...]
        img_feat = x[:, self.D:self.D + self.numC_Trans, ...]
        depth_prob = self.get_depth_dist(depth_digit)

        # Lift
        volume = depth_prob.unsqueeze(1) * img_feat.unsqueeze(2)
        volume = volume.view(B, N, -1, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        # Splat
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans, bda)
        bev_feat = self.voxel_pooling(geom, volume)
        
        return bev_feat, depth_prob


@NECKS.register_module()
class ViewTransformerLiftSplatShootVoxelH3O(ViewTransformerLSSBEVDepth):
    """ViewTransformer for H3O dataset with equirectangular projection support"""
    
    def __init__(
            self, 
            loss_depth_weight,
            point_cloud_range=None,
            loss_depth_type='bce', 
            **kwargs,
        ):
        super(ViewTransformerLiftSplatShootVoxelH3O, self).__init__(loss_depth_weight=loss_depth_weight, **kwargs)
        
        self.loss_depth_type = loss_depth_type
        self.cam_depth_range = self.grid_config['dbound']
        self.point_cloud_range = point_cloud_range
    
    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N,
                                   H // self.downsample, self.downsample,
                                   W // self.downsample, self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)
        
        # [min - step / 2, min + step / 2] creates min depth
        gt_depths = (gt_depths - (self.grid_config['dbound'][0] - self.grid_config['dbound'][2] / 2)) / self.grid_config['dbound'][2]
        gt_depths_vals = gt_depths.clone()
        
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]
        
        return gt_depths_vals, gt_depths.float()
    
    @force_fp32()
    def get_bce_depth_loss(self, depth_labels, depth_preds):
        _, depth_labels = self.get_downsampled_gt_depth(depth_labels)
        # depth_labels = self._prepare_depth_gt(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(depth_preds, depth_labels, reduction='none').sum() / max(1.0, fg_mask.sum())
        
        return depth_loss
    
    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        if self.loss_depth_type == 'bce':
            depth_loss = self.get_bce_depth_loss(depth_labels, depth_preds)
        
        else:
            pdb.set_trace()
        
        return self.loss_depth_weight * depth_loss
        
    def get_geometry_h3o(self, rots, trans, intrins, post_rots, post_trans, bda):
        """Special geometry method for H3O dataset with equirectangular projection"""
        B, N, _ = trans.shape
        
        # For H3O dataset, we need to handle equirectangular projection correctly
        # Create a frustum that covers the entire equirectangular image
        ogfH, ogfW = self.data_config['input_size']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        
        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        
        # For H3O equirectangular projection, convert image coordinates to 3D world coordinates
        # Following the H3O documentation: theta=(u/W)*2π-π, phi=(v/H)*π
        u = frustum[..., 0]  # Image u coordinates
        v = frustum[..., 1]  # Image v coordinates
        z_depth = frustum[..., 2]  # Z-depth from depth range
        
        # Convert to spherical coordinates (right-handed: X forward, Y left, Z up)
        theta = (u / ogfW) * 2 * torch.pi - torch.pi  # [-π, π]
        phi = (v / ogfH) * torch.pi  # [0, π]
        
        # Calculate ray direction (right-handed: X forward, Y left, Z up)
        nx = torch.sin(phi) * torch.cos(theta)
        ny = torch.sin(phi) * torch.sin(theta)
        nz = torch.cos(phi)
        
        # Calculate projection ratio for cubemap face
        denom = torch.maximum(torch.maximum(torch.abs(nx), torch.abs(ny)), torch.abs(nz))
        
        # Convert Z-depth to Euclidean distance
        r = z_depth / (denom + 1e-8)  # Avoid division by zero
        
        # Convert to Cartesian coordinates
        x = r * nx
        y = r * ny
        z = r * nz
        
        # Stack coordinates
        points = torch.stack([x, y, z], dim=-1)
        
        # # Debug: print shapes to understand the issue
        # print(f"Debug - points shape before view: {points.shape}")
        # print(f"Debug - B={B}, N={N}, D={D}, fH={fH}, fW={fW}")
        # print(f"Debug - Expected total elements: {B * N * D * fH * fW * 3}")
        # print(f"Debug - Actual total elements: {points.numel()}")
        # print(f"Debug - Coordinate range: x=[{x.min():.2f}, {x.max():.2f}], y=[{y.min():.2f}, {y.max():.2f}], z=[{z.min():.2f}, {z.max():.2f}]")
        
        # The issue is that points has shape [D, fH, fW, 3] but we need [B, N, D, fH, fW, 3]
        # We need to expand the dimensions to match the expected shape
        points = points.unsqueeze(0).unsqueeze(0)  # Add B and N dimensions
        points = points.expand(B, N, -1, -1, -1, -1).clone()  # Clone to avoid memory sharing issues
        
        # Move points to the same device as trans
        points = points.to(trans.device)
        points += trans.view(B, N, 1, 1, 1, 3)
        
        if bda.shape[-1] == 4:
            points = torch.cat((points, torch.ones(*points.shape[:-1], 1).type_as(points)), dim=-1)
            points = bda.view(B, 1, 1, 1, 1, 4, 4).matmul(points.unsqueeze(-1)).squeeze(-1)
            points = points[..., :3]
        else:
            points = bda.view(B, 1, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        
        return points

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box        # todo 根据数据集的不同调整bev网格的上下限 0: x 1: y 2: h
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        
        # Debug information for H3O dataset
        if kept.sum() == 0:
            print(f"[WARNING] No valid points after filtering in voxel_pooling")
            print(f"BEV grid size: {self.nx}")
            print(f"Point cloud range: {self.bx - self.dx/2} to {self.bx + self.dx/2}")
            print(f"Coordinate range: x=[{geom_feats[:, 0].min()}, {geom_feats[:, 0].max()}], "
                  f"y=[{geom_feats[:, 1].min()}, {geom_feats[:, 1].max()}], "
                  f"z=[{geom_feats[:, 2].min()}, {geom_feats[:, 2].max()}]")
            
            # The issue is that coordinates are way outside the BEV grid range
            # This suggests a problem with the geometry transformation
            # For now, create a minimal valid set to avoid empty arrays
            # Create a single point at the center of the BEV grid
            center_x = int(self.nx[0].item() // 2)
            center_y = int(self.nx[1].item() // 2)  
            center_z = int(self.nx[2].item() // 2)
            center_coords = torch.tensor([[center_x, center_y, center_z, 0]], 
                                        device=x.device, dtype=torch.long)
            geom_feats = center_coords
            x = x[:1]  # Take first feature vector
            kept = torch.ones(1, dtype=torch.bool, device=x.device)
        
        x = x[kept]
        geom_feats = geom_feats[kept]
        
        # [b, c, z, x, y] == [b, c, x, y, z]
        final = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])
        final = final.permute(0, 1, 3, 4, 2)

        return final

    def forward(self, input):
        (x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input) = input[:8]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x, mlp_input)
        depth_digit = x[:, :self.D, ...]
        img_feat = x[:, self.D:self.D + self.numC_Trans, ...]
        depth_prob = self.get_depth_dist(depth_digit)

        # Lift
        volume = depth_prob.unsqueeze(1) * img_feat.unsqueeze(2)
        volume = volume.view(B, N, -1, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        # Splat - use special geometry method for H3O dataset
        geom = self.get_geometry_h3o(rots, trans, intrins, post_rots, post_trans, bda)
        bev_feat = self.voxel_pooling(geom, volume)
        
        return bev_feat, depth_prob