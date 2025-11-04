import numpy as np
import glob
import os
import yaml
import json
from typing import Optional, Dict, Any

from mmdet.datasets import DATASETS
from mmdet3d.datasets import SemanticKITTIDataset


@DATASETS.register_module()
class CustomH3OLssDataset(SemanticKITTIDataset):
    r"""H3O (Human360Occ) Dataset.

    This dataset supports equirectangular panorama images and occupancy voxels.
    """

    def __init__(self, split, camera_used, occ_size, pc_range, 
                 split_mode='homo', grid='native', remap_yaml=None,
                 load_continuous=False, *args, **kwargs):
        
        # Store H3O-specific parameters
        self.split = split
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.camera_used = camera_used
        self.split_mode = split_mode
        self.grid = grid
        self.load_continuous = load_continuous
        
        # Load remap configuration
        if remap_yaml is None:
            yaml_path, _ = os.path.split(os.path.realpath(__file__))
            remap_yaml = os.path.join(yaml_path, 'h3o-remap-v4.yaml')
        
        self.dataset_config = yaml.safe_load(open(remap_yaml, 'r', encoding='utf-8'))
        self.remap_lut = self.get_remap_lut()
        
        # Initialize parent class first
        super().__init__(*args, **kwargs)
        
        # Load split information after parent initialization
        self.splits = self._load_splits()
        
        # Override data_infos with our custom loading
        self.data_infos = self._load_data_infos()

    def load_annotations(self, ann_file):
        """Override parent's load_annotations method for H3O dataset"""
        # Ensure splits are loaded before calling _load_data_infos
        if not hasattr(self, 'splits') or self.splits is None:
            self.splits = self._load_splits()
        return self._load_data_infos()

    def _load_splits(self):
        """Load split information from splits_homo.json or splits_heter.json"""
        # Check if data_root is available
        if not hasattr(self, 'data_root') or self.data_root is None:
            print(f"[WARNING] data_root not available, using default splits")
            return {}
        
        data_root = self.data_root
        split_file = f"splits_{self.split_mode}.json"
        split_path = os.path.join(data_root, split_file)
        
        if not os.path.isfile(split_path):
            # Try parent directory
            parent_path = os.path.join(os.path.dirname(data_root), split_file)
            if os.path.isfile(parent_path):
                split_path = parent_path
            else:
                raise FileNotFoundError(f"Split file not found: {split_path}")
        
        with open(split_path, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        
        return splits

    def get_remap_lut(self):
        """Get remap lookup table from YAML config"""
        lm = self.dataset_config['learning_map']
        maxkey = max(int(k) for k in lm.keys())
        lut = np.zeros((maxkey + 1000,), dtype=np.int32)
        for k, v in lm.items():
            lut[int(k)] = int(v)
        return lut

    def _list_frames_of_sequence(self, seq_dir: str):
        """List all frame IDs in a sequence"""
        sub = 'occupancy_gt' if self.grid == 'native' else 'occupancy_gt_128x128x16'
        pat = os.path.join(seq_dir, sub, '*.npy')
        files = sorted(glob.glob(pat))
        fids = [int(os.path.splitext(os.path.basename(f))[0]) for f in files]
        return fids

    def prepare_train_data(self, index):
        """Training data preparation."""
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None

        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        
        return example

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def get_ann_info(self, index):
        """Get annotation info for a given index."""
        info = self.data_infos[index]['voxel_path']
        if info is None:
            return None
        else:
            gt_occ = np.load(info)
            gt_occ = self.remap_lut[gt_occ.astype(np.uint16)].astype(np.float32)
        return gt_occ

    def get_data_info(self, index):
        """Get data info for a given index."""
        info = self.data_infos[index]

        input_dict = dict(
            occ_size=np.array(self.occ_size),
            pc_range=np.array(self.pc_range),
            sequence=info['sequence'],
            frame_id=info['frame_id'],
        )

        # Load images, intrinsics, extrinsics, voxels
        image_paths = []
        lidar2cam_rts = []
        lidar2img_rts = []
        cam_intrinsics = []

        for cam_type in self.camera_used:
            image_paths.append(info[cam_type + "_path"])
            
            # For H3O dataset, we use equirectangular panorama
            # Set up camera parameters for equirectangular projection
            # For equirectangular projection, we need special camera parameters
            
            # Create proper transformation matrices for equirectangular projection
            # For H3O dataset, we need to set up proper camera parameters
            # that work with the LSS (Lift-Splat-Shoot) framework
            
            # Use same image size as quad for compatibility
            img_h, img_w = 384, 1280  # Use same size as quad
            
            # For equirectangular projection, create proper camera parameters
            # that ensure the geometry transformation produces valid coordinates
            
            # Create proper intrinsic matrix for equirectangular projection
            cam_intrinsic = np.eye(3, dtype=np.float32)
            # Use focal length that ensures proper scaling for the point cloud range
            # H3O point cloud range: [-12.8, -12.8, -2.4, 12.8, 12.8, 0.8]
            # Use focal length that maps the point cloud range to image coordinates
            focal_length = min(img_w, img_h) / 2.0  # Use half of image size as focal length
            cam_intrinsic[0, 0] = focal_length  # fx
            cam_intrinsic[1, 1] = focal_length  # fy
            cam_intrinsic[0, 2] = img_w / 2     # cx: center u
            cam_intrinsic[1, 2] = img_h / 2     # cy: center v
            
            # Create proper transformation matrices
            # For equirectangular, we need to ensure the transformation produces
            # coordinates within the BEV grid range
            lidar2cam_rt = np.eye(4, dtype=np.float32)
            # Set translation to center the point cloud in the BEV grid
            lidar2cam_rt[0, 3] = 0.0  # x translation
            lidar2cam_rt[1, 3] = 0.0  # y translation  
            lidar2cam_rt[2, 3] = 0.0  # z translation
            
            # Create lidar2img transformation matrix
            # This should properly transform from lidar to image coordinates
            lidar2img_rt = np.eye(4, dtype=np.float32)
            # Copy the intrinsic parameters to the transformation matrix
            lidar2img_rt[:3, :3] = cam_intrinsic
            
            lidar2img_rts.append(lidar2img_rt)
            cam_intrinsics.append(cam_intrinsic)
            lidar2cam_rts.append(lidar2cam_rt)

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
            ))
        
        # gt_occ is None for test-set
        input_dict['gt_occ'] = self.get_ann_info(index)

        return input_dict

    def evaluate(self, results, logger=None, **kwargs):
        """Evaluation function for H3O dataset."""
        if results is None:
            logger.info('Skip Evaluation')
            return dict()
        
        if 'ssc_scores' in results:
            # for single-GPU inference
            ssc_scores = results['ssc_scores']
            class_ssc_iou = ssc_scores['iou_ssc'].tolist()
            res_dic = {
                "SC_Precision": ssc_scores['precision'].item(),
                "SC_Recall": ssc_scores['recall'].item(),
                "SC_IoU": ssc_scores['iou'],
                "SSC_mIoU": ssc_scores['iou_ssc_mean'],
            }
        else:
            # for multi-GPU inference
            assert 'ssc_results' in results
            ssc_results = results['ssc_results']
            completion_tp = sum([x[0] for x in ssc_results])
            completion_fp = sum([x[1] for x in ssc_results])
            completion_fn = sum([x[2] for x in ssc_results])
            
            tps = sum([x[3] for x in ssc_results])
            fps = sum([x[4] for x in ssc_results])
            fns = sum([x[5] for x in ssc_results])
            
            precision = completion_tp / (completion_tp + completion_fp)
            recall = completion_tp / (completion_tp + completion_fn)
            iou = completion_tp / \
                    (completion_tp + completion_fp + completion_fn)
            iou_ssc = tps / (tps + fps + fns + 1e-5)
            
            class_ssc_iou = iou_ssc.tolist()
            res_dic = {
                "SC_Precision": precision,
                "SC_Recall": recall,
                "SC_IoU": iou,
                "SSC_mIoU": iou_ssc[1:].mean(),
            }
        
        # H3O dataset class names (11 classes with unlabeled)
        class_names = [
            'empty', 'road', 'sidewalk', 'building', 'vegetation', 'car', 'truck', 
            'bus', 'two_wheeler', 'person', 'pole'
        ]
        for name, iou in zip(class_names, class_ssc_iou):
            res_dic["SSC_{}_IoU".format(name)] = iou
        
        eval_results = {}
        for key, val in res_dic.items():
            eval_results['h3o_{}'.format(key)] = round(val * 100, 2)
        
        # Return a key metric for checkpoint saving
        eval_results['h3o_combined_IoU'] = eval_results['h3o_SC_IoU'] + eval_results['h3o_SSC_mIoU']
        
        if logger is not None:
            logger.info('H3O Evaluation Results:')
            for key, val in eval_results.items():
                logger.info(f'{key}: {val}')
        
        return eval_results

    def _load_data_infos(self):
        """Load data infos for H3O dataset."""
        data_infos = []
        
        # Check if data_root is available
        if not hasattr(self, 'data_root') or self.data_root is None:
            print(f"[WARNING] data_root not available, returning empty data_infos")
            return data_infos
        
        # Check if splits is available
        if not hasattr(self, 'splits') or self.splits is None:
            print(f"[WARNING] splits not available, returning empty data_infos")
            return data_infos
        
        # Check if splits is a dictionary
        if not isinstance(self.splits, dict):
            print(f"[WARNING] splits is not a dictionary, returning empty data_infos")
            return data_infos
        
        # Check if split is available
        if not hasattr(self, 'split') or self.split is None:
            print(f"[WARNING] split not available, returning empty data_infos")
            return data_infos
        
        # Get sequences for current split
        split_sequences = self.splits.get(self.split, [])
        
        for seq in split_sequences:
            seq_dir = os.path.join(self.data_root, seq)
            if not os.path.isdir(seq_dir):
                continue
                
            # List all frames in this sequence
            frame_ids = self._list_frames_of_sequence(seq_dir)
            
            for fid in frame_ids:
                # Construct paths
                rgb_path = os.path.join(seq_dir, 'panorama_rgb', f"{fid:06d}.png")
                occ_path = os.path.join(seq_dir, 
                                      'occupancy_gt' if self.grid == 'native' else 'occupancy_gt_128x128x16', 
                                      f"{fid:06d}.npy")
                
                # Check if files exist
                if not os.path.isfile(rgb_path) or not os.path.isfile(occ_path):
                    continue
                
                # Create data info
                data_info = {
                    'sequence': seq,
                    'frame_id': f"{fid:06d}",
                    'panorama_rgb_path': rgb_path,
                    'voxel_path': occ_path,
                    # Camera parameters for equirectangular projection
                    'proj_matrix': np.eye(4),  # Identity matrix for equirect
                    'intrinsic': np.eye(3),    # Identity matrix for equirect
                    'T_velo_2_cam': np.eye(4), # Identity matrix for equirect
                }
                
                data_infos.append(data_info)
        
        return data_infos
