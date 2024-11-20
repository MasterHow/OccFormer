import numpy as np
import glob
import os
import yaml

from mmdet.datasets import DATASETS
from mmdet3d.datasets import SemanticKITTIDataset

@DATASETS.register_module()
class CustomQuadLssDataset(SemanticKITTIDataset):
    r"""Quad Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, split, camera_used, occ_size, pc_range, 
                 load_continuous=False, *args, **kwargs):
        
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.camera_map = {'images_unfold': 'images_unfold', 'images_raw': 'images_raw'}
        self.camera_used = [self.camera_map[camera] for camera in camera_used]
        self.multi_scales = ["1_1", "1_2", "1_4", "1_8", "1_16"]
        # 装载映射关系
        yaml_path, _ = os.path.split(os.path.realpath(__file__))
        self.dataset_config = yaml.safe_load(
            open(os.path.join(yaml_path, 'quad-ssc-remap.yaml'), 'r', encoding='utf-8'))
        self.remap_lut = self.get_remap_lut()
        
        self.load_continuous = load_continuous
        self.splits = {'train': ['2024-06-20-20-49-54',
                                '2024-06-20-20-52-21',
                                '2024-06-20-21-05-40',
                                '2024-06-20-21-19-23',
                                '2024-06-20-21-15-21',
                                '2024-06-20-21-05-05',
                                '2024-06-20-21-11-28'],
                       'val': ['2024-06-20-21-16-11',
                              '2024-06-20-21-08-40',
                              '2024-06-20-20-46-59'],
                       "trainval": ['2024-06-20-20-49-54',
                                '2024-06-20-20-52-21',
                                '2024-06-20-21-05-40',
                                '2024-06-20-21-19-23',
                                '2024-06-20-21-15-21',
                                '2024-06-20-21-05-05',
                                '2024-06-20-21-11-28',
                                '2024-06-20-21-16-11',
                                '2024-06-20-21-08-40',
                                '2024-06-20-20-46-59'],
                       'test': ['2024-06-20-21-16-11',
                              '2024-06-20-21-08-40',
                              '2024-06-20-20-46-59'],
                       "test-submit": ['2024-06-20-21-16-11',
                              '2024-06-20-21-08-40',
                              '2024-06-20-20-46-59'],}
        
        self.sequences = self.splits[split]
        self.n_classes = 7
        super().__init__(*args, **kwargs)
        self._set_group_flag()

    def get_remap_lut(self):
        '''
        remap_lut to remap classes of quad-ssc for training...
        :return:
        '''

        # make lookup table for mapping
        maxkey = max(self.dataset_config['learning_map'].keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 1000), dtype=np.int32)
        remap_lut[list(self.dataset_config['learning_map'].keys())] = list(self.dataset_config['learning_map'].values())

        # # 原本LABEL中0代表unlabel，255代表空体素，现在把255并入unlabel: 将 255 映射为 0，0 映射为 255
        # remap_lut[255] = 0  # 直接将索引 255 的位置映射为 0
        # remap_lut[0] = 255  # 直接将索引 0 的位置映射为 255
        # # 教师模型保存的标签不需要
        # pass

        # # # in completion we have to distinguish empty and invalid voxels.
        # # # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
        # remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
        # remap_lut[0] = 0  # only 'empty' stays 'empty'.

        return remap_lut

    @staticmethod
    def read_calib(calib_path=None):
        """Read calibration data from maps.npz for camera to LiDAR transformation."""

        # # Load map data from maps.npz
        # data = np.load(calib_path)

        # Camera to LiDAR transform (R and T)
        R_lidar_to_camera = np.array([
            [0.965595, 0.02617695, 0.25873031],
            [-0.02528499, 0.99965732, -0.00677509],
            [-0.258819, 0., 0.965926]
        ])
        T_lidar_to_camera = np.array([0.08, 0, -0.02])

        # Prepare the 4x4 transformation matrix for LiDAR to camera
        T_velo_2_cam = np.eye(4)
        T_velo_2_cam[:3, :3] = R_lidar_to_camera
        T_velo_2_cam[:3, 3] = T_lidar_to_camera

        # Extrinsics (projection matrices)
        calib_out = {
            "P2": np.eye(4),  # example, replace with actual data if necessary
            "P3": np.eye(4),  # example, replace with actual data if necessary
            "Tr": T_velo_2_cam
        }

        return calib_out

    def load_annotations(self, ann_file=None):
        scans = []
        for sequence in self.sequences:
            # 读取标定文件
            # calib = self.read_calib(os.path.join(self.data_root, "maps.npz"))     # 原始分辨率
            calib = self.read_calib(os.path.join(self.data_root, "maps_384.npz"))       # 384*1280
            P2 = calib["P2"]
            P3 = calib["P3"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix_2 = P2 @ T_velo_2_cam
            proj_matrix_3 = P3 @ T_velo_2_cam

            # voxel_base_path = os.path.join(self.data_root, "sequences", sequence, "new_static_label")
            voxel_base_path = os.path.join(self.data_root, "sequences", sequence, "new_finally_label")
            img_base_path = os.path.join(self.data_root, "sequences", sequence)

            # 根据加载模式确定文件路径
            if self.load_continuous:
                id_base_path = os.path.join(self.data_root, "sequences", sequence, 'images_unfold', '*.jpg')
            else:
                id_base_path = os.path.join(self.data_root, "sequences", sequence, 'new_static_label', '*.npy')

            # 获取文件列表，并进行排序
            id_paths = sorted(glob.glob(id_base_path))

            if self.test_mode:
                # 遍历每隔五个文件
                for id_path in id_paths[::5]:
                    img_id = os.path.basename(id_path).split(".")[0]

                    # 构建信息字典
                    info = {
                        "sequence": sequence,
                        "frame_id": img_id,
                        "P2": P2,
                        "P3": P3,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix_2": proj_matrix_2,
                        "proj_matrix_3": proj_matrix_3,
                        "voxel_path": os.path.join(voxel_base_path, img_id + '.npy')
                    }

                    # 添加每个摄像头的路径
                    for cam_type in self.camera_used:
                        info[f"{cam_type}_path"] = os.path.join(img_base_path, cam_type, f"{img_id}.jpg")

                    # 检查体素路径是否存在
                    if not os.path.exists(info["voxel_path"]):
                        info["voxel_path"] = None

                    scans.append(info)

            else:
                # 遍历每隔五个文件
                # debug eval hook
                # for id_path in id_paths[::1000]:
                for id_path in id_paths[::5]:
                    img_id = os.path.basename(id_path).split(".")[0]

                    # 构建信息字典
                    info = {
                        "sequence": sequence,
                        "frame_id": img_id,
                        "P2": P2,
                        "P3": P3,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix_2": proj_matrix_2,
                        "proj_matrix_3": proj_matrix_3,
                        "voxel_path": os.path.join(voxel_base_path, img_id + '.npy')
                    }

                    # 添加每个摄像头的路径
                    for cam_type in self.camera_used:
                        info[f"{cam_type}_path"] = os.path.join(img_base_path, cam_type, f"{img_id}.jpg")

                    # 检查体素路径是否存在
                    if not os.path.exists(info["voxel_path"]):
                        info["voxel_path"] = None

                    scans.append(info)

        return scans

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        
        input_dict = self.get_data_info(index)
        if input_dict is None:
            print('found None in training data')
            return None
        
        # init for pipeline
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
        info = self.data_infos[index]['voxel_path']
        if info is None:
            return None
        else:
            gt_occ = np.load(info)
            gt_occ = self.remap_lut[gt_occ.astype(np.uint16)].astype(np.float32)  # Remap classes Quad SSC

        return gt_occ
        # return None if info is None else np.load(info)

    def get_data_info(self, index):
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

            # 这里进行条件处理，只支持 'images_unfold' 和 'images_raw'
            if cam_type == 'images_unfold':
                proj_matrix_key = 'proj_matrix_2'
                intrinsic_key = 'P2'
            elif cam_type == 'images_raw':
                proj_matrix_key = 'proj_matrix_3'
                intrinsic_key = 'P3'
            else:
                raise ValueError(f"Unsupported camera type: {cam_type}")

            lidar2img_rts.append(info[proj_matrix_key])
            cam_intrinsics.append(info[intrinsic_key])
            lidar2cam_rts.append(info['T_velo_2_cam'])

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
            )
        )

        # Ground truth occupancy is None for test-set
        input_dict['gt_occ'] = self.get_ann_info(index)

        return input_dict

    def evaluate(self, results, logger=None, **kwargs):
        if results is None:
            logger.info('Skip Evaluation')
        
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
        
        # class_names = [
        #     'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
        #     'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
        #     'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
        #     'pole', 'traffic-sign'
        # ]
        class_names = [
            'unlabeled', 'vehicle', 'person', 'road', 'building', 'vegetation',
            'terrain',
        ]
        for name, iou in zip(class_names, class_ssc_iou):
            res_dic["SSC_{}_IoU".format(name)] = iou
        
        eval_results = {}
        for key, val in res_dic.items():
            eval_results['semkitti_{}'.format(key)] = round(val * 100, 2)
        
        eval_results['semkitti_combined_IoU'] = eval_results['semkitti_SC_IoU'] + eval_results['semkitti_SSC_mIoU']
        
        if logger is not None:
            logger.info('SemanticKITTI SSC Evaluation')
            logger.info(eval_results)
        
        return eval_results
        
        