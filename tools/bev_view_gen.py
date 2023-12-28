from glob import glob
from mayavi import mlab
import os
import numpy as np
import argparse
from PIL import Image
import sys
print(sys.path)
import data_converter.kitti_io_data as SemanticKittiIO
from collections import Counter


def parse_args():
  parser = argparse.ArgumentParser(description='SCFormer visualize')
  parser.add_argument(
    '--dset_root',
    dest='dataset_root',
    default='/workspace/mnt/storage/shihao/EventSSC/SemanticKITTI/kitti/dataset/',
    metavar='DATASET',
    help='path to dataset root folder',
    type=str,
  )
  parser.add_argument(
    '--pred_root',
    default='E:\\SSC_Datasets\\data_odometry_voxels\\',
    metavar='DATASET',
    help='path to dataset prediction folder',
    type=str,
  )
  parser.add_argument(
    '--start_index',
    default=None,
    # default=239,
    help='start index of vis frame (//5 for skip frames)',
    type=str,
  )
  args = parser.parse_args()
  return args


def crop_black_border(input_image_path, output_image_path):
    # 打开图像
    image = Image.open(input_image_path)

    # 将图像转换为NumPy数组
    image_data = np.array(image)

    # # 找到图像非零像素的坐标范围
    # nonzero_indices = np.argwhere(image_data != 0)
    # top_left = nonzero_indices.min(axis=0)
    # bottom_right = nonzero_indices.max(axis=0)
    #
    # # 打印裁剪范围
    # print(f"Top left coordinates: ({top_left[1]}, {top_left[0]})")
    # print(f"Bottom right coordinates: ({bottom_right[1]}, {bottom_right[0]})")

    # mlab 1024*1024裁剪范围
    # Top
    # left
    # coordinates: (169, 143)
    # Bottom
    # right
    # coordinates: (854, 828)

    # mlab 2048*2048裁剪范围
    # Top
    # left
    # coordinates: (320, 293)
    # Bottom
    # right
    # coordinates: (1728, 1701)

    # # 使用预先找好的裁剪范围
    top_left = [293, 320]
    bottom_right = [1701, 1728]

    # 裁剪图像
    cropped_image = Image.fromarray(image_data[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1])

    # 保存裁剪后的图像
    cropped_image.save(output_image_path)


if __name__ == '__main__':
    args = parse_args()

    dset_root = args.dataset_root
    sequences_raw = sorted(glob(os.path.join(dset_root, 'dataset', 'sequences', '*')))      # 原始数据集序列
    sequences = sorted(glob(os.path.join(args.pred_root, 'sequences', '*')))     # 预测的标签序列

    # 加载config file
    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建相对路径
    config_path = os.path.join(script_dir, 'data_converter', 'semantic-kitti.yaml')
    remap_lut = SemanticKittiIO.get_remap_lut(config_path)

    # colormap
    colormap = SemanticKittiIO.get_cmap_semanticKITTI20()

    # Selecting training/validation set sequences only (labels unavailable for test set)
    # # trainval
    # sequences_raw = sequences_raw[:11]
    # sequences = sequences[:11]

    # val
    sequences_raw = [sequences_raw[8]]
    # sequences_raw = [sequences_raw[8]]
    # sequences = [sequences[8]]
    # 对于预测 只有08序列
    pass

    assert len(sequences) > 0, 'Error, no sequences on selected dataset root path'

    scale_divide = 1    # 尺度

    for sequence_raw, sequence in zip(sequences_raw, sequences):

        print('sequence path:', sequence)
        # LABEL_paths = sorted(glob(os.path.join(sequence, 'voxels', '*.label')))       # gt
        # LABEL_paths = sorted(glob(os.path.join(sequence, 'predictions', '*.label')))     # pseudo
        LABEL_paths = sorted(glob(os.path.join(sequence,
                                               'sphere-10-01-class0-frust', 'predictions_dist_refine_2', '*.label')))     # pseudo refined
        # LABEL_paths = sorted(glob(os.path.join(sequence, 'gt_refine_2', '*.label')))  # gt refined
        INVALID_paths = sorted(glob(os.path.join(sequence_raw, 'voxels', '*.invalid')))
        out_dir = os.path.join(sequence, 'bev_vis')
        # out_dir = os.path.join(sequence, 'bev_vis_pseudo')
        if not os.path.exists(out_dir):
          os.mkdir(out_dir)

        # 删除多余的非跳帧伪标签
        redundancy_index = []
        for i in range(len(LABEL_paths)):
            # 跳帧标签序号是5的倍数
            if LABEL_paths[i].split('.')[-2].isdigit() and int(LABEL_paths[i].split('.')[-2]) % 5 != 0:
                redundancy_index.append(i)
        for i in redundancy_index:
            print('ignore:', LABEL_paths[i])
        for i in redundancy_index[::-1]:
            # 使用倒序循环防止索引错乱
            del LABEL_paths[i]

        print('len(LABEL_paths):', str(len(LABEL_paths)))
        print('len(INVALID_paths):', str(len(INVALID_paths)))
        # assert len(LABEL_paths) == len(INVALID_paths)

        for i in range(len(LABEL_paths)):

            # start from given index
            if args.start_index is not None:
                i += int(args.start_index)
                if i >= len(LABEL_paths):
                    break

            # i = 861        # for debug
            LABEL = SemanticKittiIO._read_label_SemKITTI(LABEL_paths[i])
            LABEL = remap_lut[LABEL.astype(np.uint16)].astype(np.float32)  # Remap 20 classes semanticKITTI SSC

            try:
                INVALID = SemanticKittiIO._read_invalid_SemKITTI(INVALID_paths[i])
                LABEL[np.isclose(INVALID, 1)] = 255  # Setting to unknown all voxels marked on invalid mask...
            except:
                print('Warning: No Valid Mask Found. Visualize with No Mask...')

            grid_dimensions = [256, 32, 256]
            LABEL = np.moveaxis(LABEL.reshape([int(grid_dimensions[0] / scale_divide),
                                               int(grid_dimensions[2] / scale_divide),
                                               int(grid_dimensions[1] / scale_divide)]), [0, 1, 2], [0, 2, 1])

            # Creating figure object
            # figure = mlab.figure(figure='Figure', bgcolor=(1, 1, 1), size=(1024, 1024))   # 白色背景
            figure = mlab.figure(figure='Figure', bgcolor=(0, 0, 0), size=(2048, 2048))     # 黑色背景

            # Get grid coordinates X, Y, Z
            grid_coords, _, _, _ = SemanticKittiIO.get_grid_coords([LABEL.shape[0], LABEL.shape[2], LABEL.shape[1]], 0.2)
            grid_coords = np.vstack((grid_coords.T, np.moveaxis(LABEL, [0, 1, 2], [0, 2, 1]).reshape(-1))).T
            # Obtaining voxels with semantic class
            occupied_voxels = grid_coords[(grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 255)]

            # fix color bug
            occupied_voxels[-1][-1] = 19.0  # 给最后N个点赋值为19类解决可视化bug
            occupied_voxels[-2][-1] = 18.0
            occupied_voxels[-3][-1] = 17.0
            occupied_voxels[-4][-1] = 16.0
            occupied_voxels[-5][-1] = 15.0
            occupied_voxels[-6][-1] = 14.0
            occupied_voxels[-7][-1] = 13.0
            occupied_voxels[-8][-1] = 12.0
            occupied_voxels[-9][-1] = 11.0
            occupied_voxels[-10][-1] = 10.0
            occupied_voxels[-11][-1] = 9.0
            occupied_voxels[-12][-1] = 8.0
            occupied_voxels[-13][-1] = 7.0
            occupied_voxels[-14][-1] = 6.0
            occupied_voxels[-15][-1] = 5.0
            occupied_voxels[-16][-1] = 4.0
            occupied_voxels[-17][-1] = 3.0
            occupied_voxels[-18][-1] = 2.0
            occupied_voxels[-19][-1] = 1.0

            # # debug
            # print('Counter(data)\n', Counter(sorted(occupied_voxels[:, 3].astype(np.uint8))))

            # Plot as points with cube as mode, resolution is 0.2
            plt_plot = mlab.points3d(occupied_voxels[:, 0], occupied_voxels[:, 1], occupied_voxels[:, 2], occupied_voxels[:, 3],
                                     colormap='viridis', scale_factor=0.20, mode='cube', opacity=1)

            # Scaling all voxels the same size
            plt_plot.glyph.scale_mode = 'scale_by_vector'

            # Setting correct colormap
            plt_plot.module_manager.scalar_lut_manager.lut.table = colormap

            # debug
            # print('ColorMap\n', plt_plot.module_manager.scalar_lut_manager.lut.table.to_array())

            # 调整相机视角到bev
            mlab.view(azimuth=0, elevation=0)   # 调整到BEV视角 azimuth:xy平面角度 elevation:z轴角度
            figure.scene.camera.parallel_projection = True  # 开启平行投影

            # mlab.show()   # 可视化3D交互式窗口

            # save bev vis
            filename, extension = os.path.splitext(os.path.basename(LABEL_paths[i]))
            bev_filename = os.path.join(out_dir, filename + '.png')
            mlab.savefig(bev_filename)     # 保存当前窗口图像 仅在不使用mlab.show的情况下work
            mlab.close()
            crop_black_border(bev_filename, bev_filename)
            print(bev_filename, 'is SAVED!')
            print('==========')
            # break
