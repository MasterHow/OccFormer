import numpy as np
import yaml
import imageio
# import volumentations as V
from volumentations import *


def unpack(compressed):
  ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
  uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
  uncompressed[::8] = compressed[:] >> 7 & 1
  uncompressed[1::8] = compressed[:] >> 6 & 1
  uncompressed[2::8] = compressed[:] >> 5 & 1
  uncompressed[3::8] = compressed[:] >> 4 & 1
  uncompressed[4::8] = compressed[:] >> 3 & 1
  uncompressed[5::8] = compressed[:] >> 2 & 1
  uncompressed[6::8] = compressed[:] >> 1 & 1
  uncompressed[7::8] = compressed[:] & 1

  return uncompressed


def img_normalize(img, mean, std):
  img = img.astype(np.float32) / 255.0
  img = img - mean
  img = img / std

  return img


def pack(array):
  """ convert a boolean array into a bitwise array. """
  array = array.reshape((-1))

  #compressing bit flags.
  # yapf: disable
  compressed = array[::8] << 7 | array[1::8] << 6  | array[2::8] << 5 | array[3::8] << 4 | array[4::8] << 3 | array[5::8] << 2 | array[6::8] << 1 | array[7::8]
  # yapf: enable

  return np.array(compressed, dtype=np.uint8)


def get_grid_coords(dims, resolution):
  '''
  :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
  :return coords_grid: is the center coords of voxels in the grid
  '''

  # The sensor in centered in X (we go to dims/2 + 1 for the histogramdd)
  g_xx = np.arange(-dims[0]/2, dims[0]/2 + 1)
  # The sensor is in Y=0 (we go to dims + 1 for the histogramdd)
  g_yy = np.arange(0, dims[1] + 1)
  # The sensor is in Z=1.73. I observed that the ground was to voxel levels above the grid bottom, so Z pose is at 10
  # if bottom voxel is 0. If we want the sensor to be at (0, 0, 0), then the bottom in z is -10, top is 22
  # (we go to 22 + 1 for the histogramdd)
  # ATTENTION.. Is 11 for old grids.. 10 for new grids (v1.1) (https://github.com/PRBonn/semantic-kitti-api/issues/49)
  sensor_pose = 10
  g_zz = np.arange(0 - sensor_pose, dims[2] - sensor_pose + 1)

  # Obtaining the grid with coords...
  xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
  coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
  coords_grid = coords_grid.astype(np.float)

  coords_grid = (coords_grid * resolution) + resolution/2

  temp = np.copy(coords_grid)
  temp[:, 0] = coords_grid[:, 1]
  temp[:, 1] = coords_grid[:, 0]
  coords_grid = np.copy(temp)

  return coords_grid, g_xx, g_yy, g_zz


def _get_remap_lut(config_path):
  '''
  remap_lut to remap classes of semantic kitti for training...
  :return:
  '''

  dataset_config = yaml.safe_load(open(config_path, 'r'))
  # make lookup table for mapping
  maxkey = max(dataset_config['learning_map'].keys())

  # +100 hack making lut bigger just in case there are unknown labels
  remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
  remap_lut[list(dataset_config['learning_map'].keys())] = list(dataset_config['learning_map'].values())

  # in completion we have to distinguish empty and invalid voxels.
  # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
  remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
  remap_lut[0] = 0  # only 'empty' stays 'empty'.

  return remap_lut


def _read_SemKITTI(path, dtype, do_unpack):
  bin = np.fromfile(path, dtype=dtype)  # Flattened array
  if do_unpack:
    bin = unpack(bin)
  return bin


def _read_label_SemKITTI(path):
  label = _read_SemKITTI(path, dtype=np.uint16, do_unpack=False).astype(np.float32)
  return label


def _read_invalid_SemKITTI(path):
  invalid = _read_SemKITTI(path, dtype=np.uint8, do_unpack=True)
  return invalid


def _read_occluded_SemKITTI(path):
  occluded = _read_SemKITTI(path, dtype=np.uint8, do_unpack=True)
  return occluded


def _read_occupancy_SemKITTI(path):
  occupancy = _read_SemKITTI(path, dtype=np.uint8, do_unpack=True).astype(np.float32)
  return occupancy


def _read_rgb_SemKITTI(path):
  rgb = np.asarray(imageio.imread(path))
  return rgb


def _read_pointcloud_SemKITTI(path):
  'Return pointcloud semantic kitti with remissions (x, y, z, intensity)'
  pointcloud = _read_SemKITTI(path, dtype=np.float32, do_unpack=False)
  pointcloud = pointcloud.reshape((-1, 4))
  return pointcloud


def _read_calib_SemKITTI(calib_path):
  """
  :param calib_path: Path to a calibration text file.
  :return: dict with calibration matrices.
  """
  calib_all = {}
  with open(calib_path, 'r') as f:
    for line in f.readlines():
      if line == '\n':
        break
      key, value = line.split(':', 1)
      calib_all[key] = np.array([float(x) for x in value.split()])

  # reshape matrices
  calib_out = {}
  calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
  calib_out['Tr'] = np.identity(4)  # 4x4 matrix
  calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)
  return calib_out


def get_remap_lut(path):
  '''
  remap_lut to remap classes of semantic kitti for training...
  :return:
  '''

  dataset_config = yaml.safe_load(open(path, 'r'))

  # make lookup table for mapping
  maxkey = max(dataset_config['learning_map'].keys())

  # +100 hack making lut bigger just in case there are unknown labels
  remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
  remap_lut[list(dataset_config['learning_map'].keys())] = list(dataset_config['learning_map'].values())

  # in completion we have to distinguish empty and invalid voxels.
  # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
  remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
  remap_lut[0] = 0  # only 'empty' stays 'empty'.

  return remap_lut


def data_augmentation_3Dflips(flip, data):
  # Hao: LMSCNet使用的3D Flip总共有3种情况：x轴翻转，y轴翻转和xy轴一起翻转，但是z轴不会参加Flip

  # The .copy() is done to avoid negative strides of the numpy array caused by the way numpy manages the data
  # into memory. This gives errors when trying to pass the array to torch sensors.. Solution seen in:
  # https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
  # Dims -> {XZY}
  # Flipping around the X axis...
  if np.isclose(flip, 1):
    data = np.flip(data, axis=0).copy()

  # Flipping around the Y axis...
  if np.isclose(flip, 2):
    data = np.flip(data, 2).copy()

  # Flipping around the X and the Y axis...
  if np.isclose(flip, 3):
    data = np.flip(np.flip(data, axis=0), axis=2).copy()

  return data


def data_augmentation_3Dflips_v2(flip, data):
  """
  与原始版本不同的是，v2的输入维度为 B, X, Z, Y
  """
  # Hao: LMSCNet使用的3D Flip总共有3种情况：x轴翻转，y轴翻转和xy轴一起翻转，但是z轴不会参加Flip

  # The .copy() is done to avoid negative strides of the numpy array caused by the way numpy manages the data
  # into memory. This gives errors when trying to pass the array to torch sensors.. Solution seen in:
  # https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
  # Dims -> {XZY}
  # Flipping around the X axis...
  if np.isclose(flip, 1):
    data = np.flip(data, axis=1).copy()

  # Flipping around the Y axis...
  if np.isclose(flip, 2):
    data = np.flip(data, 3).copy()

  # Flipping around the X and the Y axis...
  if np.isclose(flip, 3):
    data = np.flip(np.flip(data, axis=1), axis=3).copy()

  return data


# def get_point_augmentation(volume_augmentations_path=
#                      '/workspace/mnt/storage/shihao/shihao-cephs/SCFormer/SSC_configs/augmentation/volumentations_aug.yaml'):
#   """
#   3D point cloud augmentation from https://volumentations.readthedocs.io/
#   """
#   return V.load(
#                 volume_augmentations_path, data_format="yaml"
#             )


# V1
def get_volume_augmentation(patch_size):
  """
  3D volume augmentation from https://github.com/ZFTurbo/volumentations
  Input: [D H W]
  """
  return Compose([
    Rotate((-180, 180), (-7.5, 7.5), (-7.5, 7.5), p=0.5),
    RandomCropFromBorders(crop_value=0.1, p=0.5),
    # ElasticTransform((0, 0.25), interpolation=2, p=0.1),    # 速度比较慢注释掉
    Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
    Flip(0, p=0.5),   # z 轴需要翻转吗？
    Flip(1, p=0.5),
    Flip(2, p=0.5),
    RandomRotate90((1, 2), p=0.5),
    # GaussianNoise(var_limit=(0, 5), p=0.2),       # 可能出现非1值
    # RandomGamma(gamma_limit=(80, 120), p=0.2),    # 1做gama调制没有意义
  ], p=1.0)


# V2
def get_volume_augmentation_v2(patch_size):
  """
  3D volume augmentation from https://github.com/ZFTurbo/volumentations
  Input: [D H W]
  V2相比于V1去掉了Z轴翻转
  """
  return Compose([
    Rotate((-180, 180), (-7.5, 7.5), (-7.5, 7.5), p=0.5),
    RandomCropFromBorders(crop_value=0.1, p=0.5),
    # ElasticTransform((0, 0.25), interpolation=2, p=0.1),    # 速度比较慢注释掉
    Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
    # Flip(0, p=0.5),   # z 轴需要翻转吗？
    Flip(1, p=0.5),
    Flip(2, p=0.5),
    RandomRotate90((1, 2), p=0.5),
    # GaussianNoise(var_limit=(0, 5), p=0.2),       # 可能出现非1值
    # RandomGamma(gamma_limit=(80, 120), p=0.2),    # 1做gama调制没有意义
  ], p=1.0)


# V3
def get_volume_augmentation_v3(patch_size):
  """
  3D volume augmentation from https://github.com/ZFTurbo/volumentations
  Input: [D H W]
  V3相比于V2去掉了Crop
  """
  return Compose([
    Rotate((-180, 180), (-7.5, 7.5), (-7.5, 7.5), p=0.5),
    # RandomCropFromBorders(crop_value=0.1, p=0.5),
    # ElasticTransform((0, 0.25), interpolation=2, p=0.1),    # 速度比较慢注释掉
    Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
    # Flip(0, p=0.5),   # z 轴需要翻转吗？
    Flip(1, p=0.5),
    Flip(2, p=0.5),
    RandomRotate90((1, 2), p=0.5),
    # GaussianNoise(var_limit=(0, 5), p=0.2),       # 可能出现非1值
    # RandomGamma(gamma_limit=(80, 120), p=0.2),    # 1做gama调制没有意义
  ], p=1.0)


# V4
def get_volume_augmentation_v4(patch_size):
  """
  3D volume augmentation from https://github.com/ZFTurbo/volumentations
  Input: [D H W]
  V4相比于V3去掉了xy轴随机旋转0-3次90°
  """
  return Compose([
    Rotate((-180, 180), (-7.5, 7.5), (-7.5, 7.5), p=0.5),
    # RandomCropFromBorders(crop_value=0.1, p=0.5),
    # ElasticTransform((0, 0.25), interpolation=2, p=0.1),    # 速度比较慢注释掉
    Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
    # Flip(0, p=0.5),   # z 轴需要翻转吗？
    Flip(1, p=0.5),
    Flip(2, p=0.5),
    # RandomRotate90((1, 2), p=0.5),
    # GaussianNoise(var_limit=(0, 5), p=0.2),       # 可能出现非1值
    # RandomGamma(gamma_limit=(80, 120), p=0.2),    # 1做gama调制没有意义
  ], p=1.0)


# V5
def get_volume_augmentation_v5(patch_size):
  """
  3D volume augmentation from https://github.com/ZFTurbo/volumentations
  Input: [D H W]
  V5相比于V4增加了random scale
  """
  return Compose([
    Rotate((-180, 180), (-7.5, 7.5), (-7.5, 7.5), p=0.5),
    RandomScale(scale_limit=[0.9, 1.1], p=0.5),
    # RandomCropFromBorders(crop_value=0.1, p=0.5),
    # ElasticTransform((0, 0.25), interpolation=2, p=0.1),    # 速度比较慢注释掉
    Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
    # Flip(0, p=0.5),   # z 轴需要翻转吗？
    Flip(1, p=0.5),
    Flip(2, p=0.5),
    # RandomRotate90((1, 2), p=0.5),
    # GaussianNoise(var_limit=(0, 5), p=0.2),       # 可能出现非1值
    # RandomGamma(gamma_limit=(80, 120), p=0.2),    # 1做gama调制没有意义
  ], p=1.0)


# V6
def get_volume_augmentation_v6(patch_size):
  """
  3D volume augmentation from https://github.com/ZFTurbo/volumentations
  Input: [D H W]
  V6相比于V4增加了random drop plane
  """
  return Compose([
    Rotate((-180, 180), (-7.5, 7.5), (-7.5, 7.5), p=0.5),
    RandomDropPlane(plane_drop_prob=0.1, axes=(0,), p=1.0),
    # RandomCropFromBorders(crop_value=0.1, p=0.5),
    # ElasticTransform((0, 0.25), interpolation=2, p=0.1),    # 速度比较慢注释掉
    Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
    # Flip(0, p=0.5),   # z 轴需要翻转吗？
    Flip(1, p=0.5),
    Flip(2, p=0.5),
    # RandomRotate90((1, 2), p=0.5),
    # GaussianNoise(var_limit=(0, 5), p=0.2),       # 可能出现非1值
    # RandomGamma(gamma_limit=(80, 120), p=0.2),    # 1做gama调制没有意义
  ], p=1.0)


# V7
def get_volume_augmentation_v7(patch_size):
  """
  3D volume augmentation from https://github.com/ZFTurbo/volumentations
  Input: [D H W]
  V7相比于V4取消了xy轴的旋转
  """
  return Compose([
    Rotate((-180, 180), (0, 0), (0, 0), p=0.5),
    # RandomCropFromBorders(crop_value=0.1, p=0.5),
    # ElasticTransform((0, 0.25), interpolation=2, p=0.1),    # 速度比较慢注释掉
    Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
    # Flip(0, p=0.5),   # z 轴需要翻转吗？
    Flip(1, p=0.5),
    Flip(2, p=0.5),
    # RandomRotate90((1, 2), p=0.5),
    # GaussianNoise(var_limit=(0, 5), p=0.2),       # 可能出现非1值
    # RandomGamma(gamma_limit=(80, 120), p=0.2),    # 1做gama调制没有意义
  ], p=1.0)


# V8
def get_volume_augmentation_v8(patch_size):
  """
  3D volume augmentation from https://github.com/ZFTurbo/volumentations
  Input: [D H W]
  V8相比于V4xy轴旋转角度翻倍，测试旋转角度参数
  """
  return Compose([
    Rotate((-180, 180), (-15, 15), (-15, 15), p=0.5),
    # RandomCropFromBorders(crop_value=0.1, p=0.5),
    # ElasticTransform((0, 0.25), interpolation=2, p=0.1),    # 速度比较慢注释掉
    Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
    # Flip(0, p=0.5),   # z 轴需要翻转吗？
    Flip(1, p=0.5),
    Flip(2, p=0.5),
    # RandomRotate90((1, 2), p=0.5),
    # GaussianNoise(var_limit=(0, 5), p=0.2),       # 可能出现非1值
    # RandomGamma(gamma_limit=(80, 120), p=0.2),    # 1做gama调制没有意义
  ], p=1.0)


# V9
def get_volume_augmentation_v9(patch_size):
  """
  3D volume augmentation from https://github.com/ZFTurbo/volumentations
  Input: [D H W]
  V8相比于V4xy轴旋转角度缩小一半，测试旋转角度参数
  """
  return Compose([
    Rotate((-180, 180), (-3.75, 3.75), (-3.75, 3.75), p=0.5),
    # RandomCropFromBorders(crop_value=0.1, p=0.5),
    # ElasticTransform((0, 0.25), interpolation=2, p=0.1),    # 速度比较慢注释掉
    Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
    # Flip(0, p=0.5),   # z 轴需要翻转吗？
    Flip(1, p=0.5),
    Flip(2, p=0.5),
    # RandomRotate90((1, 2), p=0.5),
    # GaussianNoise(var_limit=(0, 5), p=0.2),       # 可能出现非1值
    # RandomGamma(gamma_limit=(80, 120), p=0.2),    # 1做gama调制没有意义
  ], p=1.0)


# V10
def get_volume_augmentation_v10(patch_size):
  """
  3D volume augmentation from https://github.com/ZFTurbo/volumentations
  Input: [D H W]
  V10相比于V4xy轴旋转角度扩大4倍，测试旋转角度参数
  """
  return Compose([
    Rotate((-180, 180), (-30, 30), (-30, 30), p=0.5),
    # RandomCropFromBorders(crop_value=0.1, p=0.5),
    # ElasticTransform((0, 0.25), interpolation=2, p=0.1),    # 速度比较慢注释掉
    Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
    # Flip(0, p=0.5),   # z 轴需要翻转吗？
    Flip(1, p=0.5),
    Flip(2, p=0.5),
    # RandomRotate90((1, 2), p=0.5),
    # GaussianNoise(var_limit=(0, 5), p=0.2),       # 可能出现非1值
    # RandomGamma(gamma_limit=(80, 120), p=0.2),    # 1做gama调制没有意义
  ], p=1.0)


# V11
def get_volume_augmentation_v11(patch_size):
  """
  3D volume augmentation from https://github.com/ZFTurbo/volumentations
  Input: [D H W]
  V11相比于V4增加了Grid Drop
  """
  return Compose([
    Rotate((-180, 180), (-7.5, 7.5), (-7.5, 7.5), p=0.5),
    GridDropout(ratio=0.5, random_offset=True, p=0.5),
    # RandomCropFromBorders(crop_value=0.1, p=0.5),
    # ElasticTransform((0, 0.25), interpolation=2, p=0.1),    # 速度比较慢注释掉
    Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
    # Flip(0, p=0.5),   # z 轴需要翻转吗？
    Flip(1, p=0.5),
    Flip(2, p=0.5),
    # RandomRotate90((1, 2), p=0.5),
    # GaussianNoise(var_limit=(0, 5), p=0.2),       # 可能出现非1值
    # RandomGamma(gamma_limit=(80, 120), p=0.2),    # 1做gama调制没有意义
  ], p=1.0)


# V12
def get_volume_augmentation_v12(patch_size):
  """
  3D volume augmentation from https://github.com/ZFTurbo/volumentations
  Input: [D H W]
  """
  return Compose([
    Rotate((-180, 180), (-7.5, 7.5), (-7.5, 7.5), p=0.5),
    Flip(0, p=0.5),
    Flip(1, p=0.5),
    Flip(2, p=0.5),
    RandomRotate90((1, 2), p=0.5),
  ], p=1.0)


# V13
def get_volume_augmentation_v13(patch_size):
  """
  3D volume augmentation from https://github.com/ZFTurbo/volumentations
  Input: [D H W]
  与V12相比去除了坐标轴抖动
  """
  return Compose([
    Flip(0, p=0.5),
    Flip(1, p=0.5),
    Flip(2, p=0.5),
    RandomRotate90((1, 2), p=0.5),
  ], p=1.0)


# V14
def get_volume_augmentation_v14(patch_size):
  """
  3D volume augmentation from https://github.com/ZFTurbo/volumentations
  Input: [D H W]
  与V13相比去除了z轴flip
  """
  return Compose([
    Flip(1, p=0.5),
    Flip(2, p=0.5),
    RandomRotate90((1, 2), p=0.5),
  ], p=1.0)


def get_cmap_semanticKITTI20():
  colors = np.array([
    # [0  , 0  , 0, 255],
    [100, 150, 245, 255],
    [100, 230, 245, 255],
    [30, 60, 150, 255],
    [80, 30, 180, 255],
    [100, 80, 250, 255],
    [255, 30, 30, 255],
    [255, 40, 200, 255],
    [150, 30, 90, 255],
    [255, 0, 255, 255],
    [255, 150, 255, 255],
    [75, 0, 75, 255],
    [175, 0, 75, 255],
    [255, 200, 0, 255],
    [255, 120, 50, 255],
    [0, 175, 0, 255],
    [135, 60, 0, 255],
    [150, 240, 80, 255],
    [255, 240, 150, 255],
    [255, 0, 0, 255]]).astype(np.uint8)

  return colors
