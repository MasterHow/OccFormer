import numpy as np
import cv2
import json
import open3d as o3d
import math
import os

class ocam_model:
    def __init__(self):
        self.pol = []
        self.length_pol = 0
        self.invpol = []
        self.length_invpol = 0
        self.xc = 0.0
        self.yc = 0.0
        self.c = 0.0
        self.d = 0.0
        self.e = 0.0
        self.width = 0
        self.height = 0

def get_ocam_model(myocam_model, filename):
    with open(filename) as f:
        f.readline()
        f.readline()
        poly = f.readline().split()
        myocam_model.length_pol = int(poly[0])
        myocam_model.pol = [float(pp) for pp in poly[1:]]

        f.readline()
        f.readline()
        f.readline()
        invpoly = f.readline().split()
        myocam_model.length_invpol = int(invpoly[0])
        myocam_model.invpol = [float(pp) for pp in invpoly[1:]]

        f.readline()
        f.readline()
        f.readline()
        cent = f.readline().split()
        myocam_model.xc = float(cent[0])
        myocam_model.yc = float(cent[1])

        f.readline()
        f.readline()
        f.readline()
        aff = f.readline().split()
        myocam_model.c = float(aff[0])
        myocam_model.d = float(aff[1])
        myocam_model.e = float(aff[2])

        f.readline()
        f.readline()
        f.readline()
        size = f.readline().split()
        myocam_model.height = int(size[0])
        myocam_model.width = int(size[1])


def world2cam(point2D, point3D, myocam_model):
    xc = myocam_model.xc
    yc = myocam_model.yc
    c = myocam_model.c
    d = myocam_model.d
    e = myocam_model.e
    length_invpol = myocam_model.length_invpol
    norm = math.sqrt(point3D[0]*point3D[0] + point3D[1]*point3D[1])
    theta = math.atan(point3D[2]/norm)
    if norm != 0:
        invnorm = 1/norm
        t = theta
        rho = myocam_model.invpol[0]
        t_i = 1
        for i in range(1,length_invpol):
            t_i *= t
            rho += t_i*myocam_model.invpol[i]
        x = point3D[0]*invnorm*rho
        y = point3D[1]*invnorm*rho
        point2D[0] = x*c + y*d + xc
        point2D[1] = x*e + y + yc
    else:
        point2D[0] = xc
        point2D[1] = yc

def world2cam_LiDAR(point2D, point3D, myocam_model):
    xc = myocam_model.yc
    yc = myocam_model.xc
    c = myocam_model.c
    d = myocam_model.d
    e = myocam_model.e
    length_invpol = myocam_model.length_invpol

    norm = np.sqrt(point3D[0] ** 2 + point3D[1] ** 2)
    theta = np.arctan2(-point3D[2], norm)
    rho = 0.0
    theta_i = 1.0

    for i in range(length_invpol):
        rho += myocam_model.invpol[i] * theta_i
        theta_i *= theta

    invNorm = 1.0 / norm if norm != 0 else 0
    x = point3D[0] * invNorm * rho
    y = point3D[1] * invNorm * rho

    point2D[0] = x * c + y * d + xc
    point2D[1] = x * e + y + yc

def project_lidar_to_image(lidar_points, R, T, mapx, mapy, myocam_model, unfold_img_h, unfold_img_w):
    projected_points = []
    corresponding_lidar_points = []

    for point in lidar_points:
        point = np.array(point)
        point_cam = R @ point + T
        point2D = np.zeros(2)
        world2cam_LiDAR(point2D, point_cam, myocam_model)

        u, v = int(point2D[0]), int(point2D[1])

        if 0 <= u < myocam_model.width and 0 <= v < myocam_model.height:
            unfolded_u = mapx[v, u]
            unfolded_v = mapy[v, u]
            # if 0 <= unfolded_u < mapx.shape[1] and 0 <= unfolded_v < mapx.shape[0]:     # 这里有bug 应该和展开图像的长宽比较
            if 0 <= unfolded_u < unfold_img_w and 0 <= unfolded_v < unfold_img_h:
                projected_points.append((int(unfolded_u), int(unfolded_v)))
                corresponding_lidar_points.append(point)

    return projected_points, corresponding_lidar_points


def load_semantic_masks(mask_path):
    data = np.load(mask_path)
    masks = data['masks']
    return masks

def load_detection_info(json_path):
    with open(json_path, 'r') as f:
        detection_info = json.load(f)
    return detection_info


def assign_semantics_to_lidar(lidar_points, projected_points, masks, class_ids):
    point_semantics = []

    h, w = masks.shape[:2]

    for point, (u, v) in zip(lidar_points, projected_points):
        if np.all(point == 0):  # 跳过坐标为 (0, 0, 0) 的点
            # print(point)
            continue
        if 0 <= u < w and 0 <= v < h:
            for mask_idx, mask in enumerate(masks.transpose(2, 0, 1)):
                if mask[int(v), int(u)] > 0:
                    semantic_label = class_ids[mask_idx]
                    point_semantics.append((point, semantic_label))
                    break

    return point_semantics



def create_panoramic_lonlat(mapx, mapy, myocam_model, angle):
    width = mapx.shape[1]
    height = mapx.shape[0]

    for i in range(height):
        for j in range(width):
            lat = (-angle[0] / 180 + i / height * (angle[0] + angle[1]) / 180) * np.pi
            lon = -(-angle[2] / 180 + j / width * (angle[2] + angle[3]) / 180) * np.pi + np.pi
            point3 = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
            point2 = np.zeros(2)
            world2cam(point2, point3, myocam_model)
            mapx[i, j] = point2[1]
            mapy[i, j] = point2[0]

def get_lonlat_map(result_txt, fov, angle):
    o_cata = ocam_model()
    get_ocam_model(o_cata, result_txt)
    pixel_h = o_cata.height * (fov[1] - fov[0]) / 2 / fov[1]
    size_pan_img = [round(pixel_h * (angle[0] + angle[1]) / (fov[1] - fov[0])),
                    round(pixel_h * (angle[2] + angle[3]) / (fov[1] - fov[0]))]
    mapx_pan = np.zeros([size_pan_img[0], size_pan_img[1]], np.float32)
    mapy_pan = np.zeros([size_pan_img[0], size_pan_img[1]], np.float32)
    create_panoramic_lonlat(mapx_pan, mapy_pan, o_cata, angle)
    return mapx_pan, mapy_pan

def ocam_lonlat(img, mapx_pan, mapy_pan):
    dst_pan = cv2.remap(img, mapx_pan, mapy_pan, cv2.INTER_LINEAR)
    dst_pan = cv2.flip(dst_pan, 1)
    return dst_pan


def compute_inverse_map(map_x, map_y, myocam_model):
    h, w = myocam_model.height, myocam_model.width
    inverse_map_x = np.zeros((h, w), dtype=np.float32)
    inverse_map_y = np.zeros((h, w), dtype=np.float32)

    # 对 mapx,y 进行水平翻转
    map_x = np.flip(map_x, axis=1)
    map_y = np.flip(map_y, axis=1)

    for i in range(map_x.shape[0]):
        for j in range(map_x.shape[1]):
            src_x = map_x[i, j]
            src_y = map_y[i, j]
            if 0 <= src_x < w and 0 <= src_y < h:
                inverse_map_x[int(src_y), int(src_x)] = j
                inverse_map_y[int(src_y), int(src_x)] = i

    # Create a mask for holes (where the value is zero)
    mask_x = (inverse_map_x == 0).astype(np.uint8)
    mask_y = (inverse_map_y == 0).astype(np.uint8)

    # Inpaint the holes
    inverse_map_x = cv2.inpaint(inverse_map_x, mask_x, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
    inverse_map_y = cv2.inpaint(inverse_map_y, mask_y, inpaintRadius=1, flags=cv2.INPAINT_TELEA)

    return inverse_map_x, inverse_map_y


def save_maps(mapx, mapy, inv_mapx, inv_mapy, filename):
    np.savez(filename, mapx=mapx, mapy=mapy, inv_mapx=inv_mapx, inv_mapy=inv_mapy)


def load_maps(filename):
    data = np.load(filename)
    return data['mapx'], data['mapy'], data['inv_mapx'], data['inv_mapy']


def main():
    result_txt = "PAL_intrinsic_calib_results.txt"
    lidar_points_file = "./assets/project_semantic/000000.pcd"
    json_file = "./assets/project_semantic/semantic_000000.json"
    mask_file = "./assets/project_semantic/masks_000000.npz"
    map_file = "maps.npz"

    # R_lidar_to_camera = np.array([
    #     [0.965926, 0, 0.258819],
    #     [0, 1, 0],
    #     [-0.258819, 0, 0.965926]
    # ])
    # adjusted
    R_lidar_to_camera = np.array([
        [0.965595, 0.02617695, 0.25873031],
        [-0.02528499, 0.99965732, -0.00677509],
        [-0.258819, 0., 0.965926]
    ])
    T_lidar_to_camera = np.array([0.08, 0, -0.02])

    pcd = o3d.io.read_point_cloud(lidar_points_file)
    lidar_points = np.asarray(pcd.points)

    o_cata = ocam_model()
    get_ocam_model(o_cata, result_txt)

    angle = [50, 10, 180, 180]

    if os.path.exists(map_file):
        mapx, mapy, inv_mapx, inv_mapy = load_maps(map_file)
    else:
        mapx, mapy = get_lonlat_map(result_txt, [0, 90], angle)
        inv_mapx, inv_mapy = compute_inverse_map(mapx, mapy, o_cata)
        save_maps(mapx, mapy, inv_mapx, inv_mapy, map_file)

    # mapx, mapy = get_lonlat_map(result_txt, [0, 90], angle)
    # inv_mapx, inv_mapy = compute_inverse_map(mapx, mapy, o_cata)
    projected_points, corresponding_lidar_points = project_lidar_to_image(lidar_points, R_lidar_to_camera,
                                                                          T_lidar_to_camera, inv_mapx, inv_mapy, o_cata)

    masks = load_semantic_masks(mask_file)
    detection_info = load_detection_info(json_file)
    class_ids = detection_info['class_ids']

    point_semantics = assign_semantics_to_lidar(corresponding_lidar_points, projected_points, masks, class_ids)

    with open("point_semantics.txt", "w") as f:
        for point, label in point_semantics:
            f.write(f"{point[0]} {point[1]} {point[2]} {label}\n")

    print(f"Saved semantic information of lidar points to point_semantics.txt")

if __name__ == '__main__':
    main()
