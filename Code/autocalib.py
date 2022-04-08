import os
import argparse
from typing import List, Tuple
import numpy as np
import scipy.optimize as opt
from cv2 import cv2

def get_image_points(images: List[np.ndarray], h: int, w: int) -> List[np.ndarray]:
    images_copy = np.copy(images)
    image_points = []
    for image in images_copy:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_image, (w, h), None)
        if ret:
            corners = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            corners = corners.reshape(-1, 2)
            image_points.append(corners)
    return image_points

def display_image_points(images: List[np.ndarray], image_points: List[np.ndarray],
                         h: int, w: int, path: str) -> None:
    i = 0
    for points, image in zip(image_points, images):
        points = np.float32(points.reshape(-1, 1, 2))
        cv2.drawChessboardCorners(image, (w, h), points, True)
        image_resized = cv2.resize(image, (image.shape[1] // 3, image.shape[0] // 3))
        image_file = os.path.join(path, f'image_{i}.png')
        cv2.imwrite(image_file, image_resized)
        i += 1

def get_world_points(length: float, h: int, w: int) -> np.ndarray:
    y, x = np.indices((h, w))
    world_points = np.stack((x.ravel() * length, y.ravel() * length)).T
    return world_points

def get_homography(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    x1, y1 = points1[:, 0], points1[:, 1]
    x2, y2 = points2[:, 0], points2[:, 1]
    a_mat = []
    for i in range(points1.shape[0]):
        a_mat.append(np.array([x1[i], y1[i], 1, 0, 0, 0, -x1[i]*x2[i], -y1[i]*x2[i], -x2[i]]))
        a_mat.append(np.array([0, 0, 0, x1[i], y1[i], 1, -x1[i]*y2[i], -y1[i]*y2[i], -y2[i]]))
    a_mat = np.array(a_mat)
    _, _, v = np.linalg.svd(a_mat, full_matrices=True)
    homography = v[-1, :].reshape(3, 3)
    homography = homography / homography[2, 2]
    return homography

def get_all_homographies(image_points: List[np.ndarray], length: float,
                         h: int, w: int) -> List[np.ndarray]:
    world_points = get_world_points(length, h, w)
    homographies = []
    for image_points_i in image_points:
        homography = get_homography(world_points, image_points_i)
        homographies.append(homography)
    return homographies

def v_i_j(hi: np.ndarray, hj: np.ndarray) -> np.ndarray:
    return np.array([
                        hi[0] * hj[0],
                        hi[0] * hj[1] + hi[1] * hj[0],
                        hi[1] * hj[1],
                        hi[2] * hj[0] + hi[0] * hj[2],
                        hi[2] * hj[1] + hi[1] * hj[2],
                        hi[2] * hj[2]
                    ])

def get_v_mat(homographies: List[np.ndarray]) -> np.ndarray:
    v_mat = []
    for homography in homographies:
        h1, h2 = homography[:, 0], homography[:, 1]
        v11 = v_i_j(h1, h1)
        v12 = v_i_j(h1, h2)
        v22 = v_i_j(h2, h2)
        v_mat.append(v12.T)
        v_mat.append(np.subtract(v11, v22).T)
    return np.array(v_mat)

def get_b_mat(homographies: List[np.ndarray]) -> np.ndarray:
    v_mat = get_v_mat(homographies)
    _, _, v = np.linalg.svd(v_mat)
    b_mat = v[-1, :]
    b_mat_ = np.zeros((3, 3))
    b_mat_[0,0] = b_mat[0]
    b_mat_[0,1] = b_mat[1]
    b_mat_[0,2] = b_mat[3]
    b_mat_[1,0] = b_mat[1]
    b_mat_[1,1] = b_mat[2]
    b_mat_[1,2] = b_mat[4]
    b_mat_[2,0] = b_mat[3]
    b_mat_[2,1] = b_mat[4]
    b_mat_[2,2] = b_mat[5]
    return b_mat_

def get_intrinsics(b_mat: np.ndarray) -> np.ndarray:
    v_0 = (b_mat[0,1] * b_mat[0,2] - b_mat[0,0] * b_mat[1,2]) / (b_mat[0,0] * b_mat[1,1] - b_mat[0,1]**2)
    lam = b_mat[2,2] - (b_mat[0,2]**2 + v_0 * (b_mat[0,1] * b_mat[0,2] - b_mat[0,0] * b_mat[1,2])) / b_mat[0,0]
    alpha = np.sqrt(lam / b_mat[0,0])
    beta = np.sqrt(lam * (b_mat[0,0] / (b_mat[0,0] * b_mat[1,1] - b_mat[0,1]**2)))
    gamma = -(b_mat[0,1] * alpha**2 * beta) / lam 
    u_0 = (gamma * v_0 / beta) - (b_mat[0,2] * alpha ** 2 / lam)

    return np.array([
                        [alpha, gamma, u_0],
                        [0, beta, v_0],
                        [0, 0, 1]
                    ])

def get_extrinsics(intrinsics: np.ndarray, homographies: List[np.ndarray]) -> List[np.ndarray]:
    intrinsics_inverse = np.linalg.inv(intrinsics)
    extrinsics = []
    for homography in homographies:
        h1 = homography[:, 0]
        h2 = homography[:, 1]
        h3 = homography[:, 2]
        lam = np.linalg.norm(np.dot(intrinsics_inverse, h1), 2)
        r1 = np.dot(intrinsics_inverse, h1) / lam
        r2 = np.dot(intrinsics_inverse, h2) / lam
        r3 = np.cross(r1, r2)
        translation = np.dot(intrinsics_inverse, h3) / lam
        extrinsics_current = np.vstack((r1, r2, r3, translation)).T
        extrinsics.append(extrinsics_current)
    return extrinsics

def encode(intrinsics: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    alpha = intrinsics[0,0]
    gamma = intrinsics[0,1]
    beta = intrinsics[1,1]
    u_0 = intrinsics[0,2]
    v_0 = intrinsics[1,2]
    k1 = distortion[0]
    k2 = distortion[1]
    return np.array([alpha, gamma, beta, u_0, v_0, k1, k2])

def decode(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    alpha, gamma, beta, u_0, v_0, k1, k2 = params
    intrinsics = np.array([
                              [alpha, gamma, u_0],
                              [0, beta, v_0],
                              [0, 0, 1]
                          ]).reshape(3, 3)
    distortion = np.array([k1, k2]).reshape(2, 1)
    return intrinsics, distortion

def loss_function(params: np.ndarray, extrinsics: List[np.ndarray],
                  image_points: List[np.ndarray], world_points: np.ndarray) -> np.ndarray:
    intrinsics, _ = decode(params)
    _, _, _, u_0, v_0, k1, k2 = params
    errors = []
    
    for image_points_i, extrinsics_i in zip(image_points, extrinsics):
        transform_2d = np.array([extrinsics_i[:, 0], extrinsics_i[:, 1], extrinsics_i[:, 3]]).reshape(3, 3).T
        intrinsics_transformed = np.dot(intrinsics, transform_2d)

        current_error = 0
        for image_point, world_point in zip(image_points_i, world_points):
            world_points_2d_homo = np.array([world_point[0], world_point[1], 1]).reshape(3, 1)
            world_points_3d_homo = np.array([world_point[0], world_point[1], 0, 1]).reshape(4, 1)
            xyz = np.dot(extrinsics_i, world_points_3d_homo)
            x = xyz[0] / xyz[2]
            y = xyz[1] / xyz[2]
            radius = np.sqrt(x ** 2 + y ** 2)
            mij = np.array([image_point[0], image_point[1], 1], dtype='float').reshape(3, 1)
            uvw = np.dot(intrinsics_transformed, world_points_2d_homo)
            u = uvw[0] / uvw[2]
            v = uvw[1] / uvw[2]
            u_ = u + (u - u_0) * (k1 * radius ** 2 + k2 * radius ** 4)
            v_ = v + (v - v_0) * (k1 * radius ** 2 + k2 * radius ** 4)
            mij_ = np.array([u_, v_, 1], dtype='float').reshape(3, 1)
            error = np.linalg.norm(np.subtract(mij, mij_), ord=2)
            current_error += error
        errors.append(current_error / 54)
    return np.array(errors)

def reproject(intrinsics: np.ndarray, distortion: np.ndarray,
              extrinsics: List[np.ndarray], image_points: List[np.ndarray],
              world_points: np.ndarray) -> Tuple[float, List[List[List[float]]]]:
    _, _, _, u_0, v_0, k1, k2 = encode(intrinsics, distortion)
    errors = []
    reprojected_points = []

    for image_points_i, extrinsics_i in zip(image_points, extrinsics):
        transform_2d = np.array([extrinsics_i[:, 0], extrinsics_i[:, 1], extrinsics_i[:, 3]]).reshape(3, 3).T
        intrinsics_transformed = np.dot(intrinsics, transform_2d)

        current_error = 0
        points = []
        for image_point, world_point in zip(image_points_i, world_points):
            world_points_2d_homo = np.array([world_point[0], world_point[1], 1]).reshape(3, 1)
            world_points_3d_homo = np.array([world_point[0], world_point[1], 0, 1]).reshape(4, 1)
            xyz = np.dot(extrinsics_i, world_points_3d_homo)
            x = xyz[0] / xyz[2]
            y = xyz[1] / xyz[2]
            radius = np.sqrt(x ** 2 + y ** 2)
            mij = np.array([image_point[0], image_point[1], 1], dtype='float').reshape(3, 1)
            uvw = np.dot(intrinsics_transformed, world_points_2d_homo)
            u = uvw[0] / uvw[2]
            v = uvw[1] / uvw[2]
            u_ = u + (u - u_0) * (k1 * radius ** 2 + k2 * radius ** 4)
            v_ = v + (v - v_0) * (k1 * radius ** 2 + k2 * radius ** 4)
            points.append([u_, v_])
            mij_ = np.array([u_, v_, 1], dtype='float').reshape(3, 1)
            error = np.linalg.norm(np.subtract(mij, mij_), ord=2)
            current_error += error
        errors.append(current_error)
        reprojected_points.append(points)
    errors_np = np.array(errors)
    avg_error = np.sum(errors_np) / (len(image_points) * world_points.shape[0])
    return avg_error, reprojected_points

def draw_reprojected(images: List[np.ndarray], reprojected_points: List[List[List[float]]],
                     intrinsics: np.ndarray, distortion: np.ndarray, path: str) -> None:
    camera_matrix = np.array(intrinsics, np.float32).reshape(3, 3)
    distortion_mat = np.array([distortion[0], distortion[1], 0, 0], np.float32)
    i = 0
    for image, image_points in zip(images, reprojected_points):
        corrected_image = cv2.undistort(image, camera_matrix, distortion_mat)
        for image_point in image_points:
            xy = int(image_point[0]), int(image_point[1])
            corrected_image = cv2.circle(corrected_image, xy, 5, (0, 0, 255), 3)
        reprojected_image_file_name = os.path.join(path, f'image_{i}.png')
        cv2.imwrite(reprojected_image_file_name, corrected_image)
        i += 1

def read_images(path: str) -> List[np.ndarray]:
    images = []
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        image = cv2.imread(file_path)
        images.append(image)
    return images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--images_path', type=str,
                        default='../Data/Calibration_Imgs/',
                        help='The path where calibration images are stored.')
    parser.add_argument('-op', '--output_path', type=str,
                        default='../Data/Ouputs/',
                        help='The path where results are stored.')
    args = parser.parse_args()
    images_path = args.images_path
    output_path = args.output_path

    if not os.path.exists(images_path):
        raise ValueError(f'The path {images_path} does not exist!')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    images = read_images(images_path)
    h, w = 6, 9
    length = 12.5

    image_points = get_image_points(images, h, w)
    world_points = get_world_points(length, h, w)
    
    image_points_path = os.path.join(output_path, 'image_points')
    if not os.path.exists(image_points_path):
        os.makedirs(image_points_path)
    display_image_points(images, image_points, h, w, image_points_path)
    
    homographies = get_all_homographies(image_points, length, h, w)
    b_mat = get_b_mat(homographies)
    intrinsics = get_intrinsics(b_mat)
    extrinsics = get_extrinsics(intrinsics, homographies)
    distortion = np.array([0, 0]).reshape(2, 1)

    print('Before optimization...')
    print(intrinsics)
    print(distortion)

    print('Optimizing...')
    params = encode(intrinsics, distortion)
    optimized = opt.least_squares(fun=loss_function, x0=params, method='lm',
                                  args=[extrinsics, image_points, world_points])
    optimized_params = optimized.x
    optimized_intrinsics, optimized_distortion = decode(optimized_params)
    optimized_extrinsics = get_extrinsics(optimized_intrinsics, homographies)

    print('After optimization...')
    print(optimized_intrinsics)
    print(optimized_distortion)

    error_before, _ = reproject(intrinsics, distortion, extrinsics, image_points, world_points)
    error_after, reprojected_points = reproject(optimized_intrinsics, optimized_distortion,
                                                optimized_extrinsics, image_points, world_points)
    print(f'Error before optimization: {error_before}')
    print(f'Error after optimization: {error_after}')
    
    reprojected_images_path = os.path.join(output_path, 'reprojected')
    if not os.path.exists(reprojected_images_path):
        os.makedirs(reprojected_images_path)
    draw_reprojected(images, reprojected_points, optimized_intrinsics,
                     optimized_distortion, reprojected_images_path)
