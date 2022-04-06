import argparse
import os
import numpy as np
from cv2 import cv2
from typing import List, Tuple

def read_images(path: str) -> List[np.ndarray]:
    images = []
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        image = cv2.imread(file_path)
        images.append(image)
    return images

def find_homographies(images: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    num_points_x = 9
    num_points_y = 6
    world_x, world_y = np.meshgrid(np.linspace(0, num_points_x - 1, num_points_x),
                                   np.linspace(0, num_points_y - 1, num_points_y))
    world_x = world_x.reshape(54, 1) * 21.5
    world_y = world_y.reshape(54, 1) * 21.5
    world_points = np.float32(np.hstack((world_x, world_y)))

    image_points = []
    homographies = []
    for image in images:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(image_gray, (num_points_x, num_points_y), None)
        if ret:
            corners = corners.reshape(-1, 2)
            corners = cv2.cornerSubPix(image_gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            image_points.append(corners)

            homography = cv2.findHomography(world_points, corners)[0]
            homographies.append(homography)
    return image_points, homographies

def find_intrinsics(homographies: List[np.ndarray]) -> np.ndarray:
    def v_i_j(homography: np.ndarray, i: int, j: int) -> np.ndarray:
        return np.array([
                            homography[0, i] * homography[0, j],
                            homography[0, i] * homography[1, j] + homography[1, i] * homography[0, j],
                            homography[1, i] * homography[1, j],
                            homography[2, i] * homography[0, j] + homography[0, i] * homography[2, j],
                            homography[2, i] * homography[1, j] + homography[1, i] * homography[2, j],
                            homography[2, i] * homography[2, j]
                        ])
    v = np.array([])
    for i, homography in enumerate(homographies):
        if i == 0:
            v = np.vstack((v_i_j(homography, 0, 1),
                           np.subtract(v_i_j(homography, 0, 0), v_i_j(homography, 1, 1))))
        else:
            v = np.vstack((v, v_i_j(homography, 0, 1)))
            v = np.vstack((v, np.subtract(v_i_j(homography, 0, 0), v_i_j(homography, 1, 1))))

    _, _, vh = np.linalg.svd(v)
    b = vh[-1]
    b11 = b[0]
    b12 = b[1]
    b22 = b[2]
    b13 = b[3]
    b23 = b[4]
    b33 = b[5]

    v_0 = (b12 * b13 - b11 * b23) / (b11 * b22 - b12 ** 2)
    lam = b33 - ((b13 ** 2 + v_0 * (b12 * b13 - b11 * b23)) / b11)
    alpha = np.sqrt(lam / b11)
    beta = np.sqrt(lam * (b11 / (b11 * b22 - b12 ** 2)))
    gamma = -b12 * alpha ** 2 * beta / lam
    u_0 = (gamma * v_0 / beta) - (b13 * alpha ** 2 / lam)

    return np.array([
                        [alpha, gamma, u_0],
                        [0, beta, v_0],
                        [0, 0, 1]
                    ])

def find_extrinsics(intrinsics: np.ndarray, homography: np.ndarray) -> np.ndarray:
    intrinsics_inverse = np.linalg.inv(intrinsics)
    lam = 1 / np.dot(intrinsics_inverse, homography[:, 0])
    r1 = lam * np.dot(intrinsics_inverse, homography[:, 0])
    r2 = lam * np.dot(intrinsics_inverse, homography[:, 1])
    r3 = np.cross(r1, r2)
    rotation = np.asarray([r1, r2, r3]).T
    translation = lam * np.dot(intrinsics_inverse, homography[:, 2])
    extrinsics = np.zeros((3, 4))
    extrinsics[:, :-1] = rotation
    extrinsics[:, -1] = translation
    return extrinsics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--images_path', type=str,
                        default='../Data/Calibration_Imgs/',
                        help='The path where calibration images are stored.')
    args = parser.parse_args()
    images_path = args.images_path

    if not os.path.exists(images_path):
        raise ValueError(f'The path {images_path} does not exist!')

    images = read_images(images_path)
    image_points, homographies = find_homographies(images)
    intrinsics = find_intrinsics(homographies)
    extrinsics = []
    for homography in homographies:
        extrinsics.append(find_extrinsics(intrinsics, homography))
