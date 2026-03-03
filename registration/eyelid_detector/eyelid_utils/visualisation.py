import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
from .math_utils import *


__all__ = ['draw_lmk', 'draw_eye_lmk', 'draw_all_lmk', 'draw_heatmap', 'get_landmark_connectivity', 'plot_landmarks']


def draw_lmk(img, lmks):
    img_copy = img.copy()
    for lmk in lmks:
        x = int(lmk[0] + 0.5)
        y = int(lmk[1] + 0.5)
        cv2.circle(img_copy, (x, y), 1, (0, 255, 0), -1)
    return img_copy


def draw_eye_lmk(img, lmks, iris_end=38):
    img_copy = img.copy()
    iris_lmks = lmks[:iris_end]
    eyelid_lmks = lmks[iris_end:]
    for lmk in eyelid_lmks:
        x = int(lmk[0] + 0.5)
        y = int(lmk[1] + 0.5)
        cv2.circle(img_copy, (x, y), 1, (255, 0, 0), -1)

    # iris_lmks[:19] = ellipse_correction(iris_lmks[:19])
    # iris_lmks[19:] = ellipse_correction(iris_lmks[19:])
    
    for lmk in iris_lmks:
        x = int(lmk[0] + 0.5)
        y = int(lmk[1] + 0.5)
        cv2.circle(img_copy, (x, y), 1, (0, 255, 255), -1)
    return img_copy


def draw_all_lmk(img, lmks):
    img_copy = img.copy()
    face_lmks = lmks[:68]
    iris_lmks = lmks[68:68+38]
    eyelid_lmks = lmks[68+38:]
    for lmk in face_lmks:
        x = int(lmk[0] + 0.5)
        y = int(lmk[1] + 0.5)
        cv2.circle(img_copy, (x, y), 1, (0, 255, 0), -1)
    for lmk in eyelid_lmks:
        x = int(lmk[0] + 0.5)
        y = int(lmk[1] + 0.5)
        cv2.circle(img_copy, (x, y), 1, (255, 0, 0), -1)
    for lmk in iris_lmks:
        x = int(lmk[0] + 0.5)
        y = int(lmk[1] + 0.5)
        cv2.circle(img_copy, (x, y), 1, (0, 255, 255), -1)
    return img_copy


def draw_heatmap(img, heatmap, cmap='jet', alpha=0.5):
    scalar_map = ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1), cmap=plt.get_cmap(cmap))
    color_heatmap = scalar_map.to_rgba(heatmap)[:, :, :3] * 255
    if color_heatmap.shape[0] != img.shape[0]:
        color_heatmap = cv2.resize(color_heatmap, (img.shape[1], img.shape[0]))
    img_overlap = img * (1 - alpha) + color_heatmap * alpha
    
    return img_overlap.astype(np.uint8)


def get_landmark_connectivity(num_landmarks):
    if num_landmarks == 68:
        return ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
                (12, 13), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (20, 21), (22, 23), (23, 24),
                (24, 25), (25, 26), (27, 28), (28, 29), (29, 30), (30, 33), (31, 32), (32, 33), (33, 34), (34, 35),
                (36, 37), (37, 38), (38, 39), (40, 41), (41, 36), (42, 43), (43, 44), (44, 45), (45, 46), (46, 47),
                (47, 42), (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57),
                (57, 58), (58, 59), (59, 48), (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67),
                (67, 60), (39, 40))
    elif num_landmarks == 100:
        return ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
                (12, 13), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (20, 21), (22, 23), (23, 24),
                (24, 25), (25, 26), (68, 69), (69, 70), (70, 71), (72, 73), (73, 74), (74, 75), (36, 76), (76, 37),
                (37, 77), (77, 38), (38, 78), (78, 39), (39, 40), (40, 79), (79, 41), (41, 36), (42, 80), (80, 43),
                (43, 81), (81, 44), (44, 82), (82, 45), (45, 46), (46, 83), (83, 47), (47, 42), (27, 28), (28, 29),
                (29, 30), (30, 33), (31, 32), (32, 33), (33, 34), (34, 35), (84, 85), (86, 87), (48, 49), (49, 88),
                (88, 50), (50, 51), (51, 52), (52, 89), (89, 53), (53, 54), (54, 55), (55, 90), (90, 56), (56, 57),
                (57, 58), (58, 91), (91, 59), (59, 48), (60, 92), (92, 93), (93, 61), (61, 62), (62, 63), (63, 94),
                (94, 95), (95, 64), (64, 96), (96, 97), (97, 65), (65, 66), (66, 67), (67, 98), (98, 99), (99, 60),
                (17, 68), (21, 71), (22, 72), (26, 75))
    else:
        return None


def plot_landmarks(frame, landmarks, scores=None, threshold=0.2,
                   connection_colour=(0, 255, 0), landmark_colour=(0, 0, 255),
                   connection_thickness=1, landmark_radius=1, landmark_connectivity=None):
    num_landmarks = len(landmarks)
    if scores is None:
        scores = np.full((num_landmarks,), threshold + 1.0, dtype=float)
    if landmark_connectivity is None:
        landmark_connectivity = get_landmark_connectivity(len(landmarks))
    if landmark_connectivity is not None:
        for (idx1, idx2) in landmark_connectivity:
            if (idx1 < num_landmarks and idx2 < num_landmarks and
                    scores[idx1] >= threshold and scores[idx2] >= threshold):
                cv2.line(frame, tuple(landmarks[idx1].astype(int).tolist()),
                         tuple(landmarks[idx2].astype(int).tolist()),
                         color=connection_colour, thickness=connection_thickness,
                         lineType=cv2.LINE_AA)
    for landmark, score in zip(landmarks, scores):
        if score >= threshold:
            cv2.circle(frame, tuple(landmark.astype(int).tolist()), landmark_radius, landmark_colour, -1)
