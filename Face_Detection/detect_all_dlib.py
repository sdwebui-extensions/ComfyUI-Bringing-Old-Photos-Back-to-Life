# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import skimage.io as io

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from PIL import Image
import os
from skimage import img_as_ubyte
import argparse
import dlib


def _standard_face_pts():
    pts = (
        np.array([196.0, 226.0, 316.0, 226.0, 256.0, 286.0, 220.0, 360.4, 292.0, 360.4], np.float32) / 256.0
        - 1.0
    )

    return np.reshape(pts, (5, 2))


def _origin_face_pts():
    pts = np.array([196.0, 226.0, 316.0, 226.0, 256.0, 286.0, 220.0, 360.4, 292.0, 360.4], np.float32)

    return np.reshape(pts, (5, 2))


def get_landmark(face_landmarks, id):
    part = face_landmarks.part(id)
    x = part.x
    y = part.y

    return (x, y)


def search(face_landmarks):
    x1, y1 = get_landmark(face_landmarks, 36)
    x2, y2 = get_landmark(face_landmarks, 39)
    x3, y3 = get_landmark(face_landmarks, 42)
    x4, y4 = get_landmark(face_landmarks, 45)

    x_nose, y_nose = get_landmark(face_landmarks, 30)

    x_left_mouth, y_left_mouth = get_landmark(face_landmarks, 48)
    x_right_mouth, y_right_mouth = get_landmark(face_landmarks, 54)

    x_left_eye = int((x1 + x2) / 2)
    y_left_eye = int((y1 + y2) / 2)
    x_right_eye = int((x3 + x4) / 2)
    y_right_eye = int((y3 + y4) / 2)

    results = np.array(
        [
            [x_left_eye, y_left_eye],
            [x_right_eye, y_right_eye],
            [x_nose, y_nose],
            [x_left_mouth, y_left_mouth],
            [x_right_mouth, y_right_mouth],
        ]
    )

    return results


def compute_transformation_matrix(img, landmark, normalize, side_length, target_face_scale=1.0):
    std_pts = _standard_face_pts()  # [-1,1]
    target_pts = (std_pts * target_face_scale + 1) / 2 * side_length

    # print(target_pts)

    h, w, c = img.shape
    if normalize == True:
        landmark[:, 0] = landmark[:, 0] / h * 2 - 1.0
        landmark[:, 1] = landmark[:, 1] / w * 2 - 1.0

    # print(landmark)

    affine = SimilarityTransform()

    affine.estimate(target_pts, landmark)

    return affine.params


def show_detection(image, box, landmark):
    plt.imshow(image)
    print(box[2] - box[0])
    plt.gca().add_patch(
        Rectangle(
            (box[1], box[0]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor="r", facecolor="none"
        )
    )
    plt.scatter(landmark[0][0], landmark[0][1])
    plt.scatter(landmark[1][0], landmark[1][1])
    plt.scatter(landmark[2][0], landmark[2][1])
    plt.scatter(landmark[3][0], landmark[3][1])
    plt.scatter(landmark[4][0], landmark[4][1])
    plt.show()


def affine2theta(affine, input_w, input_h, target_w, target_h):
    # param = np.linalg.inv(affine)
    param = affine
    theta = np.zeros([2, 3])
    theta[0, 0] = param[0, 0] * input_h / target_h
    theta[0, 1] = param[0, 1] * input_w / target_h
    theta[0, 2] = (2 * param[0, 2] + param[0, 0] * input_h + param[0, 1] * input_w) / target_h - 1
    theta[1, 0] = param[1, 0] * input_h / target_w
    theta[1, 1] = param[1, 1] * input_w / target_w
    theta[1, 2] = (2 * param[1, 2] + param[1, 0] * input_h + param[1, 1] * input_w) / target_w - 1
    return theta


def get_face_landmarks(face_detector, landmark_locator, image: np.ndarray):
    faces = face_detector(image)
    face_landmarks = []
    for face in faces:
        face_landmarks.append(search(landmark_locator(image, face)))
    return face_landmarks # memory usage okay?


def get_aligned_faces(faces_landmarks, image: np.ndarray, side_length: int) -> np.ndarray:
    aligned_faces = []
    for face_landmarks in faces_landmarks:
        affine = compute_transformation_matrix(
            image, 
            face_landmarks, 
            False, 
            float(side_length), 
            target_face_scale=1.3, 
        )
        aligned_face = warp(
            image, 
            affine, 
            output_shape=(side_length, side_length, 3), 
        )
        aligned_faces.append(aligned_face)
    return aligned_faces


def get_aligned_faces_v1(face_detector, landmark_locator, image: Image.Image, side_length: int) -> np.ndarray:
    # extract faces
    image = np.array(image)
    faces = face_detector(image)

    aligned_faces = []
    for face in faces:
        # get face landmarks
        face_landmarks = landmark_locator(image, face)
        current_fl = search(face_landmarks)

        # align face
        affine = compute_transformation_matrix(
            image, 
            current_fl, 
            False, 
            float(side_length), 
            target_face_scale=1.3, 
        )
        aligned_face = warp(
            image, 
            affine, 
            output_shape=(side_length, side_length, 3), 
        )
        aligned_faces.append(aligned_face)

    return aligned_faces


def main(checkpoint_path: str, image_dir: str, output_dir: str, face_size: int):
    # make directories
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # load models
    face_detector = dlib.get_frontal_face_detector()
    landmark_locator = dlib.shape_predictor(checkpoint_path)

    for x in os.listdir(image_dir):
        # load image
        image_path = os.path.join(image_dir, x)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        # get aligned faces
        landmarks = get_face_landmarks(face_detector, landmark_locator, image)
        aligned_faces = get_aligned_faces(landmarks,  image, face_size)
        print(str(len(aligned_faces)) + " faces in " + x)

        # save faces
        for face_id, aligned_face in enumerate(aligned_faces):
            image_name = os.path.splitext(x)[0] + "_" + str(face_id + 1)
            io.imsave(os.path.join(output_dir, image_name + ".png"), img_as_ubyte(aligned_face))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_url", type=str, default="shape_predictor_68_face_landmarks.dat", help="shape predictor")
    parser.add_argument("--url", type=str, default="/home/jingliao/ziyuwan/celebrities", help="input")
    parser.add_argument("--save_url", type=str, default="/home/jingliao/ziyuwan/celebrities_detected_face_reid", help="output")
    parser.add_argument("--face_size", type=int, default=256, help="default=256, HR=512")
    opts = parser.parse_args()

    main(opts.model_url, opts.url, opts.save_url, opts.face_size)
