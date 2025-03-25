#################################################
# Copyright (c) 2021-present, xiaobing.ai, Inc. #
# All rights reserved.                          #
#################################################
# CV Research, DEV(USA) xiaobing.               #
# Written by wangduomin@xiaobing.ai             #
#################################################

##### Python internal and external packages
import numpy as np
import torch
import cv2


class FaceLdmkDetector:
    """
    A class for face detection and landmark detection (2D and 3D).
    """

    def __init__(self, facedetector, ldmkdetector, ldmk3ddetector):
        """
        Initialize the face landmark detector.

        Args:
            facedetector: Face detection model.
            ldmkdetector: 2D landmark detection model.
            ldmk3ddetector: 3D landmark detection model.
        """
        self.facedetector = facedetector
        self.ldmkdetector = ldmkdetector
        self.ldmk3ddetector = ldmk3ddetector

        self.last_frame = None
        self.frame_count = 0
        self.frame_margin = 15

    def reset(self):
        """Reset the last frame and frame count."""
        self.last_frame = None
        self.frame_count = 0

    def get_box_from_ldmk(self, ldmks):
        """
        Compute bounding boxes from landmark points.

        Args:
            ldmks (np.ndarray): Landmark points.

        Returns:
            np.ndarray: Bounding box coordinates.
        """
        boxes = [[np.min(ldmk[:, 0]), np.min(ldmk[:, 1]),
                  np.max(ldmk[:, 0]), np.max(ldmk[:, 1])]
                 for ldmk in ldmks]
        return np.array(boxes)

    def extend_box(self, boxes, ratio=1.5):
        """
        Extend bounding boxes by a given ratio.

        Args:
            boxes (np.ndarray): Bounding boxes.
            ratio (float): Scaling factor.

        Returns:
            np.ndarray: Extended bounding boxes.
        """
        extended_boxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            center = [(xmin + xmax) / 2, (ymin + ymax) / 2]
            size = np.sqrt((ymax - ymin + 1) * (xmax - xmin + 1))
            extend_size = size * ratio

            xmine, ymine = center[0] - extend_size / 2, center[1] - extend_size / 2
            xmaxe, ymaxe = center[0] + extend_size / 2, center[1] + extend_size / 2

            extended_boxes.append([xmine, ymine, xmaxe, ymaxe])

        return np.array(extended_boxes)

    def ldmk_detect(self, img, boxes):
        """
        Perform landmark detection on given bounding boxes.

        Args:
            img (np.ndarray): Input image.
            boxes (np.ndarray): Bounding boxes.

        Returns:
            np.ndarray: Smoothed landmark points.
        """
        ldmks = []
        h, w, _ = img.shape

        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box)
            img_crop = np.zeros([ymax - ymin, xmax - xmin, 3])

            img_crop[
            0 - min(ymin, 0):(ymax - ymin) - (max(ymax, h) - h),
            0 - min(xmin, 0):(xmax - xmin) - (max(xmax, w) - w),
            ] = img[max(ymin, 0):min(ymax, h), max(xmin, 0):min(xmax, w)]

            ldmk = self.ldmkdetector(img_crop)
            ldmk = ldmk[0] + np.array([xmin, ymin]).reshape(1, 2)
            ldmks.append(ldmk)

        ldmks = np.array(ldmks)

        # Apply smoothing to landmarks
        ldmks_smooth = [ldmks[idx] if idx == 0 or idx == len(ldmks) - 1
                        else ldmks[idx - 1:idx + 2].mean(0)
                        for idx in range(len(ldmks))]

        return np.array(ldmks_smooth)

    def inference(self, img):
        """
        Perform face detection and landmark detection.

        Args:
            img (np.ndarray): Input image.

        Returns:
            tuple: 2D landmarks, 3D landmarks, bounding boxes.
        """
        h, w, _ = img.shape

        if self.last_frame is None and self.frame_count % self.frame_margin == 0:
            boxes_scores = self.facedetector(img)
            if len(boxes_scores) == 0:
                return None

            boxes = boxes_scores[:, :4]
            scores = boxes_scores[:, 4]

            # Compute box sizes and distances from the center
            box_sizes = ((boxes[:, 3] - boxes[:, 1]) + (boxes[:, 2] - boxes[:, 0])) / 2
            box_distances = ((boxes[:, 3] + boxes[:, 1]) / 2 - 255) ** 2 + ((boxes[:, 2] + boxes[:, 0]) / 2 - 255) ** 2

            # Select the largest detected face
            index = np.argmax(box_sizes)
            boxes = boxes[index:index + 1]

            # Extend the bounding box
            boxes_extend = self.extend_box(boxes, ratio=1.5)

            self.frame_count += 1

        else:
            # Use last detected landmarks to determine the bounding box
            boxes = self.get_box_from_ldmk(self.last_frame)
            box_sizes = ((boxes[:, 3] - boxes[:, 1]) + (boxes[:, 2] - boxes[:, 0])) / 2
            box_centers = np.concatenate([((boxes[:, 2] + boxes[:, 0]) / 2),
                                          ((boxes[:, 3] + boxes[:, 1]) / 2)], axis=0)

            # Extend the bounding box
            boxes_extend = self.extend_box(boxes, ratio=1.8)

            self.frame_count += 1

        # Perform landmark detection
        ldmks = self.ldmk_detect(img, boxes_extend)
        ldmks_3d = self.ldmk3ddetector(img, boxes)

        if self.last_frame is None:
            boxes = self.get_box_from_ldmk(ldmks)

        self.last_frame = ldmks

        return ldmks, ldmks_3d, boxes