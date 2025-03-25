"""This script contains the image preprocessing code for Deep3DFaceRecon_pytorch."""

import numpy as np
from scipy.io import loadmat
from PIL import Image
import cv2
import os
from skimage import transform as trans
import torch
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def POS(xp, x):
    """
    Calculate translation and scale using least squares for image alignment.

    Args:
        xp (np.ndarray): Target points, shape (2, N).
        x (np.ndarray): Source points, shape (2, N).

    Returns:
        tuple: Translation vector (t) and scale factor (s).
    """
    npts = xp.shape[1]
    A = np.zeros([2 * npts, 8])

    A[0:2 * npts - 1:2, 0:3] = x.T
    A[0:2 * npts - 1:2, 3] = 1
    A[1:2 * npts:2, 4:7] = x.T
    A[1:2 * npts:2, 7] = 1

    b = xp.T.reshape([2 * npts, 1])
    k, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    R1, R2 = k[:3], k[4:7]
    sTx, sTy = k[3], k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
    t = np.array([sTx, sTy])

    return t, s


def BBRegression(points, params):
    """
    Perform bounding box regression for 68 landmark detection.

    Args:
        points (np.ndarray): Facial landmarks, shape (5, 2).
        params (dict): Regression parameters.

    Returns:
        np.ndarray: Bounding box [x, y, w, h].
    """
    w1, b1, w2, b2 = params['W1'], params['B1'], params['W2'], params['B2']
    data = points.reshape([5, 2])
    data_mean = np.mean(data, axis=0)

    data -= data_mean
    rms = np.sqrt(np.sum(data ** 2) / 5)
    data /= rms
    data = data.reshape([1, 10]).T

    inputs = np.matmul(w1, data) + b1
    inputs = 2 / (1 + np.exp(-2 * inputs)) - 1
    inputs = np.matmul(w2, inputs) + b2
    inputs = inputs.T

    x, y = inputs[:, 0] * rms + data_mean[0], inputs[:, 1] * rms + data_mean[1]
    w = (224 / inputs[:, 2]) * rms

    return np.array([x, y, w, w]).reshape([4])


def img_padding(img, box):
    """
    Pad image to avoid cropping issues.

    Args:
        img (np.ndarray): Input image.
        box (np.ndarray): Bounding box [x, y, w, h].

    Returns:
        tuple: Padded image, updated bounding box, success flag.
    """
    success = True
    bbox = box.copy()
    h, w = img.shape[:2]
    padded_img = np.zeros([2 * h, 2 * w, 3])

    padded_img[h // 2: h + h // 2, w // 2: w + w // 2] = img
    bbox[:2] += [w // 2, h // 2]

    if bbox[0] < 0 or bbox[1] < 0:
        success = False

    return padded_img, bbox, success


def crop(img, bbox):
    """
    Crop image based on bounding box.

    Args:
        img (np.ndarray): Input image.
        bbox (np.ndarray): Bounding box [x, y, w, h].

    Returns:
        tuple: Cropped image, scale factor.
    """
    padded_img, padded_bbox, flag = img_padding(img, bbox)
    if not flag:
        return padded_img, 0

    x, y, w, h = padded_bbox
    cropped_img = padded_img[y:y + h, x:x + w]
    cropped_img = cv2.resize(cropped_img.astype(np.uint8), (224, 224), interpolation=cv2.INTER_CUBIC)

    return cropped_img, 224 / w


def scale_trans(img, lm, t, s):
    """
    Apply scaling and translation to the image and landmarks.

    Args:
        img (np.ndarray): Input image.
        lm (np.ndarray): Landmarks.
        t (np.ndarray): Translation vector.
        s (float): Scale factor.

    Returns:
        tuple: Transformed image, inverse scale parameters.
    """
    img_h, img_w = img.shape[:2]
    M_s = np.array([[1, 0, -t[0] + img_w // 2 + 0.5], [0, 1, -img_h // 2 + t[1]]], dtype=np.float32)
    img = cv2.warpAffine(img, M_s, (img_w, img_h))

    w, h = int(img_w / s * 100), int(img_h / s * 100)
    img = cv2.resize(img, (w, h))

    lm = np.stack([lm[:, 0] - t[0] + img_w // 2, lm[:, 1] - t[1] + img_h // 2], axis=1) / s * 100
    bbox = [w // 2 - 112, h // 2 - 112, 224, 224]

    cropped_img, scale2 = crop(img, bbox)
    assert scale2 != 0

    t1 = np.array([bbox[0], bbox[1]])
    scale = s / 100
    t2 = np.array([t[0] - img_w / 2, t[1] - img_h / 2])

    return cropped_img, (scale / scale2, scale * t1 + t2)


def align_for_lm(img, five_points):
    """
    Align facial image using facial landmarks for landmark detection refinement.

    Args:
        img: Input facial image (numpy array)
        five_points: Facial landmark coordinates (5 points, 10 values)

    Returns:
        crop_img: Cropped and aligned facial image
        scale: Scaling factor applied during cropping
        bbox: Bounding box coordinates [x, y, width, height]

    Process:
        1. Predict optimal face bounding box using landmark regression
        2. Crop and align image based on predicted bounding box
    """
    # Reshape landmarks to 1x10 array (5 points x 2 coordinates)
    five_points = np.array(five_points).reshape([1, 10])

    # Load bounding box regressor parameters (MATLAB format)
    params = loadmat('util/BBRegressorParam_r.mat')  # Contains regression weights

    # Predict optimal face bounding box using regression model
    bbox = BBRegression(five_points, params)  # Returns [x, y, width, height]

    # Verify valid bounding box prediction
    assert bbox[2] != 0, "Invalid bounding box width (zero detected)"

    # Convert to integer coordinates for cropping
    bbox = np.round(bbox).astype(np.int32)

    # Crop image and get scaling factor
    crop_img, scale = crop(img, bbox)  # crop() should handle boundary checks

    return crop_img, scale, bbox


def resize_n_crop_img(img, lm, ldmk_3d, t, s, s_3d, target_size=224., mask=None):
    """
    Resize and center-crop image with corresponding landmark transformation

    Args:
        img: PIL.Image - Input image
        lm: np.array - Facial landmarks in original image coordinates [N, 2]
        t: tuple - (tx, ty) translation parameters
        s: float - Scaling factor
        target_size: float - Output image dimensions (square)
        mask: PIL.Image - Optional mask image

    Returns:
        img: PIL.Image - Processed image
        lm: np.array - Transformed landmarks [N, 2]
        mask: PIL.Image - Processed mask (or None)
        left: int - Left crop coordinate
        up: int - Top crop coordinate
    """
    # Original image dimensions
    w0, h0 = img.size

    # Calculate scaled dimensions
    w = (w0 * s).astype(np.int32)
    h = (h0 * s).astype(np.int32)

    w_3d = (w0 * s_3d).astype(np.int32)
    h_3d = (h0 * s_3d).astype(np.int32)

    # Calculate crop coordinates after scaling and translation
    # Horizontal crop window
    left = (w / 2 - target_size / 2 + (t[0] - w0 / 2) * s).astype(np.int32)
    right = left + target_size

    # Vertical crop window (note inverted Y-axis in images)
    up = (h / 2 - target_size / 2 + (h0 / 2 - t[1]) * s).astype(np.int32)
    below = up + target_size
    left = int(left)
    up = int(up)
    right = int(right)
    below = int(below)
    # Resize and crop main image
    img = img.resize((w, h), resample=Image.BICUBIC)
    img = img.crop((left, up, right, below))

    # Process mask if provided
    if mask is not None:
        mask = mask.resize((w, h), resample=Image.BICUBIC)
        mask = mask.crop((left, up, right, below))

    # Transform landmarks to cropped coordinates
    # 1. Adjust for translation and original image center
    # 2. Apply scaling
    # 3. Adjust for final crop offset
    lm = np.stack([lm[:, 0] - t[0] + w0 / 2,
                   lm[:, 1] - t[1] + h0 / 2], axis=1) * s
    crop_offset = np.array([(w / 2 - target_size / 2),
                            (h / 2 - target_size / 2)])
    lm = lm - crop_offset.reshape(1, 2)

    ldmk_3d = np.stack([ldmk_3d[:, 0] - t[0] + w0 / 2, ldmk_3d[:, 1] -
                        t[1] + h0 / 2], axis=1) * s_3d
    ldmk_3d = ldmk_3d - np.reshape(
        np.array([(w_3d / 2 - 512 / 2), (h_3d / 2 - 512 / 2)]), [1, 2])

    return img, lm, mask, left, up, ldmk_3d


def extract_5p(lm):
    """
    Extract 5-point facial landmarks from 68 landmarks.

    Args:
        lm (np.ndarray): 68 facial landmarks.

    Returns:
        np.ndarray: 5-point landmarks.
    """
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack([
        lm[lm_idx[0], :],
        np.mean(lm[lm_idx[[1, 2]], :], axis=0),
        np.mean(lm[lm_idx[[3, 4]], :], axis=0),
        lm[lm_idx[5], :],
        lm[lm_idx[6], :]
    ], axis=0)

    return lm5p[[1, 2, 0, 3, 4], :]


def align_img(img, lm, lm3D, ldmk_3d, mask=None, target_size=224., rescale_factor=102., rescale_factor_3D=218.):
    """
    Align facial image using 2D-3D landmark correspondence

    Args:
        img: PIL.Image - Input facial image (H, W, 3)
        lm: np.array - Facial landmarks (68, 2) in image coordinates (y-axis inverted)
        lm3D: np.array - 3D reference landmarks (5, 3) for pose estimation
        mask: PIL.Image - Optional facial mask (H, W, 3)
        target_size: float - Output image dimensions (square)
        rescale_factor: float - Normalization factor for face scale

    Returns:
        trans_params: np.array - [raw_W, raw_H, scale, tx, ty] transformation parameters
        img_new: PIL.Image - Aligned image (target_size, target_size, 3)
        lm_new: np.array - Transformed landmarks (68, 2)
        mask_new: PIL.Image - Aligned mask (target_size, target_size)
        crop_left: int - Left crop coordinate
        crop_up: int - Top crop coordinate
        s: float - Final scaling factor

    Process:
        1. Extract 5-point landmarks if needed
        2. Estimate face scale and translation using POS algorithm
        3. Resize and crop image with landmark adjustment
    """
    # Original image dimensions
    w0, h0 = img.size

    # Extract 5 facial landmarks if not provided
    if lm.shape[0] != 5:
        lm5p = extract_5p(lm)  # Convert 68-point to 5-point landmarks
    else:
        lm5p = lm

    # Calculate scale and translation using PnP algorithm
    # POS (Perspective-n-Point algorithm) implementation
    t, s = POS(lm5p.T, lm3D.T)  # Returns translation vector and scale factor
    s_3d = rescale_factor_3D / s
    s = rescale_factor / s  # Normalize scale using reference face size
    # Apply geometric transformation
    img_new, lm_new, mask_new, crop_left, crop_up, ldmk_3d_align = resize_n_crop_img(
        img,
        lm,
        ldmk_3d,
        t,
        s,
        s_3d=s_3d,
        target_size=target_size,
        mask=mask
    )

    # Package transformation parameters [original_w, original_h, scale, tx, ty]
    trans_params = np.array([w0, h0, s, t[0][0], t[1][0]])

    return trans_params, img_new, lm_new, mask_new, crop_left, crop_up, s, ldmk_3d_align


def estimate_norm(lm_68p, H):
    """
    Estimate similarity transformation matrix for face alignment.

    Args:
        lm_68p (np.ndarray): 68 facial landmarks.
        H (int): Image height.

    Returns:
        np.ndarray: Transformation matrix (2, 3).
    """
    lm = extract_5p(lm_68p)
    lm[:, -1] = H - 1 - lm[:, -1]

    tform = trans.SimilarityTransform()
    src = np.array([
        [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
        [41.5493, 92.3655], [70.7299, 92.2041]
    ], dtype=np.float32)

    tform.estimate(lm, src)
    M = tform.params

    return M[0:2, :] if np.linalg.det(M) != 0 else np.eye(2, 3)


def estimate_norm_torch(lm_68p, H):
    """
    Estimate similarity transformation matrix for face alignment using PyTorch.

    Args:
        lm_68p (torch.Tensor): 68 facial landmarks.
        H (int): Image height.

    Returns:
        torch.Tensor: Transformation matrices.
    """
    lm_68p_ = lm_68p.detach().cpu().numpy()
    M = [estimate_norm(lm, H) for lm in lm_68p_]

    return torch.tensor(np.array(M), dtype=torch.float32, device=lm_68p.device)
