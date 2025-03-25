# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# Includes modifications proposed by Jeremy Fix
# from here: https://github.com/NVlabs/ffhq-dataset/pull/3


import os
import sys
import json
import argparse
import numpy as np
import multiprocessing
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

# Image processing libraries
import PIL
from PIL import Image, ImageFile

# Project-specific imports
from lib.preprocess import align_img

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True  # avoid "Decompressed Data Too Large" error


def save_detection_as_txt(dst, lm5p):
    outLand = open(dst, "w")
    for i in range(lm5p.shape[0]):
        outLand.write(str(float(lm5p[i][0])) + " " + str(float(lm5p[i][1])) + "\n")
    outLand.close()


def process_image(kwargs):
    """
    Processes an image by aligning and cropping it based on facial landmarks.

    Args:
        kwargs (dict): Dictionary containing the following keys:
            - src_dir (str): Directory containing the source image.
            - dst_dir (str): Directory to save the processed image.
            - lm5p (np.ndarray): Array of shape (N, 2) representing facial landmarks.
            - im_name (str): Name of the image file.
            - save_realign_dir (str or None): Directory to save realigned images.
            - save_detection_dir (str or None): Directory to save detection results.

    Returns:
        None
    """

    # Extract parameters from kwargs
    src_dir = kwargs['src_dir']
    dst_dir = kwargs['dst_dir']
    lm5p = kwargs['lm5p']
    lm3d = kwargs['lm3d']
    im_name = kwargs['im_name']
    save_realign_dir = kwargs.get('save_realign_dir', None)
    save_detection_dir = kwargs.get('save_detection_dir', None)
    save_align3d_dir = kwargs['save_align3d_dir']

    # Ensure the destination directory exists
    os.makedirs(dst_dir, exist_ok=True)

    # Construct file paths
    src_file = os.path.join(src_dir, im_name)

    # Ensure the source file exists before proceeding
    assert os.path.isfile(src_file), f"Source file not found: {src_file}"

    # Open the image
    img = Image.open(src_file)
    _, H = img.size  # Get image dimensions

    # Prepare alignment parameters
    params = {'name': src_file, 'lm': lm5p.tolist()}
    aligned_lm5p = lm5p.copy()
    aligned_lm3d = lm3d.copy()

    # Flip Y-coordinates to match the image coordinate system
    aligned_lm5p[:, -1] = H - 1 - aligned_lm5p[:, -1]
    aligned_lm3d[:, 1] = H - 1 - aligned_lm3d[:, 1]
    # Convert image name to PNG format
    im_name = im_name.rsplit('.', 1)[0] + '.png'
    dst_file = os.path.join(dst_dir, im_name)

    # Optionally save the realigned image
    if save_realign_dir:
        img.save(os.path.join(save_realign_dir, im_name))

    # Optionally save detected landmarks as a text file
    if save_detection_dir:
        save_detection_as_txt(
            os.path.join(save_detection_dir, im_name.replace('.png', '.txt')), aligned_lm5p
        )

    # Crop the image based on aligned landmarks
    img_cropped, crop_param, aligned_lm3d_save = crop_image(img, aligned_lm5p.copy(), aligned_lm3d.copy(), output_size=kwargs['output_size'])
    params['crop'] = crop_param
    aligned_lm3d_save = np.concatenate([aligned_lm3d_save[:, 0:1], 512 - aligned_lm3d_save[:, 1:2]], 1)
    np.save(os.path.join(save_align3d_dir,
                         im_name.replace(".png", ".npy").replace(".jpg", ".npy").replace(".jpeg", ".npy")),
            aligned_lm3d_save)
    # Save the cropped image
    img_cropped.save(dst_file)

def crop_image(im, lm, ldmk_3d, center_crop_size=700, rescale_factor=300,
               target_size=1024., output_size=512):
    """
    Crops and resizes an image based on facial landmarks.

    Args:
        im (PIL.Image.Image): Input image.
        lm (np.ndarray): Facial landmarks array of shape (N, 2).
        center_crop_size (int, optional): Size of the centered crop. Defaults to 700.
        rescale_factor (int, optional): Scaling factor for alignment. Defaults to 300.
        target_size (float, optional): Target size for transformation. Defaults to 1024.
        output_size (int, optional): Final resized output size. Defaults to 512.

    Returns:
        tuple:
            - im_cropped (PIL.Image.Image): The cropped and resized image.
            - crop_param (list): List of cropping parameters.
    """

    # Get image height
    _, H = im.size

    # Define a standardized 3D landmark set for alignment
    lm3D_std = np.array([
        [-0.31148657,  0.09036078,  0.13377953],  # Left eye corner
        [ 0.30979887,  0.08972035,  0.13179526],  # Right eye corner
        [ 0.0032535,  -0.24617933,  0.55244243],  # Nose tip
        [-0.25216928, -0.5813392,   0.22405732],  # Left mouth corner
        [ 0.2484662,  -0.5812824,   0.22235769],  # Right mouth corner
    ])

    # Adjust standard landmarks for better alignment
    lm3D_std[:, 2] += 0.4  # Adjust depth (Z-axis)
    lm3D_std[:, 1] += 0.1  # Adjust vertical position (Y-axis)

    # Align the image based on landmarks
    _, im_high, _, _, crop_left, crop_up, s, ldmk_3d_align = align_img(
        im, lm, lm3D_std, ldmk_3d, target_size=target_size, rescale_factor=rescale_factor, rescale_factor_3D=218
    )

    # Compute center crop coordinates
    left = int(im_high.size[0] / 2 - center_crop_size / 2)
    upper = int(im_high.size[1] / 2 - center_crop_size / 2)
    right = left + center_crop_size
    lower = upper + center_crop_size

    # Crop the image
    im_cropped = im_high.crop((left, upper, right, lower))

    # Resize the cropped image to the output size
    im_cropped = im_cropped.resize((output_size, output_size), resample=Image.LANCZOS)

    # Define cropping parameters for reference
    crop_param = [
        int(left), int(upper), int(center_crop_size),
        int(crop_left), int(crop_up), float(H * s), int(target_size)
    ]

    return im_cropped, crop_param, ldmk_3d_align


def process_video(kwargs):
    """
    Processes a video by aligning images based on facial landmarks.

    Args:
        kwargs (dict): Dictionary containing the following keys:
            - src_dir (str): Directory containing video frames.
            - dst_dir (str): Directory to save processed images.
            - lm5p (dict): Dictionary of image filenames and their corresponding 5-point landmarks.
            - im_names (list): List of image filenames.
            - output_size (int): Final output image resolution.
            - transform_size (int): Size used for transformations before cropping.
            - enable_padding (bool): Whether to apply padding.
            - enable_warping (bool): Whether to apply warping transformation.
            - save_realign_dir (str or None): Directory to save realigned images.
            - save_detection_dir (str or None): Directory to save detection results.
            - apply_GF (int): Gaussian filtering level for smoothing keypoints.

    Returns:
        None
    """

    # Extract parameters from kwargs
    video_dir = kwargs['src_dir']
    dst_dir = kwargs['dst_dir']
    lm5p_dict = kwargs['lm5p']
    lm3d_dict = kwargs['lm3d']
    output_size = kwargs['output_size']
    enable_padding = kwargs['enable_padding']
    enable_warping = kwargs['enable_warping']
    save_realign_dir = kwargs['save_realign_dir']
    save_detection_dir = kwargs['save_detection_dir']
    save_align3d_dir = kwargs['save_align3d_dir']
    apply_GF = kwargs['apply_GF']

    # Use landmark dictionary keys as image names
    im_names = list(lm5p_dict.keys())

    # Apply Gaussian filtering for smoother keypoint transitions (if enabled)
    if apply_GF > 0:
        im_names.sort(key=lambda x: int(x.split('.')[0]))  # Sort images by frame index
        kps_sequence = np.asarray([lm5p_dict[key] for key in im_names], dtype=np.float32)
        kps_sequence = gaussian_filter1d(kps_sequence, sigma=apply_GF, axis=0)  # Apply Gaussian smoothing
    else:
        kps_sequence = np.asarray([lm5p_dict[key] for key in im_names], dtype=np.float32)

    # Ensure number of images matches the number of keypoints
    assert len(im_names) == kps_sequence.shape[0], "Mismatch between image count and keypoint data."

    # Create directories for saving realigned images and detections (if specified)
    if save_realign_dir:
        os.makedirs(save_realign_dir, exist_ok=True)
    if save_detection_dir:
        os.makedirs(save_detection_dir, exist_ok=True)
    kps_sequence_3d = np.asarray([lm3d_dict[key] for key in im_names], dtype=np.float32)

    # Process each image in the video sequence
    for idx, im_name in enumerate(im_names):
        lm5p = kps_sequence[idx].reshape([-1, 2])  # Reshape keypoints to (N, 2) format
        lm3d = kps_sequence_3d[idx].reshape([-1, 3])
        # Prepare input dictionary for image processing
        input_data = {
            'src_dir': video_dir,
            'dst_dir': dst_dir,
            'im_name': im_name,
            'lm5p': lm5p,
            'lm3d': lm3d,
            'save_realign_dir': save_realign_dir,
            'save_detection_dir': save_detection_dir,
            'save_align3d_dir':save_align3d_dir,
            'output_size': output_size,
            'enable_padding': enable_padding,
            'enable_warping': enable_warping
        }

        # Process the image using the defined function
        process_image(input_data)

    # Create a 'finish' file to mark completion of processing
    with open(os.path.join(dst_dir, 'finish'), "w") as f:
        pass  # Creates an empty file


def recreate_aligned_images(
        root_dir, lms_root_dir, dst_dir,  valid_imgs_json,
        output_size=512, enable_padding=True, already_align=False
):
    """
    Recreates aligned images by applying facial landmark-based transformations.

    Args:
        root_dir (str): Directory containing original images.
        lms_root_dir (str): Directory containing facial landmark JSON files.
        dst_dir (str): Directory to save aligned images.
        save_realign_dir (str): Directory to save realigned images.
        valid_imgs_json (str): JSON file containing valid video names and image lists.
        output_size (int, optional): Final output image resolution. Defaults to 512.
        enable_padding (bool, optional): Whether to apply padding. Defaults to True.

    Returns:
        None
    """

    print("Recreating aligned images...")

    # Load valid video names and corresponding image lists from JSON file
    with open(valid_imgs_json, 'r') as f:
        valid_idx = json.load(f)

    inputs = []  # List to store image processing parameters

    # Iterate over each valid video
    for video_name, img_names in valid_idx:
        video_dir = os.path.join(root_dir, video_name)  # Path to video images
        dst_save_dir = os.path.join(dst_dir, video_name)  # Destination folder for aligned images
        base_dir = os.path.dirname(os.path.dirname(dst_dir))
        save_realign_dir = os.path.join(base_dir, 'realign', video_name)
        save_detection_dir = os.path.join(base_dir, 'realign_detections', video_name)
        save_align3d_dir = os.path.join(base_dir, 'align_3d_landmark', video_name)
        os.makedirs(save_align3d_dir, exist_ok=True)
        if save_realign_dir:
            os.makedirs(save_realign_dir, exist_ok=True)
            os.makedirs( save_detection_dir, exist_ok=True)

        # Skip processing if video directory does not exist
        # if not os.path.isdir(video_dir):
        #     continue

        # Load facial landmark data for this video

        lm5p_path = os.path.join(lms_root_dir, f"{video_name}.json")
        lm3d_path = os.path.join(lms_root_dir, f"{video_name}3d.json")

        with open(lm5p_path, 'r') as f:
            lm5p_dict = json.load(f)

        with open(lm3d_path, 'r') as f:
            lm3d_dict = json.load(f)

        # Iterate over images in the video
        for im_name in img_names:
            if im_name not in lm5p_dict:
                continue  # Skip if landmarks for this image are missing
            if im_name not in lm3d_dict:
                continue
            # Convert and reshape landmark points
            lm5p = np.asarray(lm5p_dict[im_name], dtype=np.float32).reshape([-1, 2])
            lm3d = np.asarray(lm3d_dict[im_name], dtype=np.float32).reshape([-1, 3])

            # Prepare input dictionary for processing
            input_data = {
                'src_dir': video_dir,
                'dst_dir': dst_save_dir,
                'im_name': im_name,
                'lm5p': lm5p,
                'lm3d': lm3d,
                'save_realign_dir': save_realign_dir,
                'save_detection_dir': save_detection_dir,
                'save_align3d_dir':save_align3d_dir,
                'output_size': output_size,
                'enable_padding': enable_padding
            }

            inputs.append(input_data)

        # break  # Stops after processing the first video (Is this intentional?)

    # Parallel Processing using multiprocessing (commented out for now)
    # with multiprocessing.Pool(n_threads) as pool:
    #     results = list(tqdm(pool.imap(process_image, inputs), total=len(inputs), smoothing=0.1))

    # Sequential processing (useful for debugging)
    if already_align:
        for input_data in tqdm(inputs, desc="Processing images"):
            src_dir = input_data['src_dir']
            dst_dir = input_data['dst_dir']
            im_name = input_data['im_name']
            lm5p = input_data['lm5p']
            save_realign_dir = input_data.get('save_realign_dir', None)
            save_detection_dir = input_data.get('save_detection_dir', None)
            save_align3d_dir = input_data['save_align3d_dir']
            # Ensure the destination directory exists
            os.makedirs(dst_dir, exist_ok=True)
            # Construct file paths
            src_file = os.path.join(src_dir, im_name)
            # Ensure the source file exists before proceeding
            assert os.path.isfile(src_file), f"Source file not found: {src_file}"
            # Open the image
            img = Image.open(src_file)
            _, H = img.size  # Get image dimensions
            im_name = im_name.rsplit('.', 1)[0] + '.png'
            dst_file = os.path.join(dst_dir, im_name)
            # Optionally save the realigned image
            if save_realign_dir:
                os.makedirs(save_realign_dir, exist_ok=True)
                img.save(os.path.join(save_realign_dir, im_name))
            aligned_lm5p = lm5p.copy()

            # Flip Y-coordinates to match the image coordinate system
            aligned_lm5p[:, -1] = H - 1 - aligned_lm5p[:, -1]
            # Optionally save detected landmarks as a text file
            if save_detection_dir:
                os.makedirs(save_detection_dir, exist_ok=True)

                save_detection_as_txt(
                    os.path.join(save_detection_dir, im_name.replace('.png', '.txt')), aligned_lm5p
                )
            # Save the cropped image
            img.save(dst_file)
            lm3d = input_data['lm3d'][:, 0:2]
            np.save(os.path.join(save_align3d_dir,
                                 im_name.replace(".png", ".npy").replace(".jpg", ".npy").replace(".jpeg", ".npy")),
                    lm3d)
    else:
        for input_data in tqdm(inputs, desc="Processing images"):
            process_image(input_data)


def recreate_aligned_videos_multiprocessing(
        root_dir, lms_root_dir, dst_dir, valid_video_json, save_realign=True, skip=True,
        enable_warping=False, output_size=512,
        enable_padding='zero_padding', n_threads=12, apply_GF=0
):
    """
    Recreates aligned videos by processing images with landmark-based transformations.

    Args:
        root_dir (str): Directory containing original video frames.
        lms_root_dir (str): Directory with corresponding facial landmark JSON files.
        dst_dir (str): Directory to save aligned images.
        valid_video_json (str): JSON file containing valid video names and frame lists.
        save_realign (bool, optional): Whether to save realigned images. Defaults to True.
        skip (bool, optional): Skip already processed videos if 'finish' file exists. Defaults to False.
        enable_warping (bool, optional): Apply warping transformation. Defaults to True.
        output_size (int, optional): Desired output image resolution. Defaults to 1024.
        transform_size (int, optional): Size used for transformation before cropping. Defaults to 4096.
        enable_padding (str, optional): Padding mode ('zero_padding', 'blur_padding', 'reflect_padding', or None). Defaults to None.
        n_threads (int, optional): Number of parallel threads for processing. Defaults to 12.
        apply_GF (int, optional): Gaussian filtering level. Defaults to 0.

    Returns:
        None
    """

    print("Recreating aligned images...")

    # Validate `enable_padding` argument
    assert enable_padding in [None, 'zero_padding', 'blur_padding', 'reflect_padding'], \
        f"Invalid enable_padding value: {enable_padding}"

    # Load valid video indices from JSON
    with open(valid_video_json, 'r') as f:
        valid_idx = json.load(f)

    inputs = []  # List to store parameters for multiprocessing

    # Iterate through each valid video and prepare processing inputs
    for video_name, im_names in valid_idx:
        video_dir = os.path.join(root_dir, video_name)  # Path to video frames
        dst_save_dir = os.path.join(dst_dir, video_name)  # Destination path for aligned images
        base_dir = os.path.dirname(os.path.dirname(dst_dir))

        save_align3d_dir = os.path.join(base_dir, 'align_3d_landmark', video_name)
        os.makedirs(save_align3d_dir, exist_ok=True)
        # Paths for saving realigned images and detections (if enabled)
        save_realign_dir = save_detection_dir = None
        if save_realign:
            save_realign_dir = os.path.join(base_dir, 'realign', video_name)
            save_detection_dir = os.path.join(base_dir, 'realign_detections', video_name)

        # Skip processing if video directory or landmark JSON does not exist
        if not os.path.isdir(video_dir):
            continue
        if not os.path.exists(os.path.join(lms_root_dir, f"{video_name}.json")):
            continue

        # Skip if already processed and `skip=True`
        if skip and os.path.exists(os.path.join(dst_save_dir, 'finish')):
            continue

            # Load facial landmark data
        with open(os.path.join(lms_root_dir, f"{video_name}.json"), 'r') as f:
            lm5p_dict = json.load(f)

        with open(os.path.join(lms_root_dir, f"{video_name}3d.json"), 'r') as f:
            lm3d_dict = json.load(f)
        # Prepare input dictionary for processing
        input_data = {
            'src_dir': video_dir,
            'dst_dir': dst_save_dir,
            'lm5p': lm5p_dict,
            'lm3d': lm3d_dict,
            'im_names': im_names,
            'save_realign_dir': save_realign_dir,
            'save_detection_dir': save_detection_dir,
            'save_align3d_dir':save_align3d_dir,
            'output_size': output_size,
            'enable_padding': enable_padding,
            'apply_GF': apply_GF,
            'enable_warping': enable_warping
        }
        inputs.append(input_data)

    # Process videos in parallel using multiprocessing
    with multiprocessing.Pool(n_threads) as pool:
        results = list(tqdm(pool.imap(process_video, inputs), total=len(inputs), smoothing=0.1))

    # Alternative: Process sequentially (useful for debugging)
    # for input_data in tqdm(inputs):
    #     process_video(input_data)


# # ----------------------------------------------------------------------------
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--source', type=str, default='.')
#     parser.add_argument('--lm_source', type=str, default='')
#     parser.add_argument('--dest', type=str, default='realign1500')
#     parser.add_argument('--valid_video_json', type=str, default=None)
#     parser.add_argument('--threads', type=int, default=12)
#     parser.add_argument('--output_size', type=int, default=768)
#     parser.add_argument('--transform_size', type=int, default=768)
#     parser.add_argument('--apply_GF', type=float, default=0)
#     parser.add_argument('--save_realign', action='store_true')
#     parser.add_argument('--skip', action='store_true')
#     parser.add_argument('--disable_warping', action='store_true')
#     parser.add_argument('--padding_mode', type=str, default=None)
#     args = parser.parse_args()
#
#     # recreate_aligned_images_fast(args.source, args.lm_source, args.dest, args.save_realign_dir, args.valid_video_json,
#     #                              output_size=args.output_size, transform_size=args.transform_size, n_threads=args.threads)
#     recreate_aligned_videos_fast(args.source, args.lm_source, args.dest, args.valid_video_json,
#                                  save_realign=args.save_realign, skip=args.skip,
#                                  output_size=args.output_size, transform_size=args.transform_size,
#                                  n_threads=args.threads, apply_GF=args.apply_GF,
#                                  enable_padding=args.padding_mode, enable_warping=False)
#
#     # run_cmdline(sys.argv)

# ----------------------------------------------------------------------------
