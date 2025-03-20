import os
import sys
import json
import time
import glob
import logging
import argparse
from datetime import datetime

import cv2
import torch
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from PIL import Image
from natsort import ns, natsorted
from lib.config.config import cfg
from lib.faceverse_process.fit_faceverse import fit_faceverse
from lib.face_detect_ldmk_pipeline import FaceLdmkDetector
from lib.model_builder import make_model
from lib.align_in_the_wild import recreate_aligned_images, recreate_aligned_videos_multiprocessing
from lib.preprocess_faceverse import make_cam_dataset_FFHQ, render_orth_mp
from lib.pdfgc.encoder import FanEncoder
from lib.pdfgc.utils import get_motion_feature
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def is_image_file(filename):
    """
    Check if a file has a common image format extension.

    Args:
        filename (str): The name or path of the file.

    Returns:
        bool: True if the file has an image extension (.jpg, .png, etc.), otherwise False.

    """
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    return os.path.splitext(filename)[1].lower() in image_extensions


def get_images_in_folder(folder_path):
    """
    Check if a given folder contains images and return a list of image file names.

    Args:
        folder_path (str): The path of the folder to check.

    Returns:
        list: A list of image file names if found, otherwise an empty list.

    """
    if not os.path.isdir(folder_path):  # Check if the path is a directory
        return []

    image_files = [file for file in os.listdir(folder_path) if
                   os.path.isfile(os.path.join(folder_path, file)) and is_image_file(file)]

    return image_files  # Return a list of image names


def is_video_file(file_path):
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv" }
    ext = os.path.splitext(file_path)[1].lower()
    return ext in video_extensions


def extract_imgs(input_path, save_dir, skip=1, center_crop=False, res=512, is_video=False, is_img=True):
    os.makedirs(save_dir, exist_ok=True)
    if is_video and is_video_file(input_path):
        videoCapture = cv2.VideoCapture(input_path)
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if center_crop:
            length = min(size) // 2
            top, bottom, left, right = max(0, size[1] // 2 - length), min(size[1], size[1] // 2 + length), max(0, size[
                0] // 2 - length), min(size[0], size[0] // 2 + length)
        else:
            length = max(size) // 2
            top, bottom, left, right = max(0, length - size[1] // 2), max(0, length - size[1] // 2), max(0,
                                                                                                         length - size[
                                                                                                             0] // 2), max(
                0, length - size[0] // 2)
        count = -1
        while True:
            flag, frame = videoCapture.read()
            count += 1
            if not flag:
                break
            if skip > 1 and not (count % skip == 0):
                continue
            if center_crop:
                crop_frame = frame[top: bottom, left: right]
            else:
                crop_frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            if not res == crop_frame.shape[0]:
                crop_frame = cv2.resize(crop_frame, dsize=(res, res), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(save_dir, str(count) + '.png'), crop_frame)
        videoCapture.release()
        logging.info(f"Video frames saved in {save_dir}")

    elif is_img:
        all_imgs = get_images_in_folder(input_path)
        if len(all_imgs) == 0:
            raise ValueError("The input file has no images")
        else:
            count = -1
            for image_name in all_imgs:
                count += 1
                img = cv2.imread(os.path.join(input_path, image_name))
                size = (img.shape[1], img.shape[0])
                if center_crop:
                    length = min(size) // 2
                    top, bottom, left, right = max(0, size[1] // 2 - length), min(size[1], size[1] // 2 + length), max(
                        0, size[0] // 2 - length), min(size[0], size[0] // 2 + length)
                else:
                    length = max(size) // 2
                    top, bottom, left, right = max(0, length - size[1] // 2), max(0, length - size[1] // 2), max(0,
                                                                                                                 length -
                                                                                                                 size[
                                                                                                                     0] // 2), max(
                        0, length - size[0] // 2)
                if center_crop:
                    crop_frame = img[top: bottom, left: right]
                else:
                    crop_frame = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
                if not res == crop_frame.shape[0]:
                    crop_frame = cv2.resize(crop_frame, dsize=(res, res), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(save_dir, image_name.split('.')[0].replace(" ", "_") + '.png'), crop_frame)
            logging.info(f"Images saved in {save_dir}")
    else:
        raise ValueError("The input file is not a video")


def get_valid_input_idx(input_path, save_dir, is_video, is_img, skip=1):
    """
    Extracts images from the input file, organizes them, and creates a JSON file listing valid images.

    Args:
        input_path (str): Path to the input file (e.g., a video or archive).
        save_dir (str): Directory to save extracted images.
        skip (int, optional): Step size for selecting images (default: 1).

    Returns:
        str: Path to the generated JSON file containing valid image names.
    """
    # Extract images to a subdirectory named after the input file
    raw_imgs_save_dir = os.path.join(save_dir, os.path.splitext(os.path.basename(input_path))[0])
    extract_imgs(input_path, raw_imgs_save_dir, is_video=is_video, is_img=is_img)

    # Define JSON save path
    valid_imgs_json_save_path = os.path.join(save_dir, 'valid_imgs.json')
    valid_videos, count, img_nums = [], 0, [0]

    # Process subdirectories in `save_dir`
    for video_imgs_name in tqdm(os.listdir(save_dir)):
        print(video_imgs_name)
        video_imgs_dir = os.path.join(save_dir, video_imgs_name)
        if not os.path.isdir(video_imgs_dir):
            continue

        # Collect and sort image files
        img_names = natsorted([x for x in os.listdir(video_imgs_dir) if x.endswith((".jpg", ".png", ".webp"))], alg=ns.PATH)

        # Store selected images with skip interval
        valid_videos.append([video_imgs_name, img_names[::skip]])
        count += len(valid_videos[-1][1])
        img_nums.append(count)

    # Save results to JSON
    with open(valid_imgs_json_save_path, 'w') as f:
        json.dump(valid_videos, f, indent=4)

    return valid_imgs_json_save_path  # Return JSON file path


def make_coeff_dataset_FFHQ(tracking_dir, save_dir, smooth=False, is_img=False):
    """
    Processes and organizes FaceVerse coefficients into a structured dataset.

    Parameters:
        tracking_dir (str): Source directory containing tracked coefficient sequences
        save_dir (str): Target directory to save processed coefficients
        smooth (bool): Apply temporal smoothing to coefficients when True
    """

    # Iterate through each sequence directory
    for prefix in tqdm(os.listdir(tracking_dir)):
        seq_path = os.path.join(tracking_dir, prefix)

        # Skip non-directory entries
        if not os.path.isdir(seq_path):
            continue

        # Collect valid frame directories containing 'finish' flag file
        frame_dirs = [
            name for name in os.listdir(seq_path)
            if os.path.exists(os.path.join(seq_path, name, 'finish'))
        ]

        # Sort frames numerically (requires directory names to be integer strings)
        if not is_img:
            frame_dirs.sort(key=lambda x: int(x))

        try:
            # Load all coefficient sequences for this sequence
            coeff_seq = np.stack([
                np.load(os.path.join(seq_path, fname, 'coeffs.npy'))
                for fname in frame_dirs
            ], axis=0)  # Shape: [num_frames, num_coeffs]

            # Apply temporal smoothing if enabled
            if smooth:
                # Gaussian filter with σ=0.5 across time dimension
                coeff_seq = gaussian_filter1d(coeff_seq, sigma=0.5, axis=0)

            # Create output directory structure
            output_seq_dir = os.path.join(save_dir, prefix)
            os.makedirs(output_seq_dir, exist_ok=True)

            # Save processed coefficients per frame
            for idx, fname in enumerate(frame_dirs):
                output_path = os.path.join(output_seq_dir, fname + '.npy')
                np.save(output_path, coeff_seq[idx])

        except Exception as e:
            # Note: Consider adding error logging in production code
            # print(f"Skipping sequence {prefix} due to error: {str(e)}")
            continue


class Process(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.net_fd, self.net_ldmk, self.net_ldmk_3d = make_model(self.cfg)
        self.net_fd.cuda()
        self.net_ldmk.cuda()
        self.net_ldmk_3d.cuda()
        pd_fgc = FanEncoder()
        weight_dict = torch.load(cfg.pdfgc_path)
        pd_fgc.load_state_dict(weight_dict, strict=False)
        pd_fgc = pd_fgc.eval().cuda()
        self.pd_fgc = pd_fgc
        ### set eval and freeze models
        self.net_fd.eval()
        self.net_ldmk.eval()
        self.net_ldmk_3d.eval()
        self.stand_index = np.array([96, 97, 54, 76, 82])
        self.fd_ldmk_detector = FaceLdmkDetector(self.net_fd, self.net_ldmk, self.net_ldmk_3d)

    def get_faceverse_labels_FFHQ(self, tracking_dir, root_dir, fv2fl_T_path='lib/FaceVerse/v3/fv2fl_30.npy',
                                  focal=4.2647, need_render=False,
                                  save_uv=True, save_mesh=False, save_name=None, render_normal_uv=False,
                                  num_thread=1, use_smooth=False, test_data=False, skip=False, is_img=False):
        """
        Processes FaceVerse tracking data to generate dataset labels for FFHQ-style datasets.

        Parameters:
            tracking_dir (str): Path to directory containing FaceVerse tracking data
            root_dir (str): Root directory for dataset outputs
            fv2fl_T_path (str): Path to FaceVerse-to-FLAME transformation matrix
            focal (float): Camera focal length
            need_render (bool): Whether to render visualizations
            save_uv (bool): Save UV texture maps if True
            save_mesh (bool): Save 3D meshes if True
            save_name (str): Custom name for output directory
            render_normal_uv (bool): Render normal maps in UV space if True
            num_thread (int): Number of parallel processing threads
            use_smooth (bool): Apply temporal smoothing to coefficients if True
            test_data (bool): Process as test data with different output structure if True
            skip (bool): Skip existing files if True
        """

        # Setup base directories
        save_dir = os.path.join(root_dir, 'dataset')
        os.makedirs(save_dir, exist_ok=True)

        # Load FaceVerse to FLAME transformation matrix
        fv2fl_T = np.load(fv2fl_T_path).astype(np.float32)

        # Coordinate transformation parameters
        orth_scale = 5.00  # Orthographic projection scaling factor
        orth_shift = np.array([0, 0.005, 0.], dtype=np.float32)  # Coordinate shift
        box_warp = 2.0  # Normalization scaling factor

        # Path configurations
        face_model_dir = 'lib/FaceVerse/v3'  # Pre-trained FaceVerse model
        save_render_dir = os.path.join(save_dir, 'orthRender256x256_face_eye' if save_name is None else save_name)
        save_mesh_dir = os.path.join(save_dir, 'FVmeshes512x512') if save_mesh else None
        save_uv_dir = os.path.join(save_dir, 'uvRender256x256') if save_uv else None

        # Render orthographic projections and process geometry
        render_orth_mp(
            tracking_dir,
            save_render_dir,
            face_model_dir,
            fv2fl_T,
            {'scale': orth_scale, 'shift': orth_shift},
            focal,
            render_vis=need_render,
            save_mesh_dir=save_mesh_dir,
            save_uv_dir=save_uv_dir,
            render_normal_uv=render_normal_uv,
            skip=skip,
            num_thread=num_thread,
            crop_param=[128, 114, 256, 256],  # Crop parameters: x_offset, y_offset, width, height
            save_coeff=True
        )

        # Compute normalization transformation matrix
        normalizeFL_T = np.eye(4, dtype=np.float32)
        scale_T = (orth_scale / box_warp) * np.eye(3, dtype=np.float32)  # Scaling component
        shift_T = scale_T.dot(orth_shift.reshape(3, 1))  # Translation component
        normalizeFL_T[:3, :3], normalizeFL_T[:3, 3:] = scale_T, shift_T

        # Update FaceVerse to FLAME transformation with normalization
        fv2fl_T = np.dot(normalizeFL_T, fv2fl_T)

        # Generate camera parameters dataset
        cam_params, cond_cam_params, fv_exp_eye_params = make_cam_dataset_FFHQ(
            tracking_dir, fv2fl_T, focal, test_data=test_data
        )

        # Handle different output structures for test vs training data
        if test_data:
            # Save per-sequence camera parameters
            for prefix in cam_params.keys():
                save_json_name = f'dataset_{prefix}_realcam.json'
                output_path = os.path.join(save_dir, 'images512x512', save_json_name)
                with open(output_path, "w") as f:
                    json.dump({"labels": cam_params[prefix]}, f, indent=4)
        else:
            # Save unified camera parameters with optional temporal smoothing
            save_json_name = 'dataset_realcam.json'
            if use_smooth:
                smoothed_params = []
                # Process each subdirectory sequence
                for sub_name in os.listdir(os.path.join(save_dir, 'images512x512')):
                    sub_path = os.path.join(save_dir, 'images512x512', sub_name)
                    if not os.path.isdir(sub_path):
                        continue

                    # Extract and sort sequence frames
                    sub_json = [case for case in cam_params if case[0].split('/')[0] == sub_name]
                    sub_json.sort(key=lambda x: int(x[0].split('/')[1].split('.')[0]))

                    # Apply Gaussian smoothing to coefficients
                    coeff_seq = np.array([x[1] for x in sub_json], dtype=np.float32)
                    coeff_seq = gaussian_filter1d(coeff_seq, sigma=1.5, axis=0)

                    # Rebuild parameter list with smoothed coefficients
                    smoothed_params.extend([
                        [x[0], coeff_seq[idx].tolist()]
                        for idx, x in enumerate(sub_json)
                    ])
                cam_params = smoothed_params

            # Save final parameters
            output_path = os.path.join(save_dir, 'images512x512', save_json_name)
            with open(output_path, "w") as f:
                json.dump({"labels": cam_params}, f, indent=4)

        # Generate final coefficient dataset
        make_coeff_dataset_FFHQ(tracking_dir, os.path.join(save_dir, 'coeffs'), smooth=use_smooth, is_img=is_img)

    def get_landmarks(self, imgs_root, save_dir, valid_imgs_json, skip=False, is_img=False):
        self.fd_ldmk_detector.reset()
        out_detection = save_dir
        os.makedirs(out_detection, exist_ok=True)
        valid_idx = json.loads(open(valid_imgs_json).read())
        no_face_log = []
        for vidx, (video_name, imgs) in enumerate(valid_idx):
            if skip and os.path.exists(os.path.join(out_detection, video_name + '.json')):
                continue
            bar = tqdm(imgs)
            save_kps = dict()
            save_kps_3d = dict()
            for img_name in bar:
                bar.set_description('%d/%d: %s' % (vidx, len(valid_idx), video_name))
                img_path = os.path.join(imgs_root, video_name, img_name)
                img = cv2.imread(img_path)
                with torch.no_grad():
                    try:
                        ldmks, ldmks_3d, boxes = self.fd_ldmk_detector.inference(img)
                    except Exception as e:
                        self.fd_ldmk_detector.reset()
                        logging.error(f"Error during inference: {e}")  # Error log
                        no_face_log.append([video_name, img_name])
                        continue
                    if is_img:
                        self.fd_ldmk_detector.reset()
                    # default the first one face
                    keypoints = ldmks[0, self.stand_index]
                    ldmks_3d = ldmks_3d[0]
                    kps = [[float(int(keypoints[0][0])), float(int(keypoints[0][1]))],
                           [float(int(keypoints[1][0])), float(int(keypoints[1][1]))],
                           [float(int(keypoints[2][0])), float(int(keypoints[2][1]))],
                           [float(int(keypoints[3][0])), float(int(keypoints[3][1]))],
                           [float(int(keypoints[4][0])), float(int(keypoints[4][1]))]
                           ]

                    save_kps[img_name] = kps
                    save_kps_3d[img_name] = ldmks_3d.tolist()
            logging.info(f"landmarks: {os.path.join(out_detection, video_name + '.json')}")
            with open(os.path.join(out_detection, video_name + '.json'), 'w') as f:
                f.write(json.dumps(save_kps, indent=4))
            with open(os.path.join(out_detection, video_name + '3d.json'), 'w') as f:
                f.write(json.dumps(save_kps_3d, indent=4))
        if len(no_face_log) > 0:
            jstr = json.dumps(no_face_log, indent=4)
            with open(os.path.join(out_detection, str(datetime.now()) + '_total_no_face_log.json'), 'w') as f:
                f.write(jstr)
        self.fd_ldmk_detector.reset()

    def get_pdfgc(self, input_imgs_dir, input_ldm3d_dir, motion_save_base_dir):
        all_items = os.listdir(input_ldm3d_dir)
        folders = [item for item in all_items if os.path.isdir(os.path.join(input_ldm3d_dir, item))]

        with torch.no_grad():
            for file_name in folders:
                motion_save_dir = os.path.join(motion_save_base_dir, file_name)
                os.makedirs(motion_save_dir, exist_ok=True)
                img_list = sorted(
                    [f for f in os.listdir(os.path.join(input_imgs_dir, file_name))
                     if os.path.splitext(f)[-1].lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}]
                )
                for img_name in img_list:
                    img_dir = os.path.join(input_imgs_dir, file_name, img_name)
                    lmks_dir = os.path.join(input_ldm3d_dir, file_name,
                                            img_name.replace('.png', '.npy').replace('.jpg', '.npy').replace('.jpeg',
                                                                                                             '.npy'))
                    img = np.array(Image.open(img_dir))
                    img = torch.from_numpy((img.astype(np.float32) / 127.5 - 1)).cuda()
                    img = img.permute([2, 0, 1]).unsqueeze(0)
                    lmks = torch.from_numpy(np.load(lmks_dir)).cuda().unsqueeze(0)
                    motion = get_motion_feature(self.pd_fgc, img, lmks).squeeze(0).cpu().numpy()
                    np.save(os.path.join(motion_save_dir, img_name.replace('.png', '.npy').replace('.jpg', '.npy')),
                            motion)


    def get_faceverse(self, save_dir, save_tracking_dir, is_img):
        fit_faceverse(save_dir, save_tracking_dir, is_img=is_img)

    def clean_labels(self, tracking_dir, labels_path, final_label_path):
        """
        Filter out labels containing frames with no detected faces

        Args:
            tracking_dir: Path to directory containing no-face detection logs
            labels_path: Path to original labels JSON file
            final_path: Output path for cleaned labels JSON
        """
        # Initialize collection of frames with no faces
        no_face_entries = []

        # Load all no-face detection logs
        for log_file in os.listdir(tracking_dir):
            if not log_file.endswith("_total_no_face_log.json"):
                continue
            with open(os.path.join(tracking_dir, log_file)) as f:
                no_face_entries.extend(json.load(f))

        # Load original labels and extract filenames
        with open(labels_path) as f:
            original_labels = json.load(f)['labels']
        label_filenames = [label[0] for label in original_labels]

        # Identify frames to exclude
        excluded_frames = set()
        for entry in no_face_entries:
            # Extract video and frame names from log entry path
            path_parts = entry[1].split('/')
            video_name = path_parts[-2]
            frame_name = path_parts[-1]
            composite_key = f"{video_name}/{frame_name}"

            logging.debug(f"Processing no-face entry: {composite_key}")

            if composite_key in label_filenames:
                excluded_frames.add(frame_name)

        logging.info(f"Identified {len(excluded_frames)} frames to exclude")

        # Filter out excluded frames from labels
        cleaned_labels = []
        for label in tqdm(original_labels, desc="Filtering labels"):
            # Label format: "video_name/frame_name"
            frame_id = label[0].split('/')[1]
            if frame_id not in excluded_frames:
                cleaned_labels.append(label)

        # Save cleaned labels
        logging.info(f"Original labels: {len(original_labels)}, Cleaned labels: {len(cleaned_labels)}")
        with open(final_label_path, 'w') as f:
            json.dump({'labels': cleaned_labels}, f, indent=4)

    def inference(self, input_dir, save_dir, is_video=True, is_img=False, smooth_cropping_mode=3.0,
                  no_extract_frames=False, no_extract_landmarks=False, no_align=False,
                  no_fitting_faceverse=False, no_render_faceverse=False, already_align=False, no_pdfgc_motion=False):
        """
        End-to-end processing pipeline for facial analysis and reconstruction.

        Parameters:
            input_dir (str): Source directory containing input videos/images
            save_dir (str): Root directory for all processed outputs
            is_video (bool): True when processing video files
            is_img (bool): True when processing image sequences
            smooth_cropping_mode (float): Smoothing strength for video frame alignment
            no_extract_frames (bool): Extract frames from video if True
            no_extract_landmarks (bool): Detect facial landmarks if True
            no_align (bool): Align and crop face regions if True
            no_fitting_faceverse (bool): Fit FaceVerse model if True
            no_render_faceverse (bool): Render FaceVerse outputs if True
        """

        # Initialize directory paths
        motions_save_dir = os.path.join(save_dir, 'dataset', "motions")
        data_save_dir = os.path.join(save_dir, 'dataset', "images512x512")  # Final processed images
        raw_detection_dir = os.path.join(save_dir, "raw_detections")  # Landmark detection results
        save_tracking_dir = os.path.join(save_dir, 'crop_fv_tracking')  # FaceVerse tracking data
        indir = os.path.join(save_dir, 'raw_frames')  # Raw extracted frames
        aligned3d_save_dir = os.path.join(save_dir, 'align_3d_landmark')

        # --- Frame Extraction Stage ---
        if not no_extract_frames:
            logging.info("Extracting frames from input source")
            valid_imgs_json = get_valid_input_idx(
                input_dir,
                indir,
                is_video=is_video,
                is_img=is_img,
                skip=1  # Frame sampling interval
            )
            logging.info(f"Frame extraction complete. Results in {indir}")
        else:
            valid_imgs_json = os.path.join(indir, 'valid_imgs.json')
            logging.info(f"Using pre-extracted frames from {indir}")

        # --- Landmark Detection Stage ---
        if not no_extract_landmarks:
            logging.info("Performing facial landmark detection")
            self.get_landmarks(indir, raw_detection_dir, valid_imgs_json, is_img=is_img)
            logging.info(f"Landmark detection complete. Results in {raw_detection_dir}")
        else:
            logging.info(f"Using pre-computed landmarks from {raw_detection_dir}")

        # --- Face Alignment Stage ---
        if not no_align:
            logging.info("Aligning and cropping face regions")
            if is_video:
                # Video processing with temporal smoothing
                recreate_aligned_videos_multiprocessing(
                    indir,
                    raw_detection_dir,
                    data_save_dir,
                    valid_imgs_json,
                    apply_GF=smooth_cropping_mode  # Gaussian filtering strength
                )
            elif is_img:
                # Image sequence processing
                recreate_aligned_images(
                    indir,
                    raw_detection_dir,
                    data_save_dir,
                    already_align=already_align,
                    valid_imgs_json=valid_imgs_json
                )
            else:
                raise ValueError("Invalid input type - must be video or image sequence")
            logging.info(f"Alignment complete. Results in {data_save_dir}")
        else:
            logging.info(f"Using pre-aligned images from {data_save_dir}")

        if not no_pdfgc_motion:
            logging.info("Getting the pdfgc motions")
            self.get_pdfgc(data_save_dir, aligned3d_save_dir, motions_save_dir)
            logging.info(f"Alignment complete. Results in {motions_save_dir}")
        else:
            logging.info(f"Using pre-aligned images from {motions_save_dir}")

        # --- FaceVerse Model Fitting Stage ---
        if not no_fitting_faceverse:
            logging.info("Fitting FaceVerse 3D face model")
            self.get_faceverse(
                data_save_dir,
                save_tracking_dir,
                is_img  # Different processing for images vs video
            )
            logging.info(f"FaceVerse fitting complete. Results in {save_tracking_dir}")
        else:
            logging.info(f"Using pre-computed FaceVerse fits from {save_tracking_dir}")

        # --- Rendering and Final Output Stage ---
        if not no_render_faceverse:
            logging.info("Generating FaceVerse renders and camera parameters")
            self.get_faceverse_labels_FFHQ(
                save_tracking_dir,
                save_dir,
                is_img=is_img
            )
            logging.info("Rendering and label generation complete")
        else:
            logging.info("Skipping final rendering stage")

        if is_video:
            logging.info("Clean labels")
            self.clean_labels(save_tracking_dir, os.path.join(data_save_dir, 'dataset_realcam.json'),
                              os.path.join(data_save_dir, 'dataset_realcam_clean.json'))


def main():
    os.makedirs(cfg.save_dir, exist_ok=True)
    process = Process(cfg)
    process.inference(cfg.input_dir, cfg.save_dir, is_video=cfg.is_video, is_img=cfg.is_img,
                      no_extract_frames=cfg.no_extract_frames, no_extract_landmarks=cfg.no_extract_landmarks,
                      no_align=cfg.no_align,
                      no_fitting_faceverse=cfg.no_fitting_faceverse, no_render_faceverse=cfg.no_render_faceverse,
                      already_align=cfg.already_align, no_pdfgc_motion=cfg.no_pdfgc_motion)


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)
    main()
