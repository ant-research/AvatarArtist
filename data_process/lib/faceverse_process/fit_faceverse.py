# Standard libraries
import os
import time
import json
import traceback
import multiprocessing
from datetime import datetime

# Numerical and image processing libraries
import numpy as np
import cv2
import torch

# Core functionalities
from lib.faceverse_process.core import get_recon_model
import lib.faceverse_process.core.utils as utils
import lib.faceverse_process.core.losses as losses

# Third-party libraries
from tqdm import tqdm  # Progress bar
from pytorch3d.renderer import look_at_view_transform  # 3D transformations
import mediapipe as mp  # Face landmark detection
count = multiprocessing.Value('i', 0)  # multiprocessing.Value对象和Process一起使用的时候，可以像上面那样作为全局变量使用，也可以作为传入参数使用。但是和Pool一起使用的时候，只能作为全局变量使用
total = multiprocessing.Value('i', 0)



def fit_faceverse(
    base_dir: str,
    save_dir: str = None,
    skip: bool = False,
    save_fvmask: str = None,
    save_lmscounter: str = None,
    num_threads: int = 8,
    trick: int = 0,
    focal_ratio: float = 4.2647,  # Focal length used by EG3D
    is_img = False
):
    """
    Processes multiple video frames for face reconstruction using multiprocessing.

    Args:
        base_dir (str): Base directory containing input images.
        save_dir (str): Directory to save results (default: auto-generated).
        skip (bool): Whether to skip already processed frames.
        save_fvmask (str or None): Path to save face visibility mask.
        save_lmscounter (str or None): Path to save landmark counter visualization.
        num_threads (int): Number of threads to use for multiprocessing.
        trick (int): Processing strategy (-1, 0, or 1) for selecting frames.
        focal_ratio (float): Focal length scaling factor.

    """
    data_save_dir = os.path.join(save_dir, 'dataset', "images512x512")  # Final processed images
    save_tracking_dir = os.path.join(save_dir, 'crop_fv_tracking')
    # Ensure base directory exists
    assert os.path.exists(base_dir), f"Base directory '{base_dir}' does not exist."

    # Ensure base_dir contains 'images512x512' when saving masks or landmark counters
    if save_lmscounter or save_fvmask:
        assert 'images512x512' in base_dir, "Base directory must contain 'images512x512' when saving masks or landmark counters."

    # Define save directory (default: `fv_tracking`)
    save_dir = save_dir if save_dir else os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'fv_tracking')
    os.makedirs(save_dir, exist_ok=True)

    # Image resolution
    img_res = 512

    # Initialize camera intrinsic matrix
    cam_K = np.eye(3, dtype=np.float32)
    cam_K[0, 0] = cam_K[1, 1] = focal_ratio * img_res
    cam_K[0, 2] = cam_K[1, 2] = img_res // 2

    all_frames = 0
    sub_class_ls = []  # List to store video metadata

    # Get list of subdirectories (video folders) that haven't been processed
    sub_classes = [
        sub_class for sub_class in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, sub_class)) and sub_class not in os.listdir(save_dir)
    ]

    # Apply processing strategy based on `trick` argument
    if trick != 0:
        assert trick in [-1, 1], "Invalid trick value. Must be -1, 0, or 1."
        sub_classes = sub_classes[::2] if trick == 1 else sub_classes[1::2]

    # Process each subdirectory (video folder)
    for sub_class in tqdm(sub_classes, desc="Processing Videos"):
        sub_dir = os.path.join(base_dir, sub_class)
        if not os.path.isdir(sub_dir):
            continue

        frame_ls = []  # List to store frames for the current video

        # Iterate through images in the subdirectory
        for img_name in os.listdir(sub_dir):
            if not img_name.endswith('png'):
                continue

            # Define save folder for the current frame
            res_folder = os.path.join(sub_dir.replace(base_dir, save_dir), img_name.split('.')[0])

            # Skip processing if a 'finish' flag exists
            if skip and os.path.exists(os.path.join(res_folder, 'finish')):
                continue

            # Store frame metadata
            frame_ls.append({
                'img_path': os.path.join(sub_dir, img_name),
                'save_dir': res_folder
            })

        # Skip videos with no valid frames
        if not frame_ls:
            continue
        # Sort frames by numerical index extracted from filename
        if not is_img:
            frame_ls.sort(key=lambda x: int(os.path.basename(x['img_path']).split('.')[0].split('_')[-1]))

        # Store video metadata
        sub_class_ls.append({'video_name': sub_class, 'frame_ls': frame_ls})
        all_frames += len(frame_ls)

    # Store total number of frames for processing
    total.value = all_frames
    num_threads = min(num_threads, len(sub_class_ls))  # Adjust thread count based on available videos

    # Logging
    print(f"Base Directory: {base_dir}")
    print(f"Save Directory: {save_dir}")
    print(f"Skip Processed: {skip}")
    print(f"Number of Threads: {num_threads}")
    print(f"Total Frames: {total.value}")

    # Multi-threaded processing
    if num_threads > 1:
        p = multiprocessing.Pool(num_threads)

        # Distribute videos across threads
        num_videos = len(sub_class_ls)
        all_list = [
            sub_class_ls[i * (num_videos // num_threads): (i + 1) * (num_videos // num_threads)]
            for i in range(num_threads)
        ] + [sub_class_ls[num_threads * (num_videos // num_threads):]]

        # Prepare data for parallel processing
        data_ls = [
            {
                'img_res': img_res, 'video_ls': ls, 'save_dir': save_dir, 'cam_K': cam_K,
                'save_fvmask': save_fvmask, 'save_lmscounter': save_lmscounter, 'is_img':is_img
            }
            for ls in all_list
        ]

        # Start multiprocessing
        p.map(fit_videos_, data_ls)
        p.close()
        p.join()
    else:
        # Single-threaded processing (fallback)
        fit_videos_({
            'img_res': img_res, 'video_ls': sub_class_ls, 'save_dir': save_dir, 'cam_K': cam_K,
            'save_fvmask': save_fvmask, 'save_lmscounter': save_lmscounter, 'is_img':is_img
        })

    # Collect and aggregate no-face logs
    no_face_log = []
    for name in os.listdir(save_dir):
        if name.endswith('no_face_log.json'):
            with open(os.path.join(save_dir, name), 'r') as f:
                no_face_log += json.load(f)

    # Save aggregated no-face log if any entries exist
    if no_face_log:
        log_filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_total_no_face_log.json'
        with open(os.path.join(save_dir, log_filename), 'w') as f:
            json.dump(no_face_log, f, indent=4)

def fit_videos_(data):
    """
    Process and fit multiple videos using a face reconstruction model.

    Args:
        data (dict): Dictionary containing the parameters:
            - 'img_res' (int): Image resolution.
            - 'video_ls' (list): List of video dictionaries containing frame information.
            - 'save_dir' (str): Directory to save results.
            - 'cam_K' (numpy array): Camera intrinsic matrix.
            - 'save_fvmask' (str or None): Path to save face visibility mask.
            - 'save_lmscounter' (str or None): Path to save landmark counter visualization.
    """
    config = {
        "tar_size": 512,
        "recon_model": "meta_simplify_v31",
        "lm_loss_w": 1e3,
        "rgb_loss_w": 1e-2,
        "id_reg_w": 3e-3,
        "exp_reg_w": 1e-3,  # Previously 8e-3
        "tex_reg_w": 3e-5,
        "tex_w": 1.0,
        "skip": False,
        "save_fvmask": None,
        "save_lmscounter": None,
        "num_threads": 8,
        "trick": 0,
        "focal_ratio": 4.2647,  # Focal length used by EG3D
        "cam_dist": 5.0,
        "device": "cuda:0"
    }
    # Extract data parameters
    img_res = data['img_res']
    video_ls = data['video_ls']
    save_dir = data['save_dir']
    cam_K = data['cam_K']
    save_fvmask = data['save_fvmask']
    save_lmscounter = data['save_lmscounter']
    is_img = data['is_img']

    print(f'Fitting {len(video_ls)} Videos')

    # Scale camera intrinsic matrix based on target image size
    cam_K[:2] *= config["tar_size"]/ img_res

    # Initialize MediaPipe face mesh detector
    mp_tracker = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.2, min_tracking_confidence=0.2
    )

    # Initialize the face reconstruction model
    recon_model = get_recon_model(
        model=config["recon_model"],
        device=config["device"],
        batch_size=1,
        img_size=config["tar_size"],
        intr=cam_K,
        cam_dist=config["cam_dist"]
    )

    no_face_log = []  # Log for frames where no face is detected

    # Iterate through each video in the list
    for vidx, video_info in enumerate(video_ls):
        print(video_info['frame_ls'][0]['img_path'], vidx)

        # Process the frames using the `fit_()` function
        no_face_log_ = fit_(
            video_info['frame_ls'], recon_model, img_res, config, mp_tracker,
            cont_opt=False, first_video=(vidx == 0), reg_RT=True,
            save_fvmask=save_fvmask, save_lmscounter=save_lmscounter, is_img=is_img
        )

        # Create a "finish" flag file or log issues if face fitting fails
        video_save_path = os.path.join(save_dir, video_info['video_name'])
        if not no_face_log_:
            open(os.path.join(video_save_path, 'finish'), "w").close()
        else:
            issue_type = no_face_log_[0][0]
            if issue_type == 'LargeRot':
                open(os.path.join(video_save_path, 'LargeRot'), "w").close()
            elif issue_type == 'NoFace':
                open(os.path.join(video_save_path, 'NoFace'), "w").close()
            elif issue_type == 'SamllFace':  # Fixed typo ('SamllFace' → 'SmallFace')
                open(os.path.join(video_save_path, 'SmallFace'), "w").close()

        # Append detected no-face logs
        no_face_log += no_face_log_

    # Save log of frames where no face was detected
    if no_face_log:
        log_path = os.path.join(save_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_no_face_log.json")
        with open(log_path, 'w') as f:
            json.dump(no_face_log, f, indent=4)
    else:
        print('No face log entries recorded.')


def fit_(frame_ls, recon_model, img_res, config, mp_tracker, first_video=False, save_mesh=False, keep_id=True, reg_RT=False,
         save_fvmask=None, save_lmscounter=None, cont_opt=False, is_img=False):

    if is_img:
        keep_id = False
    lm_weights = utils.get_lm_weights(config["device"], use_mediapipe=True)
    resize_factor = config["tar_size"] / img_res

    rt_reg_w = 0.1 if reg_RT else 0.
    num_iters_rf = 100 if keep_id else 500

    frame_ind = 0
    no_face_log = []

    for frame_dict in frame_ls:
        frame_ind += 1
        if is_img:
            frame_ind = 0
        res_folder = frame_dict['save_dir']

        # Create the results folder if it doesn't exist
        os.makedirs(res_folder, exist_ok=True)

        img_path = frame_dict['img_path']

        # Use a lock to safely update and print the processing count
        with count.get_lock():
            count.value += 1
            print('(%d / %d) Processing frame %s, first_video=%d' %
                  (count.value, total.value, img_path, int(first_video)))

        # Read the image and convert from BGR to RGB
        img_arr = cv2.imread(img_path)[:, :, ::-1]

        # Resize the face image to the target size
        resized_face_img = cv2.resize(img_arr, (config["tar_size"], config["tar_size"]))

        # Process the image using MediaPipe face tracking
        results = mp_tracker.process(resized_face_img)

        # If no face landmarks are detected, log and skip processing
        if results.multi_face_landmarks is None:
            print('No face detected!', img_path)
            no_face_log.append(['NoFace', img_path])
            continue

        # Initialize a numpy array to store facial landmarks (478 points, 2D coordinates)
        lms = np.zeros((478, 2), dtype=np.int64)

        # Extract face landmarks and store in the array
        for idx, landmark in enumerate(results.multi_face_landmarks[0].landmark):
            lms[idx, 0] = int(landmark.x * config["tar_size"])
            lms[idx, 1] = int(landmark.y * config["tar_size"])
        # Check if the detected face is too small based on bounding box size
        if max(max(lms[:, 0]) - min(lms[:, 0]), max(lms[:, 1]) - min(lms[:, 1])) < config["tar_size"] / 3:
            print('Too small face detected!', img_path)
            no_face_log.append(['SmallFace', img_path])
            continue

        # Convert landmarks to a PyTorch tensor and move to the specified device
        lms_tensor = torch.tensor(lms[np.newaxis, :, :], dtype=torch.float32, device=config["device"])

        # If continuation option is enabled, check for existing coefficient file
        if cont_opt:
            coeffs_path = os.path.join(res_folder, 'coeffs.npy')

            # Load and initialize coefficients if they already exist
            if os.path.exists(coeffs_path):
                coeffs = torch.from_numpy(np.load(coeffs_path)).unsqueeze(0).cuda()

                # Split the loaded coefficients into respective components
                (id_coeff, exp_coeff, tex_coeff, angles, gamma, translation,
                 eye_coeff, scale) = recon_model.split_coeffs(coeffs)

                # Initialize the reconstruction model with the loaded coefficients
                recon_model.init_coeff_tensors(
                    id_coeff=id_coeff, tex_coeff=tex_coeff, exp_coeff=exp_coeff,
                    gamma_coeff=gamma, trans_coeff=translation,
                    rot_coeff=angles, scale_coeff=scale, eye_coeff=eye_coeff
                )

                first_video = False  # Indicate that this is not the first video

        # Determine which parameters to optimize based on `keep_id` and frame index
        if keep_id and frame_ind > 1:
            # Keep identity coefficients fixed when optimizing rigid parameters
            rigid_optim_params = [
                recon_model.get_rot_tensor(), recon_model.get_trans_tensor(),
                recon_model.get_exp_tensor(), recon_model.get_eye_tensor()
            ]
        else:
            # Optimize identity coefficients along with other rigid parameters
            rigid_optim_params = [
                recon_model.get_rot_tensor(), recon_model.get_trans_tensor(),
                recon_model.get_exp_tensor(), recon_model.get_eye_tensor(),
                recon_model.get_id_tensor()
            ]

        # Define optimizers for rigid parameter optimization
        rigid_optimizer = torch.optim.Adam(
            rigid_optim_params,
            lr=5e-2 if (first_video and frame_ind == 1) else 1e-2,
            betas=(0.8, 0.95)
        )

        # Learning-rate-adjusted optimizer for rigid parameters
        lr_rigid_optimizer = torch.optim.Adam(
            rigid_optim_params,
            lr=1e-3,
            betas=(0.5, 0.9)
        )

        # Determine the number of iterations for rigid optimization
        num_iters = 5 * num_iters_rf if (keep_id and frame_ind == 1) else num_iters_rf

        # Increase iterations significantly for the first frame of the first video
        if first_video and frame_ind == 1:
            num_iters *= 5
        # Perform rigid optimization for num_iters * 5 iterations
        for iter_rf in range(num_iters * 5):
            # Forward pass: get predicted landmarks without rendering
            pred_dict = recon_model(recon_model.get_packed_tensors(), render=False)

            # Compute landmark loss between predicted and ground truth landmarks
            lm_loss_val = losses.lm_loss(pred_dict['lms_proj'], lms_tensor, lm_weights, img_size=config["tar_size"])

            # Early stopping condition: if loss is sufficiently low, break the loop
            if iter_rf > num_iters and lm_loss_val.item() < 5e-5:
                break

            # Regularization losses to prevent overfitting
            id_reg_loss = losses.get_l2(recon_model.get_id_tensor())  # Identity regularization
            exp_reg_loss = losses.get_l2(recon_model.get_exp_tensor())  # Expression regularization

            # Compute total loss with weighted sum of different loss components
            total_loss = (config["lm_loss_w"] * lm_loss_val +
                          exp_reg_loss * config["exp_reg_w"] +
                          id_reg_loss * config["id_reg_w"])

            # Add rotation and translation regularization if not processing the first frame
            if frame_ind > 1:
                rt_reg_loss = (losses.get_l2(recon_model.get_rot_tensor() - rot_c) +
                               losses.get_l2(recon_model.get_trans_tensor() - trans_c))
                total_loss += rt_reg_loss * rt_reg_w  # Apply regularization weight

            # Choose optimizer based on iteration count and frame number
            if frame_ind > 1 and iter_rf > num_iters * 0.6:
                lr_rigid_optimizer.zero_grad()
                total_loss.backward()
                lr_rigid_optimizer.step()
            else:
                rigid_optimizer.zero_grad()
                total_loss.backward()
                rigid_optimizer.step()

            # Ensure all expression values remain non-negative (zero negative expressions)
            with torch.no_grad():
                recon_model.exp_tensor[recon_model.exp_tensor < 0] *= 0
        rot_c, trans_c = recon_model.get_rot_tensor().clone().detach(), recon_model.get_trans_tensor().clone().detach()
        with torch.no_grad():
            # Get the packed coefficient tensors from the reconstruction model
            coeffs = recon_model.get_packed_tensors()

            # Forward pass to get predictions, including rendering and face masking
            pred_dict = recon_model(coeffs, render=True, mask_face=True)

            # Clip rendered image values to [0, 255] and convert to NumPy format
            rendered_img = torch.clip(pred_dict['rendered_img'], 0, 255).cpu().numpy().squeeze()
            out_img = rendered_img[:, :, :3].astype(np.uint8)

            # Resize output image to match the specified resolution
            resized_out_img = cv2.resize(out_img, (img_res, img_res))

            # Save the coefficient tensors as a NumPy file
            np.save(os.path.join(res_folder, 'coeffs.npy'), coeffs.detach().cpu().numpy().squeeze())

            # Extract specific coefficients for later use
            split_coeffs = recon_model.split_coeffs(coeffs)
            tex_coeff, angles, translation, scale = split_coeffs[2], split_coeffs[3], split_coeffs[5], split_coeffs[-1]

            # Save the 3D mesh in .obj format if required
            if save_mesh:
                vs = pred_dict['vs'].cpu().numpy().squeeze()  # Vertex positions
                tri = pred_dict['tri'].cpu().numpy().squeeze()  # Triangle indices

                # Compute vertex colors and normalize to [0,1]
                color = torch.clip(recon_model.get_color(tex_coeff), 0, 255).cpu().numpy().squeeze().astype(
                    np.float32) / 255

                # Save the mesh as an OBJ file
                utils.save_obj(os.path.join(res_folder, 'mesh.obj'), vs, tri + 1, color)

            # Compute extrinsic camera parameters
            rotation = recon_model.compute_rotation_matrix(angles)  # Compute rotation matrix

            # Initialize transformation matrices
            cam_T = torch.eye(4, dtype=torch.float32).to(config["device"])  # Camera transformation
            tmp_T = torch.eye(4, dtype=torch.float32).to(config["device"])  # Temporary transformation

            # Compute camera rotation and translation matrices
            cam_R, cam_t = look_at_view_transform(dist=config["cam_dist"], elev=0, azim=0)
            tmp_T[:3, :3] = cam_R[0]  # Set rotation
            tmp_T[-1, :3] = cam_t[0]  # Set translation

            # Compute metaFace extrinsic matrix
            cam_T[:3, :3] = torch.abs(scale[0]) * torch.eye(3, dtype=torch.float32).to(config["device"])
            cam_T[-1, :3] = translation[0]
            metaFace_extr = torch.matmul(cam_T, tmp_T).clone()  # Left-multiply transformation

            # Compute final transformation matrix
            cam_T[:3, :3] = torch.abs(scale[0]) * rotation[0]
            cam_T[-1, :3] = translation[0]
            transformation = torch.matmul(cam_T, tmp_T)  # Left-multiply transformation

            # Save extrinsic parameters as a NumPy archive
            np.savez(os.path.join(res_folder, 'metaFace_extr'),
                     extr=metaFace_extr.cpu().numpy().astype(np.float32).T,  # Transposed for right multiplication
                     transformation=transformation.cpu().numpy().astype(np.float32).T,
                     self_rotation=rotation[0].cpu().numpy().astype(np.float32).T,
                     self_scale=scale[0].cpu().numpy().astype(np.float32),
                     self_translation=translation[0].cpu().numpy().astype(np.float32),
                     self_angle=angles[0].cpu().numpy().astype(np.float32))

            # Blend original and rendered images for visualization
            composed_img = img_arr * 0.6 + resized_out_img * 0.4

            # Resize and normalize landmark coordinates
            resized_lms = lms_tensor.cpu().detach().squeeze().numpy() / resize_factor
            resized_lms_proj = pred_dict['lms_proj'].cpu().detach().squeeze().numpy() / resize_factor

            # Overlay landmarks on the composed image
            composed_img = visualize_render_lms(composed_img, resized_lms, resized_lms_proj)
            cv2.imwrite(os.path.join(res_folder, 'composed_render.png'), composed_img[:, :, ::-1].astype(np.uint8))

            # Save face visibility mask if required
            if save_fvmask is not None:
                out_mask = (np.linalg.norm(resized_out_img, axis=-1) > 0).astype(np.float32) * 255
                os.makedirs(os.path.dirname(img_path.replace('images512x512', save_fvmask)), exist_ok=True)
                cv2.imwrite(img_path.replace('images512x512', save_fvmask), out_mask.astype(np.uint8))

            # Save landmark counter visualization if required
            if save_lmscounter is not None:
                lms_proj = pred_dict['lms_proj'].cpu().detach().squeeze().numpy()
                black_img = np.zeros((config["tar_size"], config["tar_size"], 3), dtype=np.uint8)
                draw_img = draw_lms_counter(black_img, lms_proj)
                os.makedirs(os.path.dirname(img_path.replace('images512x512', save_lmscounter)), exist_ok=True)
                cv2.imwrite(img_path.replace('images512x512', save_lmscounter), draw_img)

        # Create a 'finish' file to indicate processing completion
        open(os.path.join(res_folder, 'finish'), "w")

    return no_face_log


def visualize_render_lms(composed_img, resized_lms, resized_lms_proj):
    """
    Visualizes facial landmarks on an image.

    Args:
        composed_img (np.ndarray): The input image to draw on.
        resized_lms (np.ndarray): Original 2D facial landmarks (shape: [N, 2]).
        resized_lms_proj (np.ndarray): Projected facial landmarks (shape: [N, 2]).

    Returns:
        np.ndarray: The image with drawn facial landmarks.
    """

    # Convert landmark coordinates to integer values for drawing
    resized_lms = np.round(resized_lms).astype(np.int32)
    resized_lms_proj = np.round(resized_lms_proj).astype(np.int32)

    # Landmark indices to annotate with numbers
    annotated_indices = [0, 8, 16, 20, 24, 30, 47, 58, 62]

    # Draw original landmarks (Blue)
    for (x, y) in resized_lms:
        cv2.circle(composed_img, (x, y), radius=1, color=(255, 0, 0), thickness=-1)

    # Annotate specific original landmarks (Yellow)
    for i in annotated_indices:
        cv2.putText(composed_img, str(i), tuple(resized_lms[i]),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 0), thickness=1)

    # Draw projected landmarks (Green)
    for (x, y) in resized_lms_proj:
        cv2.circle(composed_img, (x, y), radius=1, color=(0, 255, 0), thickness=-1)

    # Annotate specific projected landmarks (Cyan)
    for i in annotated_indices:
        cv2.putText(composed_img, str(i), tuple(resized_lms_proj[i]),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 255), thickness=1)

    return composed_img


def draw_lms_counter(img, lms_proj):
    """
    Draws facial landmarks on an image, including mouth, eyes, and specific points.

    Args:
        img (np.ndarray): The input image.
        lms_proj (np.ndarray): The projected 2D facial landmarks (shape: [N, 2]).

    Returns:
        np.ndarray: The image with drawn facial landmarks.
    """

    # Convert landmark coordinates to integer values
    lms_proj_coords = np.round(lms_proj).astype(np.int32)

    # Define landmark indices for different facial features
    outter_mouth_idx = [0, 267, 269, 270, 409, 291, 375, 321, 405,
                        314, 17, 84, 181, 91, 146, 76, 185, 40, 39, 37]
    inner_mouth_idx = [13, 312, 311, 310, 415, 308, 324, 318, 402,
                       317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82]
    left_eye_idx = [33, 246, 161, 160, 159, 158, 157, 173, 133,
                    155, 154, 153, 145, 144, 163, 7]
    right_eye_idx = [362, 398, 384, 385, 386, 387, 388, 466, 263,
                     249, 390, 373, 374, 380, 381, 382]
    left_brow_idx = [283, 282, 295, 285, 336, 296, 334]
    right_brow_idx = [53, 52, 65, 55, 107, 66, 105]

    # Create a copy of the image to draw on
    draw_img = img.copy()

    # Draw facial landmarks for mouth (outer and inner)
    draw_img = cv2.polylines(draw_img, [lms_proj_coords[outter_mouth_idx]],
                             isClosed=True, color=(255, 0, 0), thickness=4)
    draw_img = cv2.polylines(draw_img, [lms_proj_coords[inner_mouth_idx]],
                             isClosed=True, color=(255, 0, 0), thickness=4)

    # Draw facial landmarks for eyes
    draw_img = cv2.polylines(draw_img, [lms_proj_coords[left_eye_idx]],
                             isClosed=True, color=(0, 255, 0), thickness=2)
    draw_img = cv2.polylines(draw_img, [lms_proj_coords[right_eye_idx]],
                             isClosed=True, color=(0, 255, 0), thickness=2)

    # Uncomment to draw eyebrows
    # draw_img = cv2.polylines(draw_img, [lms_proj_coords[left_brow_idx]],
    #                          isClosed=True, color=(0, 255, 0), thickness=2)
    # draw_img = cv2.polylines(draw_img, [lms_proj_coords[right_brow_idx]],
    #                          isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw specific landmark points (e.g., pupils or reference points)
    draw_img = cv2.circle(draw_img, tuple(lms_proj_coords[473]),
                           radius=4, color=(0, 0, 255), thickness=-1)
    draw_img = cv2.circle(draw_img, tuple(lms_proj_coords[468]),
                           radius=4, color=(0, 0, 255), thickness=-1)

    # Uncomment to draw additional facial contours
    # draw_img = cv2.polylines(draw_img, [lms_proj_coords[474:478]],
    #                          isClosed=True, color=(0, 255, 0), thickness=1)
    # draw_img = cv2.polylines(draw_img, [lms_proj_coords[469:473]],
    #                          isClosed=True, color=(0, 255, 0), thickness=1)

    return draw_img

#
#
# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--base_dir', type=str, default=None)
#     parser.add_argument('--save_dir', type=str, default=None)
#     parser.add_argument('--tar_size', type=int, default=512, help='size for rendering window. We use a square window.')
#     parser.add_argument('--recon_model', type=str, default='meta', help='choose a 3dmm model, default: meta')
#     parser.add_argument('--lm_loss_w', type=float, default=1e3, help='weight for landmark loss')
#     parser.add_argument('--rgb_loss_w', type=float, default=1e-2, help='weight for rgb loss')
#     parser.add_argument('--id_reg_w', type=float, default=3e-3, help='weight for id coefficient regularizer')
#     parser.add_argument('--exp_reg_w', type=float, default=1e-3,  # 8e-3
#                         help='weight for expression coefficient regularizer')
#     parser.add_argument('--tex_reg_w', type=float, default=3e-5, help='weight for texture coefficient regularizer')
#     parser.add_argument('--tex_w', type=float, default=1, help='weight for texture reflectance loss.')
#     parser.add_argument('--skip', action='store_true', default=False)
#     parser.add_argument('--save_fvmask', type=str, default=None)
#     parser.add_argument('--save_lmscounter', type=str, default=None)
#     parser.add_argument('--num_threads', default=8)
#     parser.add_argument('--trick', type=int, default=0)
#     args = parser.parse_args()
#     args.focal_ratio = 4.2647  # the focal used by EG3D
#     args.cam_dist = 5.
#     args.device = 'cuda:0'
#     args.recon_model = 'meta_simplify_v31'
#     fit_faceverse(args)
