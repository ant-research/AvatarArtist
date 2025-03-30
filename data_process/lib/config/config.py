import os.path

from .yacs import CfgNode as CN
import argparse
from pathlib import Path

current_file = Path(__file__).resolve()

project_root = current_file.parent.parent.parent.parent
cfg_path = os.path.join(project_root, 'data_process/configs/pipeline_config_demo.yaml')

def parse_cfg(cfg, args):
    """Transfer command line arguments to configuration node"""
    # Input/output configuration
    cfg.input_dir = args.input_dir  # Source input directory path
    cfg.save_dir = args.save_dir  # Output directory path

    # Pipeline control flags
    cfg.is_video = args.is_video  # Video processing mode flag
    cfg.is_img = args.is_img  # Image processing mode flag

    # Processing stage flags
    cfg.no_extract_frames = args.no_extract_frames  # Frame extraction enabled
    cfg.no_extract_landmarks = args.no_extract_landmarks  # Landmark detection enabled
    cfg.no_align = args.no_align  # Face alignment enabled
    cfg.no_fitting_faceverse = args.no_fitting_faceverse  # 3D face fitting enabled
    cfg.no_render_faceverse = args.no_render_faceverse  # Final rendering enabled
    cfg.already_align = args.already_align  # Final rendering enabled
    cfg.no_pdfgc_motion = args.no_pdfgc_motion

def make_cfg(args):
    """Create and merge configuration from file and command line"""
    # Initialize configuration node
    cfg = CN()

    # Merge with YAML configuration file
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)

    # Override with command line arguments
    parse_cfg(cfg, args)
    cfg.model.facerecon.checkpoints_dir = os.path.join(project_root, cfg.model.facerecon.checkpoints_dir)
    cfg.model.fd.model_path = os.path.join(project_root, cfg.model.fd.model_path)
    cfg.model.ldmk.model_path= os.path.join(project_root, cfg.model.ldmk.model_path)
    cfg.model.ldmk_3d.model_path = os.path.join(project_root, cfg.model.ldmk_3d.model_path)
    cfg.model.ldmk_3d.model_depth_path = os.path.join(project_root, cfg.model.ldmk_3d.model_depth_path)
    cfg.pdfgc_path = os.path.join(project_root, cfg.pdfgc_path)

    return cfg


# Command line argument configuration -------------------------------------------------
parser = argparse.ArgumentParser(description="Face Processing Pipeline Configuration")

# I/O configuration
parser.add_argument("--input_dir", default='', type=str,
                    help="Input directory containing source media")
parser.add_argument("--save_dir", default='', type=str,
                    help="Output directory for processed results")

# Configuration file
parser.add_argument("--cfg_file", default=cfg_path, type=str,
                    help="Path to YAML configuration file")

# Processing mode flags
parser.add_argument("--is_video", action='store_true',
                    help="Enable video processing mode")
parser.add_argument("--is_img", action='store_true',
                    help="Enable image sequence processing mode")

# Pipeline stage control flags (default enabled, use flag to disable)
parser.add_argument('--no_extract_frames', action='store_true',
                    help="Disable frame extraction stage")
parser.add_argument('--no_extract_landmarks', action='store_true',
                    help="Disable facial landmark detection")
parser.add_argument('--no_align', action='store_true',
                    help="Disable face alignment stage")
parser.add_argument('--no_fitting_faceverse', action='store_true',
                    help="Disable FaceVerse model fitting")
parser.add_argument('--no_render_faceverse', action='store_true',
                    help="Disable final rendering stage")
parser.add_argument('--already_align', action='store_true',
                    help="already_align")
parser.add_argument('--no_pdfgc_motion', action='store_true',)
# Parse arguments and build configuration ---------------------------------------------
args = parser.parse_args()
cfg = make_cfg(args)  # Final merged configuration