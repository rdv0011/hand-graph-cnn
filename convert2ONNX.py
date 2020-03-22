from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os.path as osp
import torch

from hand_shape_pose.config import cfg
from hand_shape_pose.model.shape_pose_network import ShapePoseNetwork
from hand_shape_pose.data.build import build_dataset

from hand_shape_pose.util.logger import setup_logger, get_logger_filename
from hand_shape_pose.util.miscellaneous import mkdir
from hand_shape_pose.util.vis import save_batch_image_with_mesh_joints
from hand_shape_pose.util import renderer


def main():
    parser = argparse.ArgumentParser(description="3D Hand Shape and Pose Inference")
    parser.add_argument(
        "--config-file",
        default="configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = osp.join(cfg.EVAL.SAVE_DIR, args.config_file)
    mkdir(output_dir)
    logger = setup_logger("hand_shape_pose_inference", output_dir, filename='eval-' + get_logger_filename())
    logger.info(cfg)

    # 1. Load network model
    model = ShapePoseNetwork(cfg, output_dir)
    device = cfg.MODEL.DEVICE
    model.to(device)
    model.load_model(cfg)

    # Convert to ONNX
    # Define input / output names
    input_names2 = ["images", "cam_param", "bbox", "pose_root", "pose_scale"]
    output_names2 = ["mesh", "pose_uv", "pose_xyz"] # B x 1280 x 3, B x K x 3, B x K x 3

    dynamic_axes = {'images': {0: 'batch'}, 'cam_param': {0: 'batch'},
    'bbox': {0: 'batch'}, 'pose_root': {0: 'batch'}, 'pose_scale': {0: 'batch'}}

    dummy_images = torch.rand(cfg.MODEL.BATCH_SIZE, 256, 256, 3)
    dummy_cam_param = torch.rand(cfg.MODEL.BATCH_SIZE, 4)
    dummy_bbox = torch.rand(cfg.MODEL.BATCH_SIZE, 4)
    dummy_pose_root = torch.rand(cfg.MODEL.BATCH_SIZE, 3)
    dummy_pose_scale = torch.rand(cfg.MODEL.BATCH_SIZE)

    # Convert the PyTorch model to ONNX
    torch.onnx.export(model,
                    (dummy_images, dummy_cam_param, dummy_bbox, dummy_pose_root, dummy_pose_scale),
                    "./converted/hand3d.onnx",
                    verbose=True,
                    opset_version=11,
                    input_names=input_names2,
                    output_names=output_names2,
                    dynamic_axes=dynamic_axes)

if __name__ == "__main__":
    main()
