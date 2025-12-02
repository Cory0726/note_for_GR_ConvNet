import argparse
import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL import Image

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import plot_results, save_results


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--rgb_path', type=str, default='test_img/M1_08_intensity_grayscale_img.png', help='RGB Image path')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1, help='Use RGB image for evaluation (1/0)')  # 1
    # Number of grasp candidates to visualize
    parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
    # Whether to save results instead of only plotting
    parser.add_argument('--save', type=int, default=1, help='Save the results')
    # Force CPU mode even if GPU is available
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False, help='Force code to run in CPU mode')

    args = parser.parse_args()
    return args

def main(
        input_depth_path='test_img/M1_08_depth_refined.npy',
        network_path="trained-models/jacquard-d-grconvnet3-drop0-ch32/epoch_50_iou_0.94",
):
    args = parse_args()

    # Configure logging level
    logging.basicConfig(level=logging.INFO)
    logging.info('Loading depth...')
    # Load depth image
    depth_mm = np.load(input_depth_path)  # shape (H, W)
    depth = np.expand_dims(depth_mm, axis=2)  # shape (H, W, 1)
    print(f'Depth : {depth.shape}, {depth.dtype}, {depth.max()}, {depth.min()}')
    # Load pre-trained model
    logging.info('Loading model...')
    net = torch.load(network_path, weights_only=False)
    logging.info('Loading network complete')
    # Get the compute device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device: {}'.format(device))
    # Pre-process the depth data
    img_data = CameraData()
    x, depth_img = img_data.get_data(depth=depth)
    print(f'x : {x.shape}, {x.dtype}, {x.max()}, {x.min()}')
    print(f'depth_img : {depth_img.shape}, {depth_img.dtype}, {depth_img.max()}, {depth_img.min()}')
    # Predict the grasp pose by GR-ConvNet model
    with torch.no_grad():
        xc = x.to(device)
        pred = net.predict(xc)

        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

        # if args.save:
        #     save_results(
        #         rgb_img=img_data.get_rgb(fake_rgb, False),
        #         depth_img=np.squeeze(img_data.get_depth(depth)),
        #         grasp_q_img=q_img,
        #         grasp_angle_img=ang_img,
        #         no_grasps=args.n_grasps,
        #         grasp_width_img=width_img
        #     )
        # else:
        #     fig = plt.figure(figsize=(10, 10))
        #     plot_results(fig=fig,
        #                  rgb_img=img_data.get_rgb(fake_rgb, False),
        #                  grasp_q_img=q_img,
        #                  grasp_angle_img=ang_img,
        #                  no_grasps=args.n_grasps,
        #                  grasp_width_img=width_img)
        #     fig.savefig('img_result.pdf')


if __name__ == '__main__':
    main()
