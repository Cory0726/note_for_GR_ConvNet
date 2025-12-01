import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL import Image

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import plot_results, save_results

# Configure logging level
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str, default='epoch_50_iou_094.pth',help='Path to saved network to evaluate')
    parser.add_argument('--rgb_path', type=str, default='test_img/M1_08_intensity_image.png', help='RGB Image path')
    parser.add_argument('--depth_path', type=str, default='test_img/M1_08_depth_refined.npy', help='Depth Image path')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for evaluation (1/0)')  # 1
    # Number of grasp candidates to visualize
    parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
    # Whether to save results instead of only plotting
    parser.add_argument('--save', type=int, default=0, help='Save the results')
    # Force CPU mode even if GPU is available
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False, help='Force code to run in CPU mode')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Load image
    logging.info('Loading image...')
    # Load RGB image
    pic = Image.open(args.rgb_path, 'r')
    rgb = np.array(pic)
    # Load depth image
    depth_mm = np.load('test_img/M1_08_depth_refined.npy')  # shape (H, W)
    depth = np.expand_dims(depth_mm, axis=2)  # shape (H, W, 1)

    # Load Network
    logging.info('Loading model...')
    net = torch.load(args.network)
    logging.info('Done')

    # Get the compute device
    device = get_device(args.force_cpu)

    img_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)

    x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)

    with torch.no_grad():
        xc = x.to(device)
        pred = net.predict(xc)

        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

        if args.save:
            save_results(
                rgb_img=img_data.get_rgb(rgb, False),
                depth_img=np.squeeze(img_data.get_depth(depth)),
                grasp_q_img=q_img,
                grasp_angle_img=ang_img,
                no_grasps=args.n_grasps,
                grasp_width_img=width_img
            )
        else:
            fig = plt.figure(figsize=(10, 10))
            plot_results(fig=fig,
                         rgb_img=img_data.get_rgb(rgb, False),
                         grasp_q_img=q_img,
                         grasp_angle_img=ang_img,
                         no_grasps=args.n_grasps,
                         grasp_width_img=width_img)
            fig.savefig('img_result.pdf')


if __name__ == '__main__':
    main()
