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

def vis_heatmap(img:np.ndarray):
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    return img

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    # Number of grasp candidates to visualize
    parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
    # Whether to save results instead of only plotting
    parser.add_argument('--save', type=int, default=1, help='Save the results')

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
    # Pre-process the depth data to tensor for the model
    img_data = CameraData()
    x, depth_img = img_data.get_data(depth=depth)
    x = x.unsqueeze(0)  # Increase the dimension of batch
    print(f'depth_img : {depth_img.shape}, {depth_img.dtype}, {depth_img.max()}, {depth_img.min()}')
    print(f'x : {x.shape}, {x.dtype}, {x.max()}, {x.min()}')
    # Predict the grasp pose by GR-ConvNet model
    with torch.no_grad():
        xc = x.to(device)
        pred = net.predict(xc)

        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
        logging.info('Predict complete')
        cv2.imshow('depth', vis_heatmap(depth_mm))
        cv2.imshow('q_img', vis_heatmap(q_img))
        cv2.imshow('ang_img', vis_heatmap(ang_img))
        cv2.imshow('width_img', vis_heatmap(width_img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
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
