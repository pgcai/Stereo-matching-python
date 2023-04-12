"""
Author:pgcai
Date:20230209
zkhy
"""
from __future__ import print_function
import sys
sys.path.append('./')
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import listflowfile as lt
from dataloader import ZKHYLoader as DA
from models import *
from torch.optim import RMSprop
from tqdm import tqdm
import cv2

batchSize = 1
root_path = "/home/caipuguang/code/StereoNet"

def get_args():
    parser = argparse.ArgumentParser(description='StereoNet')
    parser.add_argument('--maxdisp', type=int ,default=128,
                        help='maxium disparity')
    parser.add_argument('--datapath', default='/media/ip/data/caipuguang/dataset/zkhy_limit/images_011',
                        help='datapath')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    # parser.add_argument('--loadmodel', default= None, help='load model')
    parser.add_argument('--loadmodel', default='/checkpoints', help='load model')
    parser.add_argument('--savemodel', default='/checkpoints',
                        help='save model')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

def init_model(args):
    # cost_volume_method = "concat"
    cost_volume_method = "subtract"
    model = stereonet(batch_size=batchSize, cost_volume_method=cost_volume_method)
    print("-- model using stereonet --")

    if args.cuda:
        model = nn.DataParallel(model)
        device = torch.device('cuda')
        model.to(device, non_blocking=True)

    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
    # optimizer = RMSprop(model.parameters(), lr=1e-3, weight_decay=0.0001)
    epoch_start = 0
    total_train_loss_save = 0

    if args.loadmodel is not None and os.path.exists(args.checkpoint_path):
        state_dict = torch.load(args.checkpoint_path)
        model.load_state_dict(state_dict['state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        epoch_start = state_dict['epoch']
        total_train_loss_save = state_dict['total_train_loss']
        print("-- checkpoint loaded --")
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=epoch_start)
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        print("-- no checkpoint --")

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    return model, optimizer, scheduler, epoch_start


def depth_to_color(depth_map):
    # Normalize the depth values to the range [0, 255]
    # depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map *= 2
    depth_map = depth_map.astype(np.uint8)
    # Apply a colormap to the depth map
    # depth_color1 = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_TURBO)
    return depth_color


def test(args, model, TestImgLoader):
    model.eval()
    tbar = tqdm(TestImgLoader)
    for batch_idx, (imgL, imgR) in enumerate(tbar):
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))
        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()

        with torch.no_grad():
            start_time = time.time()
            output3 = model(imgL,imgR)
            print("inference", time.time() - start_time)

        output_img = imgL[0].cpu().detach().numpy().transpose(1,2,0)[...,::-1]
        output_img = cv2.normalize(output_img, None, 0, 255, cv2.NORM_MINMAX)
        output = torch.squeeze(output3.data.cpu(),1)[:,4:,:]
        output_color = output[0].cpu().detach().numpy()
        output_color = depth_to_color(output_color) # [..., ::-1]
        show = np.vstack((output_img, output_color))

        cv2.imwrite(os.path.join(args.output_save_path,f"{batch_idx}.jpg"), show)

    return


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = get_args()
    args.checkpoint_path = "/home/caipuguang/code/weights/checkpoints4/checkpoint_sceneflow_956.tar"
    # args.checkpoint_path = root_path + "/checkpoints5/checkpoint_sceneflow_558.tar"
    # args.datapath = '/home/caipuguang/dataset/stereo/zkhy_limit/images_005'   # 27
    args.datapath = '/home/caipuguang/dataset/stereo/flyingthings3d_format'  # 27
    args.output_save_path = '/home/caipuguang/code/inference_output/'
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    test_left_img, test_right_img = lt.dataloader_zkhy_TEST(
        args.datapath)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.zkhyImageFloder(test_left_img, test_right_img, False),
        batch_size=batchSize, shuffle=False, num_workers=12, drop_last=False)

    model, optimizer, scheduler, epoch_start = init_model(args)

    # ------------- TEST ------------------------------------------------------------
    test(args, model, TestImgLoader)



if __name__ == '__main__':
    main()
