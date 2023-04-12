
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import pytorch_ssim

SSIM_WIN = 5

def warp_disp(x, disp, args):
    # result + flow(-disp) = x
    # warp back to result
    N, _, H, W = x.shape

    x_ = torch.arange(W).view(1, -1).expand(H, -1)
    y_ = torch.arange(H).view(-1, 1).expand(-1, W)
    grid = torch.stack([x_, y_], dim=0).float()
    if args.cuda:
        grid = grid.cuda()
    grid = grid.unsqueeze(0).expand(N, -1, -1, -1).clone()
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (W - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (H - 1) - 1
    # disp = 30*torch.ones(N, H, W).cuda()
    grid2 = grid.clone()
    grid2[:, 0, :, :] = grid[:, 0, :, :] + 2 * disp / W
    grid2 = grid2.permute(0, 2, 3, 1)

    return F.grid_sample(x, grid2, padding_mode='zeros')


def criterion1_2frame(imgC, imgR, outputR, maxdisp, args, down_factor=1):
    if down_factor != 1:
        imgC = F.interpolate(imgC, scale_factor=1.0 / down_factor, mode='bicubic')
        imgR = F.interpolate(imgR, scale_factor=1.0 / down_factor, mode='bicubic')
        outputR = F.interpolate(outputR.unsqueeze(1), scale_factor=1.0 / down_factor, mode='bicubic') / down_factor

        outputR = outputR.squeeze(1)

    imgR2C = warp_disp(imgR, -outputR, args)
    # imgR2C[0].cpu().detach().numpy().transpose(1,2,0)

    alpha2 = 0.85
    crop_edge = 0
    if imgC.shape[2] > SSIM_WIN:
        ssim_loss = pytorch_ssim.SSIM(window_size=SSIM_WIN)
    else:
        ssim_loss = pytorch_ssim.SSIM(window_size=imgC.shape[2])

    if crop_edge == 0:
        diff_ssim = (1 - ssim_loss(imgC, imgR2C)) / 2.0
        diff_L1 = (F.smooth_l1_loss(imgC, imgR2C, reduction='mean'))
    else:
        diff_ssim = (1 - ssim_loss(imgC[:, :, :, crop_edge:], imgR2C[:, :, :, crop_edge:])) / 2.0
        diff_L1 = (F.smooth_l1_loss(imgC[:, :, :, crop_edge:], imgR2C[:, :, :, crop_edge:], reduction='mean'))

    loss1 = (alpha2 * diff_ssim + (1 - alpha2) * diff_L1)

    return loss1, imgR2C

# loss3
# smooth loss: force grident of intensity to be small
def criterion3(disp, img):
    disp = disp.unsqueeze(1)
    disp_gx, disp_gy = gradient_xy(disp)
    intensity_gx, intensity_gy = gradient_xy(img)

    weights_x = torch.exp(-10 * torch.abs(intensity_gx).mean(1).unsqueeze(1))
    weights_y = torch.exp(-10 * torch.abs(intensity_gy).mean(1).unsqueeze(1))

    disp_gx = torch.abs(disp_gx)
    gx = disp_gx.clone()
    gx[gx>0.5] = disp_gx[disp_gx>0.5] + 10

    disp_gy = torch.abs(disp_gy)
    gy = disp_gy.clone()
    gy[gy>0.5] = disp_gy[disp_gy>0.5] + 10

    smoothness_x = gx * weights_x
    smoothness_y = gy * weights_y

    return smoothness_x.mean() + smoothness_y.mean()


def gradient_xy(img):
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]

    return gx, gy