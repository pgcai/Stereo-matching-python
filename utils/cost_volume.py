import torch
import numpy as np
import torch.nn as nn


def CostVolume(input_feature, candidate_feature, position="left", method="subtract", k=4, batch_size=4, channel=32, D=192, H=256, W=512):
    """
    Some parameters:
        position
            means whether the input feature img is left or right
        k
            the conv counts of the first stage, the feature extraction stage
    """
    origin = input_feature  # img shape : [batch_size, channel, H // 2**k, W // 2**k]
    candidate = candidate_feature
    """ if the input image is the left image, and needs to compare with the right candidate.
        Then it should move to left and pad in right"""
    if position == "left":
        leftMinusRightMove_List = []
        for disparity in range(D // 2**k):
            if disparity == 0:
                if method == "subtract":
                    """ subtract method"""
                    leftMinusRightMove = origin - candidate
                else:
                    """ concat mathod """
                    leftMinusRightMove = torch.cat((origin, candidate), 1)
                # leftMinusRightMove_norm = torch.norm(leftMinusRightMove, 1, 1)  # 1111
                leftMinusRightMove_List.append(leftMinusRightMove)
            else:
                zero_padding = np.zeros((origin.shape[0], channel, origin.shape[2], disparity))
                zero_padding = torch.from_numpy(zero_padding).float()
                zero_padding = zero_padding.cuda()

                left_move = torch.cat((origin, zero_padding), 3)

                if method == "subtract":
                    """ subtract method"""
                    leftMinusRightMove = left_move[:, :, :, :origin.shape[3]] - candidate
                else:
                    """ concat mathod """
                    leftMinusRightMove = torch.cat((left_move[:, :, :, :origin.shape[3]], candidate), 1)  # concat the channels
                # leftMinusRightMove_norm = torch.norm(leftMinusRightMove, 1, 1)
                leftMinusRightMove_List.append(leftMinusRightMove)
        cost_volume = torch.stack(leftMinusRightMove_List, dim=1)  # [batch_size, count(disparitys), channel, H, W]

        return cost_volume

import torch
import numpy as np


def CostVolume_2d(input_feature, candidate_feature, position="left", method="subtract", k=4, batch_size=4, channel=32, D=192, H=256, W=512):
    """
    Some parameters:
        position
            means whether the input feature img is left or right
        k
            the conv counts of the first stage, the feature extraction stage
    """
    origin = input_feature  # img shape : [batch_size, channel, H // 2**k, W // 2**k]
    candidate = candidate_feature
    """ if the input image is the left image, and needs to compare with the right candidate.
        Then it should move to left and pad in right"""
    if position == "left":
        leftMinusRightMove_List = []
        for disparity in range(D // 2**k):
            if disparity == 0:
                if method == "subtract":
                    """ subtract method"""
                    leftMinusRightMove = origin - candidate
                else:
                    """ concat mathod """
                    leftMinusRightMove = torch.cat((origin, candidate), 1)
                # leftMinusRightMove_norm = torch.norm(leftMinusRightMove, 1, 1)  # 1111
                # leftMinusRightMove_List.append(leftMinusRightMove_norm)
            else:
                zero_padding = np.zeros((origin.shape[0], channel, origin.shape[2], disparity))
                zero_padding = torch.from_numpy(zero_padding).float()
                zero_padding = zero_padding.cuda()

                # left_move = torch.cat((origin, zero_padding), 3)
                # left_move = torch.cat((zero_padding, origin), 3)
                right_move = torch.cat((zero_padding, candidate), 3)

                if method == "subtract":
                    """ subtract method"""
                    leftMinusRightMove = origin - right_move[:, :, :, :origin.shape[3]]
                else:
                    """ concat mathod """
                    raise ValueError("这儿是错的还没改")
                    leftMinusRightMove = torch.cat((origin[:, :, :, :origin.shape[3]], candidate), 1)  # concat the channels
            leftMinusRightMove_norm = torch.norm(leftMinusRightMove, 1, 1)  # leftMinusRightMove_norm=[batch_size, chl, H, W]
            leftMinusRightMove_List.append(leftMinusRightMove_norm)
        cost_volume = torch.stack(leftMinusRightMove_List, dim=1)  # [batch_size, count(disparitys), H, W]
        return cost_volume

def CostVolume_2d_v2(input_feature, candidate_feature, position="left", method="subtract", k=4, batch_size=4, channel=32, D=192, H=256, W=512):
    """
    Some parameters:
        position
            means whether the input feature img is left or right
        k
            the conv counts of the first stage, the feature extraction stage
    """
    origin = input_feature  # img shape : [batch_size, channel, H // 2**k, W // 2**k]
    candidate = candidate_feature
    """ if the input image is the left image, and needs to compare with the right candidate.
        Then it should move to left and pad in right"""
    if position == "left":
        leftMinusRightMove_List = []
        for disparity in range(D // 2**k):
            if disparity == 0:
                if method == "subtract":
                    """ subtract method"""
                    leftMinusRightMove = origin - candidate
                else:
                    """ concat mathod """
                    leftMinusRightMove = torch.cat((origin, candidate), 1)
                # leftMinusRightMove_norm = torch.norm(leftMinusRightMove, 1, 1)  # 1111
                # leftMinusRightMove_List.append(leftMinusRightMove_norm)
            else:
                zero_padding = np.zeros((origin.shape[0], channel, origin.shape[2], disparity))
                zero_padding = torch.from_numpy(zero_padding).float()
                zero_padding = zero_padding.cuda()

                # left_move = torch.cat((origin, zero_padding), 3)
                # left_move = torch.cat((zero_padding, origin), 3)
                right_move = torch.cat((zero_padding, candidate), 3)

                if method == "subtract":
                    """ subtract method"""
                    leftMinusRightMove = origin - right_move[:, :, :, :origin.shape[3]]
                else:
                    """ concat mathod """
                    raise ValueError("这儿是错的还没改")
                    leftMinusRightMove = torch.cat((origin[:, :, :, :origin.shape[3]], candidate), 1)  # concat the channels

            # leftMinusRightMove_norm = torch.norm(leftMinusRightMove, 1, 1)  # leftMinusRightMove_norm=[batch_size, chl, H, W]
            # v2 加绝对值
            leftMinusRightMove_norm = torch.abs(leftMinusRightMove)
            # leftMinusRightMove_norm = torch.nn.functional.normalize(leftMinusRightMove_norm, dim=1)
            leftMinusRightMove_norm = torch.mean(leftMinusRightMove_norm, dim=1)
            leftMinusRightMove_List.append(leftMinusRightMove_norm)
        cost_volume = torch.stack(leftMinusRightMove_List, dim=1)  # [batch_size, count(disparitys), H, W]
        return cost_volume


# new
# def CostVolume_2d(input_feature, candidate_feature, position="left", method="subtract", k=4, batch_size=4, channel=32, D=192, H=256, W=512):
#     """
#     Some parameters:
#         position
#             means whether the input feature img is left or right
#         k
#             the conv counts of the first stage, the feature extraction stage
#     """
#     origin = input_feature  # img shape : [batch_size, channel, H // 2**k, W // 2**k]
#     candidate = candidate_feature
#     b, c, h, w = origin.size()
#     leftMinusRightMove = origin.new_zeros(b, D // 2**k, h, w)
#     """ if the input image is the left image, and needs to compare with the right candidate.
#         Then it should move to left and pad in right"""
#     if position == "left":
#         leftMinusRightMove_List = []
#         for disparity in range(D // 2**k):
#             if disparity == 0:
#                 if method == "subtract":
#                     """ subtract method"""
#                     leftMinusRightMove[:, disparity, :, disparity:] = (origin - candidate).mean(dim=1)
#                 else:
#                     """ concat mathod """
#                     leftMinusRightMove = torch.cat((origin, candidate), 1)
#             else:
#
#                 if method == "subtract":
#                     """ subtract method"""
#                     leftMinusRightMove[:, disparity, :, disparity:] = (origin[:, :, :, disparity:] - candidate[:, :, :, :-disparity]).mean(dim=1)
#                 else:
#                     """ concat mathod """
#                     leftMinusRightMove = torch.cat((origin[:, :, :, disparity:], candidate[:, :, :, :-disparity]), 1)  # concat the channels
#         return leftMinusRightMove

class CostVolume_2d_v3(nn.Module):
    def __init__(self, k=4, D=192, channel=32):
        super(CostVolume_2d_v3, self).__init__()
        self.D = D
        self.k = k
        self.channel = channel

        self.filter_0 = Filter()
        self.filter_1 = Filter()
        self.filter_2 = Filter()
        self.filter_3 = Filter()
        self.filter_4 = Filter()
        self.filter_5 = Filter()
        self.filter_6 = Filter()
        self.filter_7 = Filter()
        self.filter_8 = Filter()
        self.filter_9 = Filter()
        self.filter_10 = Filter()
        self.filter_11 = Filter()
        self.filter_12 = Filter()
        self.filter_13 = Filter()
        self.filter_14 = Filter()
        self.filter_15 = Filter()
        self.filter_16 = Filter()
        self.filter_17 = Filter()
        self.filter_18 = Filter()
        self.filter_19 = Filter()
        self.filter_20 = Filter()
        self.filter_21 = Filter()
        self.filter_22 = Filter()
        self.filter_23 = Filter()


    def forward(self, input_feature, candidate_feature):
        origin = input_feature  # img shape : [batch_size, channel, H // 2**k, W // 2**k]
        candidate = candidate_feature

        disp_0 = self.filter_0(origin - candidate)
        disp_1 = self.filter_1(self.caculate(origin, candidate, 1))
        disp_2 = self.filter_2(self.caculate(origin, candidate, 2))
        disp_3 = self.filter_3(self.caculate(origin, candidate, 3))
        disp_4 = self.filter_4(self.caculate(origin, candidate, 4))
        disp_5 = self.filter_5(self.caculate(origin, candidate, 5))
        disp_6 = self.filter_6(self.caculate(origin, candidate, 6))
        disp_7 = self.filter_7(self.caculate(origin, candidate, 7))
        disp_8 = self.filter_8(self.caculate(origin, candidate, 8))
        disp_9 = self.filter_9(self.caculate(origin, candidate, 9))
        disp_10 = self.filter_10(self.caculate(origin, candidate, 10))
        disp_11 = self.filter_11(self.caculate(origin, candidate, 11))
        disp_12 = self.filter_12(self.caculate(origin, candidate, 12))
        disp_13 = self.filter_13(self.caculate(origin, candidate, 13))
        disp_14 = self.filter_14(self.caculate(origin, candidate, 14))
        disp_15 = self.filter_15(self.caculate(origin, candidate, 15))
        disp_16 = self.filter_16(self.caculate(origin, candidate, 16))
        disp_17 = self.filter_17(self.caculate(origin, candidate, 17))
        disp_18 = self.filter_18(self.caculate(origin, candidate, 18))
        disp_19 = self.filter_19(self.caculate(origin, candidate, 19))
        disp_20 = self.filter_20(self.caculate(origin, candidate, 20))
        disp_21 = self.filter_21(self.caculate(origin, candidate, 21))
        disp_22 = self.filter_22(self.caculate(origin, candidate, 22))
        disp_23 = self.filter_23(self.caculate(origin, candidate, 23))

        return torch.cat([disp_0,
                          disp_1,
                          disp_2,
                          disp_3,
                          disp_4,
                          disp_5,
                          disp_6,
                          disp_7,
                          disp_8,
                          disp_9,
                          disp_10,
                          disp_11,
                          disp_12,
                          disp_13,
                          disp_14,
                          disp_15,
                          disp_16,
                          disp_17,
                          disp_18,
                          disp_19,
                          disp_20,
                          disp_21,
                          disp_22,
                          disp_23],dim=1)

    def caculate(self, left, right, disp):
        zero_padding = np.zeros((left.shape[0], self.channel, left.shape[2], disp))
        zero_padding = torch.from_numpy(zero_padding).float()
        zero_padding = zero_padding.cuda()
        right_move = torch.cat((zero_padding, right), 3)
        leftMinusRightMove = left - right_move[:, :, :, :left.shape[3]]
        return leftMinusRightMove


class MetricBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride = 1):
        super(MetricBlock, self).__init__()
        # self.sq = nn.
        self.conv3d_1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv3d_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class Filter(nn.Module):
    def __init__(self):
        super(Filter, self).__init__()
        # self.sq = nn.
        self.filter = nn.Sequential(
            MetricBlock(32, 32),
            # MetricBlock(32, 32),
            MetricBlock(32, 32),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        return self.filter(x)

class CostVolume_2d_v4(nn.Module):
    def __init__(self, k=4, D=192, channel=32):
        super(CostVolume_2d_v3, self).__init__()
        self.D = D
        self.k = k
        self.channel = channel

        self.filter_0 = Filter()
        self.filter_1 = Filter()
        self.filter_2 = Filter()
        self.filter_3 = Filter()
        self.filter_4 = Filter()
        self.filter_5 = Filter()
        self.filter_6 = Filter()
        self.filter_7 = Filter()
        self.filter_8 = Filter()
        self.filter_9 = Filter()
        self.filter_10 = Filter()
        self.filter_11 = Filter()
        self.filter_12 = Filter()
        self.filter_13 = Filter()
        self.filter_14 = Filter()
        self.filter_15 = Filter()
        self.filter_16 = Filter()
        self.filter_17 = Filter()
        self.filter_18 = Filter()
        self.filter_19 = Filter()
        self.filter_20 = Filter()
        self.filter_21 = Filter()
        self.filter_22 = Filter()
        self.filter_23 = Filter()


    def forward(self, input_feature, candidate_feature):
        origin = input_feature  # img shape : [batch_size, channel, H // 2**k, W // 2**k]
        candidate = candidate_feature

        disp_0 = self.filter_0(origin - candidate)
        disp_1 = self.filter_1(self.caculate(origin, candidate, 1))
        disp_2 = self.filter_2(self.caculate(origin, candidate, 2))
        disp_3 = self.filter_3(self.caculate(origin, candidate, 3))
        disp_4 = self.filter_4(self.caculate(origin, candidate, 4))
        disp_5 = self.filter_5(self.caculate(origin, candidate, 5))
        disp_6 = self.filter_6(self.caculate(origin, candidate, 6))
        disp_7 = self.filter_7(self.caculate(origin, candidate, 7))
        disp_8 = self.filter_8(self.caculate(origin, candidate, 8))
        disp_9 = self.filter_9(self.caculate(origin, candidate, 9))
        disp_10 = self.filter_10(self.caculate(origin, candidate, 10))
        disp_11 = self.filter_11(self.caculate(origin, candidate, 11))
        disp_12 = self.filter_12(self.caculate(origin, candidate, 12))
        disp_13 = self.filter_13(self.caculate(origin, candidate, 13))
        disp_14 = self.filter_14(self.caculate(origin, candidate, 14))
        disp_15 = self.filter_15(self.caculate(origin, candidate, 15))
        disp_16 = self.filter_16(self.caculate(origin, candidate, 16))
        disp_17 = self.filter_17(self.caculate(origin, candidate, 17))
        disp_18 = self.filter_18(self.caculate(origin, candidate, 18))
        disp_19 = self.filter_19(self.caculate(origin, candidate, 19))
        disp_20 = self.filter_20(self.caculate(origin, candidate, 20))
        disp_21 = self.filter_21(self.caculate(origin, candidate, 21))
        disp_22 = self.filter_22(self.caculate(origin, candidate, 22))
        disp_23 = self.filter_23(self.caculate(origin, candidate, 23))

        return torch.cat([disp_0,
                          disp_1,
                          disp_2,
                          disp_3,
                          disp_4,
                          disp_5,
                          disp_6,
                          disp_7,
                          disp_8,
                          disp_9,
                          disp_10,
                          disp_11,
                          disp_12,
                          disp_13,
                          disp_14,
                          disp_15,
                          disp_16,
                          disp_17,
                          disp_18,
                          disp_19,
                          disp_20,
                          disp_21,
                          disp_22,
                          disp_23],dim=1)

    def caculate(self, left, right, disp):
        zero_padding = np.zeros((left.shape[0], self.channel, left.shape[2], disp))
        zero_padding = torch.from_numpy(zero_padding).float()
        zero_padding = zero_padding.cuda()
        right_move = torch.cat((zero_padding, right), 3)
        leftMinusRightMove = left - right_move[:, :, :, :left.shape[3]]
        return leftMinusRightMove