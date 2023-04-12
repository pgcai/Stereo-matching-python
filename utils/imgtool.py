"""
图像相关处理方法
"""
import copy
import os

import numpy as np
import cv2
from PIL import Image
import zkhy_common_function.dirtool



def cutout(img, loc=(0, 0), width=0, Length=0):
    """
    从img中扣出目标部分。
    :param img:
    :param loc:目标部分左上角坐标(w,l)
    :param width:目标w
    :param Length:目标l
    :return:
    """
    res = img[loc[0]:loc[0] + width, loc[1]:loc[1] + Length]
    return res


def random_cutout(img, width=0, height=0):
    """
    随机抠图
    :param img:
    :param width:
    :param height:
    :return:
    """
    width_max, height_max, _ = img.shape
    width_range = width_max - width
    height_range = height_max - height
    w_rand = np.random.randint(0, width_range)
    h_rand = np.random.randint(0, height_range)
    return cutout(img, (w_rand, h_rand), width, height)


def video2img(video_path: str, save_path: str, img_extensions='jpg', frame_interval=1):
    """
    视频转图片
    """
    video = cv2.VideoCapture(video_path)
    read_count = 0
    save_count = 0
    frame = None

    if video.isOpened():
        rt, frame = video.read()
    else:
        rt = False
    while rt:
        if read_count % frame_interval == 0:
            save_count += 1
            cv2.imwrite(save_path + f'{save_count}' + img_extensions, frame)
        read_count += 1
        cv2.waitKey(1)
        rt, frame = video.read()
    video.release()
    print("Video to Image success:)")


def img2video(img_path: str, save_path: str, fps=25, img_extensions='jpg', only_root=True):
    """
    图像转视频
    :param img_path:
    :param save_path:
    :param fps:
    :param img_extensions:
    :param only_root:
    :return:
    """
    imglist = dirtool.get_file(img_path, (img_extensions), only_root)
    img = cv2.imread(imglist[0])
    size = (img.shape[1], img.shape[0])
    fourcc = cv2.VideoWriter_fourcc('X', '2', '6', '4')
    videowrite = cv2.VideoWriter(os.path.join(save_path, 'output.mp4'), fourcc, fps, size)
    videowrite.write(img)
    for i in imglist:
        img = cv2.imread(i)
        videowrite.write(img)
    print("Image to Video success:)")


def get_xyz(disp_file, img_W = 1280, img_H = 720, f_value = 1.4572434232064456e+03, bf_value = 1.7424560546875000e+02, optic_center_x = 6.3895917322353671e+02, optic_center_y = 3.4756185285151531e+02):
    """
    获取视差点
    """
    baseline = bf_value/f_value
    data = np.fromfile(disp_file, dtype=np.uint16)
    data = data.reshape(img_H, img_W) / 32.
    idx = data > 1.
    xs = np.zeros_like(data)
    ys = np.zeros_like(data)
    zs = np.zeros_like(data)
    offset_x = np.tile(np.arange(img_W) - optic_center_x, (img_H, 1))
    offset_y = np.tile(np.arange(img_H)[:, None] - optic_center_y, (1, img_W))
    xs[idx] = baseline * offset_x[idx] / data[idx]
    ys[idx] = baseline * offset_y[idx] / data[idx]
    zs[idx] = bf_value / data[idx]
    points = np.hstack((xs[:, None], ys[:, None], zs[:, None]))
    return points


def get_disp_img(disp_file, img_W = 1280, img_H = 720):
    """
    获取视差图
    """
    data = np.fromfile(disp_file, dtype=np.uint16)
    data = data.reshape(img_H, img_W) / 32.
    return data


def sp_noise_yuv(img, snr=0.99):
    """
    椒盐噪声+均值滤波\n
    parameter:\n
        img: yuv4:4:4 numrange:0-255\n
        snr: 信噪比\n
    return:\n
        img: yuv4:4:4 numrange:0-255
    """
    noiserate = 1-snr
    h, w, _ = img.shape
    mask = np.random.choice((0, 1, 2), size=(h, w), p=[snr, noiserate / 2., noiserate / 2.])
    jmask = mask==1
    ymask = mask==2
    img[:,:,0][jmask] = 255
    img[:,:,1:3][jmask] = 128
    img[:,:,0][ymask] = 0
    img[:,:,1:3][ymask] = 128
    img = cv2.blur(img, (2, 2))     # 均值滤波
    return img

def gasuss_noise(img, mean=0, var=0.005):
    """
    高斯噪声\n
    parameter:\n
        img: yuv4:4:4 numrange:0-255\n
        mean: 高斯分布的均值\n
        var: 高斯分布的标准差\n
    return:\n
        img: yuv4:4:4 numrange:0-255
    """
    img = img/255.
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    img += noise
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


# brightness + saturation + hue(亮度+饱和度+色调色相)
def img_brightness_saturation_hue(img, select=[0,0,0], brightness=0.5,saturation=1.5, hue_pixel=10):
    """
    brightness + saturation + hue(亮度+饱和度+色调色相)\n
    parameter:\n
        img: yuv4:4:4 numrange:0-255\n
    return:\n
        img: rgb numrange:0-255
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if select[0]==1:img_hsv[:, :, 2] = brightness * img_hsv[:, :, 2]
    if select[1]==1:img_hsv[:, :, 1] = saturation * img_hsv[:, :, 1]
    if select[2]==1:img_hsv[:, :, 0] = img_hsv[:, :, 0] + hue_pixel
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def img_contrast(img, contrast=0.5, threshold=0.5):
    """
    对比度调节
    parameter:\n
        img:RGB\BGR 0-255
    return img 0-1
    """
    img_out = img + (img - threshold * 255.0) * (contrast/(1-contrast))   
    img_out = img_out/255.0
    img_out[img_out>1.5]=1
    img_out[img_out<0]=0
    img_out = cv2.normalize(img_out,None,0,255,cv2.NORM_MINMAX)
    return np.array(img_out, dtype=np.uint8)


def img_gamma(img_original, gamma=0.7):
    """
    gamma变换
    parameter:
        img:RGB\BGR 0-255
        gamma: gamma矫正值
    return:
        img:rgb 0-255
    """
    img_original = np.array(img_original, np.uint8)
    gamma_result=[np.power(i/255.0,gamma) *255.0 for i in range(0,256)]
    gamma_result=np.round(gamma_result).astype(np.uint8)
    return cv2.LUT(img_original,gamma_result)


def dataaug(yuv, p=0.7,snr_range=(0.95, 1.0),var_range=(0.001, 0.01),contrast_range=(0.1,0.3),gamma_range=(0.6, 1.0),brightness_range=(0.7, 1.0),saturation_range=(0.8, 1.4)):
    """
    数据增强
    > 随机选取n种处理方式,处理的数值为合理范围内的随机值
    parameter:
        yuv:YUV444
        p: 做数据增强的概率
        snr_range:椒盐噪声信噪比范围
        var_range:高斯噪声标准差范围
        contrast_range:对比度调节范围
        gamma_range:gamma调节gamma值范围
        brightness_range:亮度调节范围
        saturation_range:饱和度调节范围
    return:
        YUV444
    """
    if np.random.random()>p:
        return yuv
    img = copy.deepcopy(yuv)
    select = np.random.randint(0,2,7)
    # select[椒盐噪声， 高斯噪声， 对比度调节， gamma调节， 亮度调节， 饱和度调节， 色度调节]
    if select[0]==1:
        snr = np.random.uniform(snr_range[0], snr_range[1]) # snr范围 0.95-0.999
        img = sp_noise_yuv(img, snr=snr)
    if select[1]==1:
        var = np.random.uniform(var_range[0], var_range[1]) # var范围0.001, 0.01
        img = gasuss_noise(img, mean=0, var=var)
    if sum(select[2:])>0:
        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        if select[2]==1:
            contrast = np.random.uniform(contrast_range[0], contrast_range[1])
            img = img_contrast(img, contrast=contrast, threshold=0.5)    # contrast范围0.1，0.3
        if select[3]==1:
            gamma = np.random.uniform(gamma_range[0], gamma_range[1]) # gamma范围0.6, 1.0
            img = img_gamma(img, gamma=gamma)
        if sum(select[4:])>0:
            brightness = np.random.uniform(brightness_range[0], brightness_range[1])    # brightness范围0.7, 1.0
            saturation = np.random.uniform(saturation_range[0], saturation_range[1])    # saturation范围0.8, 1.4
            hue_pixel = np.random.choice([-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10])   # huepixel范围
            img = img_brightness_saturation_hue(img, select[4:],brightness=brightness, saturation=saturation, hue_pixel=hue_pixel)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return img


# def

def yuv422_to_yuv444(img_path):
    """
    yuv422_to_yuv444
    :param img_path:
    :return:
    """
    raw = np.array(Image.open(img_path))
    h = raw.shape[0] // 2
    Y = raw[:h, ]
    uv = np.repeat(raw[h:, ], 2).reshape(h * 2, -1)
    U = uv[:h]
    V = uv[h:]
    img = np.array([Y, U, V]).transpose([1, 2, 0])
    return img

def yuv422_to_bgr(img_path):
    """
    yuv422_to_bgr
    :param img_path:
    :return:
    """
    raw = np.array(Image.open(img_path))
    h = raw.shape[0] // 2
    Y = raw[:h, ]
    uv = np.repeat(raw[h:, ], 2).reshape(h * 2, -1)
    U = uv[:h]
    V = uv[h:]
    img = np.array([Y, U, V]).transpose([1, 2, 0])
    return cv2.cvtColor(img, code=cv2.COLOR_YUV2BGR)
