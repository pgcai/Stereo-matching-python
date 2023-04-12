import re
import numpy as np
import sys
import os
 

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    # header = header[2:4]
    # header = bytes(header)
    # print(header)
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def writePFM(file, array):
    # assert type(file) is str and type(array) is np.ndarray and \
    #        os.path.splitext(file)[1] == ".pfm"
    with open(file, 'wb') as f:
        H, W = array.shape
        headers = ["Pf\n", f"{W} {H}\n", "-1\n"]
        for header in headers:
            f.write(str.encode(header))
        array = np.flip(array, axis=0).astype(np.float32)
        f.write(array.tobytes())

if __name__=='__main__':
    import cv2
    from tqdm import tqdm
    print('ReadPFM!')
    # dirapth = '/media/ip/data/caipuguang/dataset/zkhy_yuv/960/disparity/TRAIN/left'
    # dirapth = '/home/caipuguang/dataset/stereo/1920/disparity_cstr/TRAIN/left'
    # dirapth = '/edata/caipuguang/data/png/disparity/TRAIN/left'
    dirapth = '/hdata/caipuguang/dataset/stereo/kitti_format/disparity/TRAIN/left'


    def get_all_files_path(directory):
        file_paths = []
        for root, directories, files in os.walk(directory):
            for filename in files:
                file_path = os.path.join(root, filename)
                file_paths.append(file_path)
        return file_paths


    def depth_to_color(depth_map):
        # Normalize the depth values to the range [0, 255]
        # depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map *= 2
        depth_map = depth_map.astype(np.uint8)
        # Apply a colormap to the depth map
        # depth_color1 = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
        depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_TURBO)
        return depth_color

    pfmlist = get_all_files_path(dirapth)
    for i in tqdm(pfmlist):
        savepath = i.replace('disparity', 'disparity_color').replace('.pfm', '.jpg')
        if os.path.isfile(savepath):
            print("Conver OK, Continue!")
            continue
        dsp = readPFM(i)
        disp = dsp[0]
        # writePFM(i, img)
        disp_color = depth_to_color(disp)

        # imgpath = i.replace('disparity', 'frames_finalpass').replace('.pfm', '.png')
        # img = cv2.imread(imgpath)
        # disp_color = disp_color[15:-15]
        # disp_color = np.vstack((img,disp_color))
        # ----save color
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        cv2.imwrite(savepath, disp_color)
        # pass


    print('ReadPFM End!')

