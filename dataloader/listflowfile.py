import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

    classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    image = [img for img in classes if img.find('frames_finalpass') > -1]
    disp  = [dsp for dsp in classes if dsp.find('disparity') > -1]
    # print(classes)
    # print(image)
    # print(disp)

    all_left_img=[]
    all_right_img=[]
    all_left_disp = []
    test_left_img=[]
    test_right_img=[]
    test_left_disp = []

    # ========================= monkaa =======================

    monkaa_path = filepath + [x for x in image if 'monkaa' in x][0]
    monkaa_disp = filepath + [x for x in disp if 'monkaa' in x][0]

    monkaa_dir  = os.listdir(monkaa_path)

    for dd in monkaa_dir:
        for im in os.listdir(monkaa_path+'/'+dd+'/left/'):
            if is_image_file(monkaa_path+'/'+dd+'/left/'+im):
                all_left_img.append(monkaa_path+'/'+dd+'/left/'+im)
                all_left_disp.append(monkaa_disp+'/'+dd+'/left/'+im.split(".")[0]+'.pfm')

        for im in os.listdir(monkaa_path+'/'+dd+'/right/'):
            if is_image_file(monkaa_path+'/'+dd+'/right/'+im):
                all_right_img.append(monkaa_path+'/'+dd+'/right/'+im)


    # ========================= flying =======================

    flying_path = filepath + [x for x in image if x == 'frames_cleanpass'][0]
    flying_disp = filepath + [x for x in disp if x == 'frames_disparity'][0]
    flying_dir = flying_path+'/TRAIN/'
    subdir = ['A','B','C']

    for ss in subdir:
        flying = os.listdir(flying_dir+ss)

        for ff in flying:
            imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
            for im in imm_l:
                if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
                    all_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)

                all_left_disp.append(flying_disp+'/TRAIN/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')

                if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
                    all_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)

    flying_dir = flying_path+'/TEST/'

    subdir = ['A','B','C']

    for ss in subdir:
        flying = os.listdir(flying_dir+ss)

        for ff in flying:
            imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
            for im in imm_l:
                if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
                    test_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)

                test_left_disp.append(flying_disp+'/TEST/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')

                if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
                    test_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)

    # ========================= driving =======================

    driving_dir = filepath + [x for x in image if 'driving' in x][0] + '/'
    driving_disp = filepath + [x for x in disp if 'driving' in x][0]

    subdir1 = ['35mm_focallength','15mm_focallength']
    subdir2 = ['scene_backwards','scene_forwards']
    subdir3 = ['fast','slow']

    for i in subdir1:
        for j in subdir2:
            for k in subdir3:
                imm_l = os.listdir(driving_dir+i+'/'+j+'/'+k+'/left/')
                for im in imm_l:
                    if is_image_file(driving_dir+i+'/'+j+'/'+k+'/left/'+im):
                        all_left_img.append(driving_dir+i+'/'+j+'/'+k+'/left/'+im)
                    all_left_disp.append(driving_disp+'/'+i+'/'+j+'/'+k+'/left/'+im.split(".")[0]+'.pfm')

                    if is_image_file(driving_dir+i+'/'+j+'/'+k+'/right/'+im):
                        all_right_img.append(driving_dir+i+'/'+j+'/'+k+'/right/'+im)

    # print("all_left_img")
    # print(all_left_img)
    # print("all_left_disp")
    # print(all_left_disp)
    # print("all_right_img")
    # print(all_right_img)

    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp


def dataloader2(filepath):
    all_left_img = get_all_files_path(os.path.join(filepath,'frames_finalpass', 'TRAIN', 'left'))
    all_right_img = get_all_files_path(os.path.join(filepath,'frames_finalpass', 'TRAIN', 'right'))
    all_left_disp = get_all_files_path(os.path.join(filepath,'disparity', 'TRAIN', 'left'))
    test_left_img = get_all_files_path(os.path.join(filepath,'frames_finalpass', 'TEST', 'left'))
    test_right_img = get_all_files_path(os.path.join(filepath,'frames_finalpass', 'TEST', 'right'))
    test_left_disp = get_all_files_path(os.path.join(filepath,'disparity', 'TEST', 'left'))
    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp

def dataloader_kitti2015(filepath):
    all_left_img = get_all_files_path_kitti(os.path.join(filepath,'training', 'image_2'))
    all_right_img = get_all_files_path_kitti(os.path.join(filepath,'training', 'image_3'))
    all_left_disp = get_all_files_path(os.path.join(filepath,'training', 'disp_noc_1'))
    test_left_img = get_all_files_path(os.path.join(filepath,'testing', 'image_2'))
    test_right_img = get_all_files_path(os.path.join(filepath,'testing', 'image_3'))
    # test_left_disp = get_all_files_path(os.path.join(filepath,'disparity', 'TEST', 'left'))
    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img    # , test_left_disp

def dataloader_zkhy_TEST(filepath):
    test_left_img = get_all_files_path(os.path.join(filepath,'frames_finalpass', 'TEST', 'left'))
    test_right_img = get_all_files_path(os.path.join(filepath,'frames_finalpass', 'TEST', 'right'))
    return test_left_img, test_right_img

def dataloader_zkhy_selfsupervision(filepath):
    test_left_img = get_all_files_path(os.path.join(filepath,'frames_finalpass', 'TRAIN', 'left'))
    test_right_img = get_all_files_path(os.path.join(filepath,'frames_finalpass', 'TRAIN', 'right'))
    return test_left_img, test_right_img

def dataloader_zkhy_supervision(filepath):
    all_left_img = get_all_files_path(os.path.join(filepath, 'frames_finalpass', 'TRAIN', 'left'))
    all_right_img = get_all_files_path(os.path.join(filepath, 'frames_finalpass', 'TRAIN', 'right'))
    all_left_disp = get_all_files_path(os.path.join(filepath, 'disparity', 'TRAIN', 'left'))

    test_left_img = get_all_files_path(os.path.join(filepath, 'frames_finalpass', 'TRAIN', 'left'))
    test_right_img = get_all_files_path(os.path.join(filepath, 'frames_finalpass', 'TRAIN', 'right'))
    test_left_disp = get_all_files_path(os.path.join(filepath, 'disparity', 'TRAIN', 'left'))
    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp

def dataloader_list(filepath):
    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = [],[],[],[],[],[]
    for f in filepath:
        all_left_img_now, all_right_img_now, all_left_disp_now, test_left_img_now, test_right_img_now, test_left_disp_now = dataloader2(f)

        all_left_img.extend(all_left_img_now)
        all_right_img.extend(all_right_img_now)
        all_left_disp.extend(all_left_disp_now)

        test_left_img.extend(test_left_img_now)
        test_right_img.extend(test_right_img_now)
        test_left_disp_now.extend(test_left_disp_now)

    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp

def get_all_files_path(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_paths.append(file_path)
    return file_paths

def get_all_files_path_kitti(directory):
    """
    摒弃11.png
    """
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if '11.png' in filename:
                continue
            file_path = os.path.join(root, filename)
            file_paths.append(file_path)
    return file_paths