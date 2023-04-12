"""
文件夹相关处理方法

introduce:

check_path(filepath:str)->None # 检查地址合法性, 不存在则新建
"""
import os
from tqdm import tqdm


def check_path(filepath: str) -> None:
    """
    检查地址合法性
    不存在则新建
    :param filepath:地址路径
    :return:None
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def get_file(filepath: str, file_extensions=('png', 'jpg'), only_root=False):
    """
    获取文件夹下的指定后缀文件的路径
    :param filepath:
    :param file_extensions: all:所有文件
    :param only_root:只获取当前目录下，不获取目录的目录下
    :return:
    """
    res = []
    for root, dirs, files in os.walk(filepath):
        files.sort()
        for filename in files:
            if file_extensions == 'all':
                res.append(os.path.join(root, filename))
            else:
                if filename.lower().endswith(file_extensions):
                    res.append(os.path.join(root, filename))
        if only_root:
            return res
    return res


def get_dir(dirpath: str):
    """
    获取路径下所有文件夹名
    :param dirpath:
    :return:
    """
    for root, dirs, files in os.walk(dirpath):
        return dirs


def change_name(dirpath, file_end=('.png', '.jpg'), name_add='', name_replace=('', ''), only_root=False):
    """
    批量修改文件名
    :param dirpath:文件夹路径
    :param file_end:需修改文件后缀名
    :param name_add:名字后添加内容
    :param name_replace:名字内替换内容
    :param only_root:仅仅应用于根目录
    :return:
    """
    for i in file_end:
        if name_add != '':
            file_list = get_file(dirpath, i, only_root=only_root)
            for j in tqdm(file_list):
                os.rename(j, j[:-len(i)] + name_add + j[-len(i):])
        if name_replace != ('', ''):
            file_list = get_file(dirpath, i, only_root=only_root)
            for j in tqdm(file_list):
                os.rename(j, j.replace(name_replace[0], name_replace[1]))


def find_file(filelist, dirpath, file_extension_match=True, only_root=False):
    """
    根据filelist中的文件名, 在dirpath文件夹中寻找存在的文件, 返回这些文件的地址
    :param filelist: 文件名列表
    :param dirpath: 文件夹地址
    :param file_extension_match: 是否后缀匹配 否的话只匹配文件名
    :param only_root:
    :return:
    """
    dir_file_list = get_file(dirpath, file_extensions='all', only_root=only_root)
    res = []
    if file_extension_match:
        for file_path in dir_file_list:
            file_name = file_path.split('/')[-1]
            file_name = file_name.split('\\')[-1]     # 部分windows存在\\的情况
            if file_name in filelist:
                res.append(file_path)
    else:
        filelist_new = []
        for filename in filelist:
            filename = filename.split('.')[0]
            filelist_new.append(filename)

        for file_path in dir_file_list:
            file_name = file_path.split('/')[-1]
            file_name = file_name.split('\\')[-1]  # 部分windows存在\\的情况
            file_name = file_name.split('.')[0]

            if file_name in filelist_new:
                res.append(file_path)
    return res


