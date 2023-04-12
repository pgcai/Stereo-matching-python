"""
操作txt文件的一些方法

模式    可做操作    若文件不存在    是否覆盖
r       只能读         报错         -
r+      可读可写        报错        是
w       只能写          创建        是
w+      可读可写        创建        是
a       只能写          创建        否，追加写
a+      可读可写        创建        否，追加写

"""
from zkhy_common_function import strtool
import numpy as np

def read_txt_to_array(filepath:str)->list:
    """
    读取txt文本内容,以列表形式返回
    """
    res = []
    f = open(filepath)
    line = f.readline()
    while line: 
        # print(line[:-1])
        alist = line[:-1].split()
        theline = []
        for i in alist:
            if strtool.is_real_number(i):
                i = np.float32(i)
            theline.append(i)
        if len(theline)>1:
            res.append(theline)
        else:
            res.extend(theline)
        line = f.readline()   
    f.close()
    return res

def write_array_2_txt(file_path, tList):
    """
    新建txt文件写入一个数组 "一个元素一行
    :param file_path:
    :param tList:
    :return:
    """

    tList = [str(i) + "\n" for i in tList]
    with open(file_path, 'w') as f:  # 设置文件对象
        f.writelines(tList)
    print("Write_array_2_txt ok!")


def txt_write_2d_array(file_path, tList):
    """
    新建txt文件写入一个2-dim列表
    """

    # tList = [str(i) + " " for i in tList]
    # tList += "\n"
    with open(file_path, 'w') as f:  # 设置文件对象
        for i in tList:
            theline = ''
            for j in i:
                theline += j
                theline += " "
            theline = theline[:-1] + '\n'
            f.writelines(theline)
    print("TxT write ok!")
