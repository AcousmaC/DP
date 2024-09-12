import matplotlib.pyplot as plt
import numpy as np
import os
import time

def makePicOne(x, y1,Tittle,Time=-1,Atr=-1,conf=None):
    t = time.localtime()
    print("time :")
    print(t.tm_hour, t.tm_min, t.tm_sec)
    # 设置支持中文的字体，'SimHei'
    plt.rcParams['font.family'] = 'SimHei'
    # 解决保存图像是负号'-'显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 6), dpi=300)  # 增加尺寸和分辨率
    plt.title(Tittle)
    plt.xlabel('number')
    plt.ylabel('accuracy')
    plt.grid(axis='y', linestyle='--') 
    plt.plot(x, y1, "red")
    # plt.plot(x, y1, "red",marker='o')
    current_file_path = os.path.dirname(__file__)
    # 定义目录的相对路径
    imgPath = os.path.abspath(os.path.join(current_file_path, '../adaptSave/img'))
    # 保存图片的文件名
    filename = f"{Time}-{conf['fileName']}-{Atr}-{conf['global_epochs']}.jpg"
    # 完整的文件路径
    file_path = os.path.join(imgPath, filename)
    # 保存图像
    plt.savefig(file_path)
    plt.show()
