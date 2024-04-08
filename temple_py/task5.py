import numpy as np
import pandas as pd
import os


if __name__ == '__main__':
    path = "../test_data"  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    df_total = pd.DataFrame()
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            df = pd.read_csv("test_data/" + file)
            df_total = pd.concat([df_total, df])

    print(len(df_total))