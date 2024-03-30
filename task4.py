# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os


if __name__ == '__main__':
    path = "data2"  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    df_total = pd.DataFrame()
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            if file.endswith('.csv'):
                df = pd.read_csv("data2/" + file)
                df_total = pd.concat([df_total, df])

    print(len(df_total))


    def add_labels(dataframe, column_name, num_bins):
        # 检查数据框是否为空
        if dataframe.empty:
            print("DataFrame is empty.")
            return None

        # 检查指定的列是否存在
        if column_name not in dataframe.columns:
            print(f"Column '{column_name}' does not exist in the DataFrame.")
            return None

        # 获取列数据
        column_data = dataframe[column_name]

        # 计算每个档位的数量
        bin_size = len(column_data) // num_bins

        # 根据列数据进行排序
        sorted_data = column_data.sort_values(ascending=False)

        # 创建标签列表
        labels = []
        current_label = num_bins

        # 分配标签
        for i in range(len(sorted_data)):
            if i % bin_size == 0 and i != 0:
                current_label -= 1
            labels.append(current_label)

        # 将标签添加到数据框中
        dataframe['label'] = labels

        return dataframe

    df_new = add_labels(df_total, "10C", 20)

    print(len(df_new))


