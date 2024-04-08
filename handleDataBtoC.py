import numpy as np
import pandas as pd
import os
import tools as tl


if __name__ == '__main__':
    path1 = 'data1'
    path2 = 'test_data'
    stock_list_ = tl.get_left_stock(path1, path2)
    path = "data1"  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    df_total = pd.DataFrame()
    csv_files = [file for file in files if file.endswith('.csv')]
    for file in csv_files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            stock_code = file[:8]
            if stock_code in stock_list_:
                df_B = pd.read_csv("data1/" + file)
                df_C = tl.add_columns_all(df_B)
                new_file_name = file.replace('B', 'C')
                df_C.to_csv('test_data/' + new_file_name, index=False)
                print("----------" + new_file_name + "----------Done")
                tl.get_file_num(path1, path2)
            else:
                print("already exits!")
