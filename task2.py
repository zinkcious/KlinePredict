import numpy as np
import pandas as pd
import os


def get_period(df, num, index):
    df_n = df.loc[index+1:index+num, :]
    # print(df_n)
    # 这5根k线的最高价
    high_list = df_n['high'].values.tolist()
    high_max = np.max(high_list)
    # 第5根K线的收盘价
    close_n = df_n['close'].values.tolist()[-1]
    return high_max, close_n


def get_h_c(df, num, index, close, size):
    try:
        if index < size - 5:
            high_max, close_n = get_period(df, num, index)
            # print(high_max)
            # print(close_n)
            h_n = (high_max - close) / close
            c_n = (close_n - close) / close
        else:
            h_n = None
            c_n = None
    except Exception as e:
        h_n = None
        c_n = None
    return h_n, c_n


# H：（这5根k线的最高价-当前k线的收盘价）/当前K线的收盘价 - 1
# C：（第5根K线的收盘价-当前K线的收盘价）/当前K线的收盘价 - 1
def add_columns(df_A):
    df_B = df_A.copy()
    for index, row in df_B.iterrows():
        # 当前K线收盘价
        close = row['close']

        # 当前df长度
        size = len(df_B)

        # 周期为5, 10, 20, 30, 60, 120, 240
        T = [5, 10, 20, 30, 60, 120, 240]
        for t in T:
            h, c = get_h_c(df_B, t, index, close, size)
            df_B.loc[index, str(t)+'H'] = h
            df_B.loc[index, str(t)+'C'] = c
    return df_B


if __name__ == '__main__':
    # df_A = pd.read_csv('data/SH600000_20230101_20231231_A.csv')
    # df_B = add_columns(df_A)
    # df_B.to_csv('data2/SH600000_20230101_20231231_B.csv', index=False)

    # stock_list = ['SH600585', 'SH600588', 'SH600598', 'SH600600', 'SH600690',
    #               'SH600699', 'SH600732', 'SH600754', 'SH600760', 'SH600764']
    #
    # for stock in stock_list:
    #     df_A = pd.read_csv('data/' + stock + '_20230101_20231231_A.csv')
    #     df_B = add_columns(df_A)
    #     print("----------" + stock + "----------Done")
    #     df_B.to_csv('data2/' + stock + '_20230101_20231231_B.csv', index=False)

    path = "data"  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    df_total = pd.DataFrame()
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            df_A = pd.read_csv("data/" + file)
            df_B = add_columns(df_A)
            new_file_name = file.replace('A', 'B')
            df_B.to_csv('data1/' + new_file_name, index=False)
            print("----------" + file + "----------Done")
