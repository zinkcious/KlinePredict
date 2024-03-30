import numpy as np
import pandas as pd
import os


def get_columns_all(df, num, index):
    try:
        if index >= num - 1:
            ma = np.mean(df.loc[index-num:index, :]['close'].values.tolist())

            mean_volume_n = np.mean(df.loc[index - num:index, :]['volume'].values.tolist())
            lb = df.loc[index, 'volume'] / mean_volume_n if mean_volume_n != 0 else None

            high_n = np.max(df.loc[index - num:index, :]['high'].values.tolist())
            low_n = np.min(df.loc[index - num:index, :]['low'].values.tolist())
            bdl = (high_n - low_n) / low_n / num if mean_volume_n != 0 else None

            zhengfu = (high_n - low_n) / low_n if mean_volume_n != 0 else None

            wzzb = (df.loc[index, 'close'] - low_n) / (high_n - low_n) if (high_n - low_n) != 0 else None
        else:
            ma = None
            lb = None
            bdl = None
            zhengfu = None
            wzzb = None
    except Exception as e:
        ma = None
        lb = None
        bdl = None
        zhengfu = None
        wzzb = None
    return (ma, lb, bdl, zhengfu, wzzb)


def add_columns_all(df_B):
    df_C = df_B.copy()
    for index, row in df_C.iterrows():
        if index % 100 == 0:
            print("正在处理第"+str(index)+"行")
        # 周期为5, 10, 20, 30, 60, 120, 240
        T = [5, 15, 30, 60, 120, 240]
        for t in T:
            (ma, lb, bdl, zhengfu, wzzb) = get_columns_all(df_C, t, index)
            df_C.loc[index, str(t) + 'MA'] = ma
            df_C.loc[index, str(t) + 'LB'] = lb
            df_C.loc[index, str(t) + 'BDL'] = bdl
            df_C.loc[index, str(t) + 'ZhengFu'] = zhengfu
            df_C.loc[index, str(t) + 'WZZB'] = wzzb

        if df_C.loc[index, 'datetime'].endswith('9:31:00'):
            tag = 'OM'
        elif df_C.loc[index, 'datetime'].endswith('14:58:00'):
            tag = 'CM2'
        elif df_C.loc[index, 'datetime'].endswith('14:59:00'):
            tag = 'CM1'
        elif df_C.loc[index, 'datetime'].endswith('15:00:00'):
            tag = 'CM0'
        else:
            tag = None
        df_C.loc[index, 'Tag'] = tag
        zhengfu1 = (df_C.loc[index, 'high'] - df_C.loc[index, 'low']) / df_C.loc[index, 'low'] if df_C.loc[index, 'low'] is not None else None
        df_C.loc[index, 'ZhengFu'] = zhengfu1
    return df_C


if __name__ == '__main__':

    path = "data1"  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    df_total = pd.DataFrame()
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            df_B = pd.read_csv("data1/" + file)
            df_C = add_columns_all(df_B)
            new_file_name = file.replace('B', 'C')
            df_C.to_csv('data3/' + new_file_name, index=False)
            print("----------" + new_file_name + "----------Done")
