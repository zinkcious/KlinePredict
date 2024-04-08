# -*- coding: utf-8 -*-

import pandas as pd
import time
import requests
import datetime
import numpy as np
import pandas as pd
import os
from pandas.errors import SettingWithCopyWarning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} 执行时间: {execution_time} 秒")
        return result
    return wrapper

def download_og_data(stocklist,datelist,path):
    stock_list = stocklist
    date_list = datelist
    pa = path
    errorFile = open('log.txt', 'a')
    for s in stock_list:
        df_A = pd.DataFrame(columns=["code", "datetime", "open", "close", "high", "low", "volume", "amount"])
        divide_len = len(date_list) // 4
        for i in range(divide_len):
            begin_time = date_list[i * 4]
            end_time = date_list[(i + 1) * 4]
            try:
                df = get_minute_line(s, begin_time, end_time)
                df_A = pd.concat([df_A, df])
            except Exception as e:
                errorFile.write(s)
                errorFile.write(begin_time)
                errorFile.write(end_time)
                errorFile.write(str(e))
                errorFile.write('\n\n')

        begin_time = date_list[divide_len * 4]
        end_time = '20231231'
        try:
            df = get_minute_line(s, begin_time, end_time)
            df_A = pd.concat([df_A, df])
        except Exception as e:
            errorFile.write(s)
            errorFile.write(begin_time)
            errorFile.write(end_time)
            errorFile.write(str(e))
            errorFile.write('\n\n')

        df_A.to_csv(pa + s + '_20230101_20231231_' 'A.csv', index=False)
        print("----------" + s + "----------Done")

    errorFile.close()

# 去除已生成的股票代码
def get_left_stock(path1, path2):
    stock_file_all = []
    stock_file = []
    files = os.listdir(path1)  # 得到文件夹下的所有文件名称
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            # 获取股票代码
            stock_code = file[:8]
            stock_file_all.append(stock_code)
    files = os.listdir(path2)  # 得到文件夹下的所有文件名称
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            # 获取股票代码
            stock_code = file[:8]
            stock_file.append(stock_code)
    stock_list_ = [s for s in stock_file_all if s not in stock_file]
    return stock_list_


def get_file_num(path1, path2):
    files1 = os.listdir(path1)
    num1 = len(files1)
    files2 = os.listdir(path2)
    num2 = len(files2)
    message = 'RUNNING-----' + str(num2) + '/' + str(num1)
    print(message)
    return message


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
        if index < size - num:
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

# 获取2023年1月1日到2023年12月31日所有交易日期
def get_time():

    url = "http://10.15.144.131/market/calendar?market=SH&begindate=20230101&enddata=20231231"

    resp = requests.get(url)
    result = resp.json()["Data"]["RepDataMarketTradeDate"][0]["TradeDate"]
    # print(result)

    # 取2023年1月1日到2023年12月31日数据
    date_list = []
    for d in result:
        if str.startswith(d, "2023"):
            date_list.append(d)
    # print(date_list)

    return date_list


# 获取沪深300和中证500的所有股票
def get_stock():

    url = "http://10.15.144.131/block/obj?gql=block=%E8%82%A1%E7%A5%A8/%E5%B8%82%E5%9C%BA%E5%88%86%E7%B1%BB/%E5%B8%B8%E7%94%A8%E6%8C%87%E6%95%B0%E6%88%90%E4%BB%BD/%E6%B2%AA%E6%B7%B1300%20or%20block=%E8%82%A1%E7%A5%A8/%E5%B8%82%E5%9C%BA%E5%88%86%E7%B1%BB/%E5%B8%B8%E7%94%A8%E6%8C%87%E6%95%B0%E6%88%90%E4%BB%BD/%E4%B8%AD%E8%AF%81500"

    resp = requests.get(url)
    result = resp.json()["Data"]["RepDataBlockObjOutput"][0]["obj"]

    stock_list = result

    return stock_list


# 获取股票四个交易日内的分时数据
@timer
def get_minute_line(stock, begin_time, end_time):

    url = "http://10.15.144.131/his/quote/kline"
    params = {
        "obj": stock,
        "period": "1min",
        "begin_time": begin_time,
        "end_time": end_time,
    }

    resp = requests.get(url=url, params=params)
    result = resp.json()["Data"]["RepDataQuoteKlineSingle"][0]["Data"]

    df = pd.DataFrame(columns=["code", "datetime", "open", "close", "high", "low", "volume", "amount"])

    for data in result:
        timeStamp = data["ShiJian"]
        dateArray = datetime.datetime.utcfromtimestamp(timeStamp)
        # utc加8小时
        dateArray_8 = dateArray+datetime.timedelta(hours=8)
        time = dateArray_8.strftime("%Y-%m-%d %H:%M:%S")
        line = [stock, time, data["KaiPanJia"], data["ShouPanJia"], data["ZuiGaoJia"], data["ZuiDiJia"],
                data["ChengJiaoLiang"], data["ChengJiaoE"]]
        df.loc[len(df)] = line
    return df


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
        if index % 1000 == 0:
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


# 把已有的数据库文件合并起来
@timer
def merge_df(pa):
    path = pa  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    df_total = pd.DataFrame()
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            if file.endswith('.csv'):
                df = pd.read_csv(pa + "/" + file)
                df_total = pd.concat([df_total, df])
    print("总数据量为%s" % len(df_total))
    return df_total


# 对已有的数据进行过滤
@timer
def filter_dataframe(dataframe, column_conditions):
    # 检查数据框是否为空
    if dataframe.empty:
        print("DataFrame is empty.")
        return None

    # 遍历列条件
    for column, condition in column_conditions.items():
        if column not in dataframe.columns:
            print(f"Column '{column}' does not exist in the DataFrame.")
            return None

        # 应用过滤条件
        dataframe = dataframe.loc[condition(dataframe[column])]
    print("条件过滤后的量为%s" % len(dataframe))
    return dataframe

@timer
def filter_merge(pa, column_conditions):
    path = pa  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    df_total = pd.DataFrame()
    count = 0
    count_empty = 0
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            if file.endswith('.csv'):
                df = pd.read_csv(pa + "/" + file)
                count = count + 1
                # 检查数据框是否为空
                if df.empty:
                    print("DataFrame is empty.")
                    count_empty = count_empty + 1
                    continue
                # 遍历列条件
                for column, condition in column_conditions.items():
                    if column not in df.columns:
                        print(f"Column '{column}' does not exist in the DataFrame.")
                        continue
                    # 应用过滤条件
                    df = df.loc[condition(df[column])]
                    df_total = pd.concat([df_total, df])
            print("已经完成%s个文件的过滤" % count)
    print("一共执行了%s个文件，其中有%s个文件为空" % (count, count_empty))
    print("条件过滤后的量为%s" % len(df_total))

    return df_total


# 给已有的数据通过区间涨幅打上标签
@timer
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
    print("标签已经打完，分为%s档" % num_bins)
    return dataframe


# 去除空行
@timer
def drop_na(df):
    dff = df.dropna()
    print("空行已经去除，剩下的数据量为%s" % len(dff))
    return dff


# 划分测试集和训练集
@timer
def split_dataframe(dataframe):
    # 使用 train_test_split 函数将 DataFrame 分割成训练集和测试集
    train_df, test_df = train_test_split(dataframe, test_size=0.3, random_state=42)
    print("训练集测试集划分完成，训练集大小为%s，测试集大小为%s" % (len(train_df), len(test_df)))
    return train_df, test_df


# 使用GradientBoostingClassifier进行训练
@timer
def model_ml(train_da, train_la, test_da, test_df, model_x):
    model_x.fit(train_da, train_la)
    predicted_gbc = model_x.predict(test_da)
    tt_gbc = test_df.copy()
    tt_gbc['label_p'] = predicted_gbc
    return tt_gbc


# 评分函数
@timer
def calculate_stats_for_label_pos(dataframe, label_values, column_name):
    # 检查数据框是否为空
    if dataframe.empty:
        print("DataFrame is empty.")
        return None, None

    # 检查列是否存在
    if column_name not in dataframe.columns:
        print(f"Column '{column_name}' does not exist in the DataFrame.")
        return None, None

    # 选择特定标签值的行
    selected_rows = dataframe[dataframe['label_p'].isin(label_values)]

    # 计算均值
    mean_value = round(selected_rows[column_name].mean() * 100, 4)

    # 计算大于0的比例
    positive_ratio = round((selected_rows[column_name] > 0).mean(), 4)

    print("指定的列在预测后平均上涨%s,上涨的比例为%s" % (mean_value, positive_ratio))

    return mean_value, positive_ratio

@timer
def calculate_stats_for_label_nag(dataframe, label_values, column_name):
    # 检查数据框是否为空
    if dataframe.empty:
        print("DataFrame is empty.")
        return None, None

    # 检查列是否存在
    if column_name not in dataframe.columns:
        print(f"Column '{column_name}' does not exist in the DataFrame.")
        return None, None

    # 选择特定标签值的行
    selected_rows = dataframe[dataframe['label_p'].isin(label_values)]

    # 计算均值
    mean_value = round(selected_rows[column_name].mean() * 100, 4)

    # 计算小于0的比例
    positive_ratio = round((selected_rows[column_name] < 0).mean(), 4)

    print("指定的列在预测后平均上涨%s,下跌的比例为%s"%(mean_value, positive_ratio))

    return mean_value, positive_ratio


