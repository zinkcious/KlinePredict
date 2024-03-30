import pandas as pd
import requests
import datetime


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


if __name__ == '__main__':
    date_list = get_time()
    stock_list = get_stock()

    # 获取分时数据
    # df = get_minute_line("SH601318", "20181129", "20181205")
    # print(df)

    # 写入异常信息
    errorFile = open('log.txt', 'a')

    for s in stock_list:
        df_A = pd.DataFrame(columns=["code", "datetime", "open", "close", "high", "low", "volume", "amount"])
        divide_len = len(date_list) // 4
        for i in range(divide_len):
            begin_time = date_list[i * 4]
            end_time = date_list[(i+1) * 4]
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

        df_A.to_csv('data/' + s + '_20230101_20231231_' 'A.csv', index=False)
        print("----------" + s + "----------Done")

    errorFile.close()