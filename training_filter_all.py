# -*- coding: utf-8 -*-

from pandas.errors import SettingWithCopyWarning
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
import tools as tl

if __name__ == '__main__':
    df1 = tl.filter_merge("data3", {'5ZhengFu': lambda x: x > 0.01, '5LB': lambda x: x > 1.5})
    df2 = tl.drop_na(df1)
    df2.to_csv('temple.csv', mode='w', index=False)
    print("缓存文件已经保存在temple.csv中")
