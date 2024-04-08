# -*- coding: utf-8 -*-

from pandas.errors import SettingWithCopyWarning
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
import tools as tl
import pandas as pd

if __name__ == '__main__':
    df2 = pd.read_csv('temple.csv')
    print("读取缓存文件成功！")
    df3 = tl.add_labels(df2, "10C", 10)
    train_df, test_df = tl.split_dataframe(df3)
    tlist = ["5LB", "5ZhengFu", "5WZZB", "5BDL", "15LB", "15ZhengFu","15WZZB","15BDL","30LB","30ZhengFu","30WZZB","30BDL","60LB","60ZhengFu","60WZZB","60BDL","120LB","120ZhengFu","120WZZB","120BDL","240LB","240ZhengFu","240WZZB","240BDL"]
    train_data = train_df[tlist]
    train_label = train_df["label"]
    test_data = test_df[tlist]
    model = GradientBoostingClassifier(n_estimators=8, learning_rate=0.3, subsample=1, max_depth=5, min_samples_split=100, min_samples_leaf=8, max_features=1000, random_state=42)
    t_p_df = tl.model_ml(train_data, train_label, test_data, test_df, model)
    tl.calculate_stats_for_label_pos(t_p_df, [8, 9], "10C")
    tl.calculate_stats_for_label_nag(t_p_df, [1, 2], "10C")
    print("模型训练结束")

