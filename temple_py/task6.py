# -*- coding: utf-8 -*-

from pandas.errors import SettingWithCopyWarning
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
import tools as tl

if __name__ == '__main__':
    df_all = tl.timer(tl.merge_df("data3"))
    df1 = tl.filter_dataframe(df_all, {'5ZhengFu': lambda x: x > 0.01, '5LB': lambda x: x > 1.5})
    df2 = tl.drop_na(df1)
    df3 = tl.add_labels(df2, "10C", 10)
    train_df, test_df = tl.split_dataframe(df3)
    tlist = ["5LB", "5ZhengFu", "5WZZB", "5BDL", "15LB", "15ZhengFu","15WZZB","15BDL","30LB","30ZhengFu","30WZZB","30BDL","60LB","60ZhengFu","60WZZB","60BDL","120LB","120ZhengFu","120WZZB","120BDL","240LB","240ZhengFu","240WZZB","240BDL"]
    train_data = train_df[tlist]
    train_label = train_df["label"]
    test_data = test_df[tlist]
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    t_p_df = tl.model_ml(train_data, train_label, test_data, test_df, model)
    tl.calculate_stats_for_label_pos(t_p_df, [8, 9, 10], "10C")
    tl.calculate_stats_for_label_nag(t_p_df, [1, 2, 3], "10C")

    print("模型训练结束")

