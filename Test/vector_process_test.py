# -*- coding: utf-8 -*-
from Process.vector_process import *
from Process.text_process import *


def test_df_process() -> pd.DataFrame:
    filepath = r"..\Data\Origin_concat_data.xlsx"
    df = pd.read_excel(filepath)
    stopwords_file_path = r"C:\Users\Xzhang\Desktop\移动课题\stopwords\cn_stopwords.txt"
    tag_col = "事件描述"
    df = df_text_process(df, stopwords_file_path, tag_col)
    print(df.head(5))
    return df


def test_vector_process():
    df = test_df_process()
    x = df["事件描述"]
    y = df["支撑类型"]
    x, y = vector_process(x, y, True)
    print(x.shape, y.shape)
    return x, y


