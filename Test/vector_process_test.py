# -*- coding: utf-8 -*-
from Process.vector_process import *
from Process.text_process import *

STOPWORDS_FILE_PATH = r"C:\Users\Xzhang\Desktop\移动课题\stopwords\cn_stopwords.txt"
TOKENIZER_PATH = "../Data/tokenizer.pickle"
MAX_LENGTH_PATH = "../Data/maxsize.pickle"

def test_df_process() -> pd.DataFrame:
    filepath = r"..\Data\Origin_concat_data.xlsx"
    df = pd.read_excel(filepath)
    tag_col = "事件描述"
    df = df_text_process(df, STOPWORDS_FILE_PATH, tag_col)
    print(df.head(5))
    return df


def test_vector_process():
    df = test_df_process()
    x = df["事件描述"]
    y = df["支撑类型"]
    x, y = vector_process(x, y)
    print(x.shape, y.shape)
    return x, y


def test_text2vec():
    text = "云数据库MySQL 应等保需求需添加密码复杂度和登录失败锁定"
    tokenizer = load_model_or_data(TOKENIZER_PATH)
    max_length = load_model_or_data(MAX_LENGTH_PATH)
    new_text = text_process(text, STOPWORDS_FILE_PATH)
    vec = text2vec(new_text, tokenizer, max_length)
    print(vec)
    print(vec.shape)
