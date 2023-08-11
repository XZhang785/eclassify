# -*- coding: utf-8 -*- 
from Model.model import *
from Process.text_process import *
from Process.vector_process import *
from Model.train_model import train_model, get_error_data, get_train_test

STOPWORD_FILE_PATH = "../Data/cn_stopwords.txt"
DATA_PATH = r"..\Data\Origin_concat_data.xlsx"


def test_df_process() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH)
    tag_col = "事件描述"
    df = df_text_process(df, STOPWORD_FILE_PATH, tag_col)
    print(df.head(5))
    return df


def test_vector_process():
    df = test_df_process()
    x = df["事件描述"]
    y = df["支撑类型"]
    x, y = vector_process(x, y)
    print(x.shape, y.shape)
    return x, y


def test_split():
    x, y = test_vector_process()
    train_rate = 0.8
    x_train, x_test, y_train, y_test = split_data(train_rate, x, y)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return x_train, x_test, y_train, y_test


def test_cnn_model():
    # 训练全连接层模型
    train_model(build_cnn_model, "dense_model")


def test_textcnn_model():
    train_model(build_textcnn_model, "textcnn_model")


def test_lstm_model():
    # 训练LSTM模型
    train_model(build_lstm_model, "lstm_model")


def test_get_error_data():
    model_path = "../Data/dense_model"
    model_name = "dense_model"
    x_train, x_test, y_train, y_test = get_train_test()
    get_error_data(model_path, x_test, y_test, model_name)
