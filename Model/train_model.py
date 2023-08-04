# -*- coding: utf-8 -*- 
import pandas as pd
from keras import Sequential
from Process.text_process import df_text_process
from Process.vector_process import vector_process, get_length_and_vocab_size
from Model.model import build_dense_model, model_iter, plot_iter, split_data


DATA_PATH = "../Data/Origin_concat_data.xlsx"
STOPWORD_PATH = "../Data/cn_stopwords.txt"
FEATURE_COL = "事件描述"
LABEL_COL = "支撑类型"
EMBEDDING_DIM = 100


def train_dense_model() -> Sequential:
    # 导入数据集
    df = pd.read_excel(DATA_PATH)
    # 数据预处理
    df = df_text_process(df, STOPWORD_PATH, FEATURE_COL)
    # 数据向量化
    x = df[FEATURE_COL]
    y = df[LABEL_COL]
    x, y = vector_process(x, y)
    # 划分训练集和测试集
    train_rate = 0.8
    x_train, x_test, y_train, y_test = split_data(train_rate, x, y)
    # 训练模型
    max_length, vocab_size = get_length_and_vocab_size()
    model = build_dense_model(vocab_size, EMBEDDING_DIM, max_length)
    model_path = "../Data/dense_model"
    model, history = model_iter(model, x_train, y_train, model_path)
    picture_path = "../Data/train_metrics.png"
    # 绘制图像
    plot_iter(history, picture_path)
    return model
