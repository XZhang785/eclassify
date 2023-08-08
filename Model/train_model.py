# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from keras import Sequential
from Process.text_process import df_text_process, recover_text
from Process.vector_process import vector_process, get_length_and_vocab_size, load_model_or_data
from Model.model import model_iter, plot_iter, split_data
from typing import Callable, Tuple
from keras.models import load_model
from os.path import join

DATA_PATH = "../Data/Origin_concat_data.xlsx"
STOPWORD_PATH = "../Data/cn_stopwords.txt"
LABEL_ENCODER_PATH = "../Data/label_encoder.pickle"
TOKENIZER_PATH = "../Data/tokenizer.pickle"
FEATURE_COL = "事件描述"
LABEL_COL = "支撑类型"
ERROR_COL = "预测类型"
EMBEDDING_DIM = 100


def get_train_test() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return x_train, x_test, y_train, y_test


def train_model(build_model: Callable[[int, int, int], Sequential], model_name: str) -> Sequential:
    """
    训练模型
    Parameters:
        build_model: 构造模型的方法
        model_name: 模型的名称

    Returns:
    训练好的模型
    """
    # 文件准备
    data_dir = "../Data"
    model_path = join(data_dir, model_name)
    picture_path = join(model_path, "train_metrics.png")
    # 获取训练集测试集
    x_train, x_test, y_train, y_test = get_train_test()
    # 训练模型
    max_length, vocab_size = get_length_and_vocab_size()
    model = build_model(vocab_size, EMBEDDING_DIM, max_length)
    # model_path = "../Data/dense_model"
    model, history = model_iter(model, x_train, y_train, model_path)
    # picture_path = "../Data/train_metrics.png"
    # 绘制图像
    plot_iter(history, picture_path)
    return model


def get_error_data(model_path: str, x: np.ndarray, y: np.ndarray, model_name: str) -> pd.DataFrame:
    """
    输出预测错误的数据，并保存
    Args:
        model_path: 要测试的模型
        x:  特征
        y: 标签
        model_name: 模型名称

    Returns:
    以DataFrame的格式返回错误数据的表
    """
    # 预测
    model = load_model(model_path)
    y_pre = model.predict(x)
    # 将数据转换为原始的数据
    label_encoder = load_model_or_data(LABEL_ENCODER_PATH)
    tokenizer = load_model_or_data(TOKENIZER_PATH)
    y_pre = label_encoder.inverse_transform(y_pre)
    y = label_encoder.inverse_transform(y)
    x = np.array(tokenizer.sequences_to_texts(x))
    # 整合数据
    error_df = pd.DataFrame()
    index = np.where(y != y_pre)
    error_df[FEATURE_COL] = x[index[0]]
    error_df[LABEL_COL] = y[index]
    error_df[ERROR_COL] = y_pre[index]
    error_df[FEATURE_COL] = error_df[FEATURE_COL].apply(lambda text: recover_text(text))
    print(error_df.head(5))
    # 保存
    error_data_dir = "../Data/error_data"
    file_name = model_name + ".xlsx"
    save_path = join(error_data_dir, file_name)
    error_df.to_excel(save_path, index=False)
    print("data successfully save in " + save_path)
    return error_df
