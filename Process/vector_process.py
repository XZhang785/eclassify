from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import OneHotEncoder
from keras_preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
import numpy as np
from typing import Tuple, Any


MAX_LENGTH_PATH = "../Data/maxsize.pickle"
TOKENIZER_PATH = "../Data/tokenizer.pickle"
LABEL_ENCODER_PATH = "../Data/label_encoder.pickle"
VOCAB_SIZE_PATH = "../Data/vocab_size.pickle"


def get_length_and_vocab_size() -> Tuple[int, int]:
    """
    加载训练数据最长长度和词袋长度
    Returns:
    训练数据最长长度和词袋长度
    """
    max_length = load_model_or_data(MAX_LENGTH_PATH)
    vocab_size = load_model_or_data(VOCAB_SIZE_PATH)
    return max_length, vocab_size


def cmp_max_size(ser: pd.Series) -> int:
    """
    计算最长序列的长度
    Parameters:
    ser:文本序列series

    Returns:
    最长序列的长度
    """
    max_length = ser.apply(lambda x: len(x.split())).max()
    with open(MAX_LENGTH_PATH, 'wb') as handle:
        pickle.dump(max_length, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return max_length


def train_tokenizer(ser: pd.Series) -> Tokenizer:
    """
    训练标记模型并保存
    Args:
    ser: 训练序列
    path: 模型保存地址

    Returns:
    返回标记模型
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(ser)
    vocab_size = len(tokenizer.word_index) + 1
    print("Vocabulary Size :", vocab_size)

    with open(VOCAB_SIZE_PATH, 'wb') as handle:
        pickle.dump(vocab_size, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(TOKENIZER_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return tokenizer


def load_model_or_data(path: str) -> Any:
    """
    加载模型或数据
    Parameters:
    path: 模型的存储路径

    Returns:
    返回模型
    """
    with open(path, 'rb') as handle:
        model = pickle.load(handle)
    return model


def transform_sequence(ser: pd.Series, tokenizer: Tokenizer, length: int) -> np.ndarray:
    """
    将文本序列转换为数值向量
    Parameters:
        ser: 文本序列
        tokenizer: 标记模型
        length: 向量长度

    Returns:
    返回处理好的序列
    """
    array = pad_sequences(tokenizer.texts_to_sequences(ser), maxlen=length, padding="post")
    return array


def train_label_encoder(labels: np.ndarray, path: str) -> OneHotEncoder:
    """

    Parameters:
        labels: 训练标签，必须是二维的
        path: 模型的保存的路径
    Returns:
    返回训练好的模型
    """

    encoder = OneHotEncoder(sparse=False)
    encoder.fit(labels)

    with open(path, 'wb') as handle:
        pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return encoder


def vector_process(x: pd.Series, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    对数据进行向量化处理
    Parameters:
        x: 经过预处理后的文本数据
        y: 标签数据

    Returns:
    返回向量化后的特征和标签
    """
    y = y.to_numpy().reshape(-1, 1)
    max_length = cmp_max_size(x)
    tokenizer = train_tokenizer(x)
    label_encoder = train_label_encoder(y, LABEL_ENCODER_PATH)
    x = transform_sequence(x, tokenizer, max_length)
    y = label_encoder.transform(y)
    return x, y


def text2vec(text: str, tokenizer: Tokenizer, max_length: int) -> np.ndarray:
    """
    将文本转为向量
    Parameters:
        text: 经过预处理后的文本
        tokenizer: 预训练好的标记模型
        max_length: 向量最大长度
    Returns:
    文本数据转化出的向量
    """

    x = pd.Series(text)
    x = transform_sequence(x, tokenizer, max_length)
    return x
