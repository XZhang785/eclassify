# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Embedding, Bidirectional, Concatenate, LSTM, Input
from keras.layers import Dropout, SpatialDropout1D, Reshape
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.convolutional import MaxPooling1D, MaxPooling2D
from keras.callbacks import History, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from typing import Tuple


def split_data(train_rate: float, x: np.ndarray, y: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    对数据进行切分并混淆
    Parameters:
        train_rate: 训练数据的比率
        x: 数据
        y: 标签

    Returns:

    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_rate, random_state=42)
    return x_train, x_test, y_train, y_test


def build_textcnn_model(vocab_size: int, embedding_dim: int, input_length: int) -> Sequential:
    inp = Input(shape=(input_length,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length)(inp)
    x = SpatialDropout1D(0.4)(x)
    x = Reshape((input_length, embedding_dim, 1))(x)
    filter_sizes = [1, 2, 4, 5]
    conv_list = []
    num_filters = 16
    for filter_size in filter_sizes:
        conv = Conv2D(num_filters, kernel_size=(filter_size, embedding_dim), kernel_initializer='normal',
                      activation='relu')(x)
        max_pool = MaxPooling2D(pool_size=(input_length - filter_size + 1, 1))(conv)
        conv_list.append(max_pool)
    z = Concatenate(axis=1)(conv_list)
    z = Flatten()(z)
    z = Dropout(0.4)(z)
    output = Dense(4, activation="sigmoid")(z)

    model = Model(inputs=inp, outputs=output)
    model.summary()
    return model


def build_lstm_model(vocab_size: int, embedding_dim: int, input_length: int) -> Sequential:
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length))
    model.add(SpatialDropout1D(0.2))
    model.add(Conv1D(filters=16, kernel_size=5, activation='relu'))
    model.add(Bidirectional(LSTM(16, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    model.summary()
    return model


def build_cnn_model(vocab_size: int, embedding_dim: int, input_length: int) -> Sequential:
    """
    构建全连接网络
    Parameters:
        vocab_size: 词袋大小
        embedding_dim: 嵌入的维度
        input_length: 输入的维度

    Returns:
    构建好的模型，并在命令行输出其参数
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length))
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.summary()
    return model


def model_iter(model: Sequential, x_train: np.ndarray, y_train: np.ndarray, path: str) -> Tuple[Sequential, History]:
    """
    训练模型并绘制出迭代过程
    Parameters:
        model: 模型
        x_train: 训练数据
        y_train: 训练标签
        path: 模型保存的路径
    Returns:
    返回模型
    """
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=3, factor=0.1, min_lr=0.0001, verbose=2)
    checkpoint = ModelCheckpoint(filepath=path, monitor='val_accuracy', verbose=1, save_best_only=True)
    history = model.fit(x_train, y_train, epochs=50, verbose=2, validation_split=0.1, callbacks=[reduce_lr, checkpoint])
    return model, history


def plot_iter(history: History, path: str) -> None:
    """
    绘制迭代过程中，训练集与验证集的准确率与误差
    Parameters:
        history: 迭代记录
        path: 绘制图片保存的位置

    Returns:

    """
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], 'bo', label='Training loss')
    plt.plot(history.history['val_loss'], color='b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], 'o', label='Accuracy', c='orange')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy', c='orange')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(path)
