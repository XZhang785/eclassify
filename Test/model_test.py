# -*- coding: utf-8 -*- 
from Model.model import *
from Process.text_process import *
from Process.vector_process import *


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


def test_split():
    x, y = test_vector_process()
    train_rate = 0.8
    x_train, x_test, y_train, y_test = split_data(train_rate, x, y)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return x_train, x_test, y_train, y_test


def test_model():
    max_length_path = "../Data/maxsize.pickle"
    tokenizer_path = "../Data/tokenizer.pickle"
    x_train, x_test, y_train, y_test = test_split()
    tokenizer = load_model(tokenizer_path)
    max_length = load_model(max_length_path)
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 100
    model = build_dense_model(vocab_size, embedding_dim, max_length)
    model_path = "../Data/dense_model.pickle"
    model, history = train_model(model, x_train, y_train, model_path)
    picture_path = "../Data/dense_model_metrics.png"
    plot_train(history, picture_path)
