# -*- coding: utf-8 -*- 
from fastapi import FastAPI
from keras.models import load_model
from Process import *

STOPWORDS_PATH = "Data/cn_stopwords.txt"
MODEL_PATH = "Data/textcnn_model"
TOKENIZER_PATH = "Data/tokenizer.pickle"
MAX_LENGTH_PATH = "Data/maxsize.pickle"
LABEL_ENCODER_PATH = "Data/label_encoder.pickle"
app = FastAPI()
model = load_model(MODEL_PATH)
tokenizer = load_model_or_data(TOKENIZER_PATH)
max_length = load_model_or_data(MAX_LENGTH_PATH)
label_encoder = load_model_or_data(LABEL_ENCODER_PATH)


@app.post("/classify")
def classify_text(text: str) -> str:
    """
    对问题文本进行分类
    Parameters:
        text: 问题文本

    Returns:
    分类标签
    """
    new_text = text_process(text, STOPWORDS_PATH)
    text_vec = text2vec(new_text, tokenizer, max_length)
    label_vec = model.predict(text_vec)
    label = label_encoder.inverse_transform(label_vec).reshape(1)[0]
    return label
