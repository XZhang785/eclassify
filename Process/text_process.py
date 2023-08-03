import thulac
import pandas as pd
import re
import os
from functools import reduce
from typing import List, Optional


def text_replace(text: str) -> str:
    """
    对文本中的一些特殊目标进行替换（如ip，网址，标点符号，数字）

    Parameters:
    text: 需要被处理的文本

    Returns:
    替换后的文本
    """
    new_text = text

    # 替换ip
    ip_pattern = "\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    new_text = re.sub(ip_pattern, "网络地址", new_text)
    # 替换网址
    website_pattern = r'https?://\S+'
    new_text = re.sub(website_pattern, "网址", new_text)
    # 替换符号，数字，英文
    # symbol_pattern = r'[^A-Za-z\u4e00-\u9fa5]'
    symbol_pattern = r'[^\u4e00-\u9fa5]'
    new_text = re.sub(symbol_pattern, "", new_text)
    return new_text


def concat_data(path: str, tag_cols: List[str], sheet: Optional[str] = None,
                save: Optional[str] = None) -> pd.DataFrame:
    """
    将一整个文件夹内的excel数据进行提取，并且合并

    Parameters:
    -------------------------
    dir: 文件夹路径
    tag_cols: 要提取的所有列名
    sheet: Excel的工作表名，默认为空
    save: 最终数据的保存路径，默认为空（即不保存）
    
    Returns:
    --------------------------
    返回整合好的数据
    """

    files = os.listdir(path)
    dfs = []

    for file in files:
        flag = 1
        file_path = os.path.join(path, file)
        if sheet is None:
            df = pd.read_excel(file_path)
        else:
            df = pd.read_excel(file_path, sheet_name=sheet)
        columns = list(df.columns)
        for col in tag_cols:
            if col not in columns:
                flag = 0
                break
        if flag == 0:
            print(file + " is not satisfy")
        else:
            dfs.append(df[tag_cols])
            print("done " + file)
    df = reduce(lambda x, y: pd.concat([x, y], axis=0), dfs)
    print("concat success!")
    if save is not None:
        df.to_excel(save, index=False)

    return df


def text_segmentation(text: str) -> str:
    """
    对文本进行分词，并进行初步的过滤

    Parameters:
    --------------------
    text: 为分词的文本文本

    Returns:
    --------------------
    分词后的文本
    """

    thu = thulac.thulac(seg_only=True)  # 进行分词
    new_text = thu.cut(text, text=True)
    return new_text


def df_text_segmentation(df: pd.DataFrame, tag_col: str) -> pd.DataFrame:
    """
    对数据进行分词，并且进行初步的过滤
    
    Parameters:
    --------------------------
    df: 要被处理的数据
    tag_col: 需要切分的目标列
    
    Returns:
    --------------------------
    返回处理好的数据(以空格连接分词的文本)
    """

    thu = thulac.thulac(seg_only=True)  # 进行分词
    df[tag_col] = df[tag_col].apply(lambda x: thu.cut(x, text=True))
    return df


def remove_text_stopwords(text: str, custom_stopwords: set) -> str:
    """
    去除文本的停用词
    
    Parameters:
    --------------------------
    text: 分好词的文本（空格为间隔）
    custom_stopwords: 停用词表
    """
    words = text.split()
    filtered_words = [word for word in words if word not in custom_stopwords]
    return " ".join(filtered_words)


def load_custom_stopwords(file_path: str) -> set:
    """
    加载停用词表
    
    Parameters:
    --------------------------
    file_path: 停用词库的文件路径
    
    Returns:
    --------------------------
    返回停用词表
    """
    with open(file_path, "r", encoding="utf-8") as file:
        custom_stopwords = file.read().splitlines()
    return set(custom_stopwords)


def remove_df_stopwords(df: pd.DataFrame, file_path: str, tag_col: str) -> pd.DataFrame:
    """
    去除数据的停用词
    
    Parameters:
    --------------------------
    df: 分好词的数据(空格间隔)
    file_path: 停用词库的文件路径
    tag_col: 需要切分的目标列
    
    Returns:
    --------------------------
    返回处理好的数据
    """
    custom_stopwords = load_custom_stopwords(file_path)
    df[tag_col] = df[tag_col].apply(lambda x: remove_text_stopwords(x, custom_stopwords))
    return df


def df_text_process(df: pd.DataFrame, stopword_path: str, tag_col: str) -> pd.DataFrame:
    """
    对文本数据进行预处理

    Parameters:
    --------------------------
    df: 文本数据
    file_path: 停用词库的文件路径
    tag_col: 需要切分的目标列
    
    Returns:
    --------------------------
    返回处理好的数据
    """
    df = df.dropna(axis=0)
    df = df.drop_duplicates()
    df[tag_col] = df[tag_col].apply(lambda x: text_replace(x))
    df = df_text_segmentation(df, tag_col)
    df = remove_df_stopwords(df, stopword_path, tag_col)
    return df


def text_process(text: str, file_path: str) -> str:
    """
    对文本数进行预处理
    Parameters:
    ---------------------------
    text: 待处理的文本
    file_path: 停用词库的文件路径

    Returns:
    ---------------------------
    处理好的文本
    """
    new_text = text_replace(text)
    new_text = text_segmentation(new_text)
    custom_stopwords = load_custom_stopwords(file_path)
    new_text = remove_text_stopwords(new_text, custom_stopwords)
    return new_text
