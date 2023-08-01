import pandas as pd

from Process.text_process import concat_data, text_replace, df_text_process, text_process


def concat_demo():
    # dir_path = r"C:\Users\Xzhang\Desktop\移动课题\Data\省公司工单"
    dir_path = r"C:\Users\Xzhang\eclassfy\Data"
    tag_cols = ["事件描述", "支撑类型"]
    save_path = r"..\Data\Origin_concat_data.xlsx"
    sheet_name = "Sheet1"
    df = concat_data(dir_path, tag_cols, save=save_path)
    df = concat_data(dir_path, tag_cols, sheet_name, save_path)
    print(df.shape)


def test_text_replace():
    test_data = [
        "客户反馈弹性公网IP36.139.122.198主机网络异常，随机性出现网络波动。通过https://ping.chinaz.com/ ，通过多次通过超级ping查询国内节点，出现随机性地区延迟很高，时好时坏"
        , "客户反馈36.139.123.40云主机VNC无法连接，远程无法连接",
        "在重庆节点下，多台云主机出现ping超时，如36.133.109.229 36.133.114.246，mtr信息见附件",
        "36.139.122.198"]
    for text in test_data:
        new_text = text_replace(text)
        print(text + "\n--->\n" + new_text)
        print("\n")


def test_df_process():
    filepath = r"..\Data\Origin_concat_data.xlsx"
    df = pd.read_excel(filepath)
    stopwords_file_path = r"C:\Users\Xzhang\Desktop\移动课题\stopwords\cn_stopwords.txt"
    tag_col = "事件描述"
    df = df_text_process(df, stopwords_file_path, tag_col)
    print(df.head(5))


def test_text_process():
    text_demo = "客户反馈36.139.123.40云主机VNC无法连接，远程无法连接"
    stopwords_file_path = r"C:\Users\Xzhang\Desktop\移动课题\stopwords\cn_stopwords.txt"
    print(text_process(text_demo, stopwords_file_path))


if __name__ == '__main__':
    test_text_replace()
