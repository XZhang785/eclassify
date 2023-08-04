# 部署步骤

1. 进入项目文件夹内
2. 安装python依赖`pip3 install -r requirements.txt`
3. 安装一个 ASGI 服务器`pip3 install "uvicorn[standard]"`

# 运行

1. 进入项目文件内
2. 运行命令`uvicorn main:app --reload`