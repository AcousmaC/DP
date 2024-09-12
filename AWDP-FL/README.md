
```
一:文件介绍
client.py  # 客户端文件
client_dlg_attack.py  # 模拟攻击文件
datasets.py  # 数据集文件
main.py  #主函数 
models.py  # 模型文件
server.py  # 服务器文件
adaptSave  # 存储信息文件夹
conf  # 配置文件文件夹
data  # 数据集
models  # 保存临时模型文件夹
result  # 日志保存信息
util  # 工具文件夹

二:运行
# python -u main.py -c ./conf/CIFAR-10.json
需配置所使用的配置文件 -c

三:conf文件配置介绍
{
    "model_name": "CIFAR_CNN",  # 使用模型名称
    "type": "CIFAR-10",  # 使用数据集名称
    "algorithm": "AWDP",  # 使用算法名称
    "global_epochs": 50,  # 全局通信轮次
    "no_models": 50,  # 客户端总数
    "k": 30,  # 参与训练客户端
    "local_epochs": 5,  # 本地迭代次数
    "batch_size": 64,  # 训练批次大小
    "lambda": 0.01,  # 聚合权重
    "lr": 0.01,  # 学习率
    "fileName": "",  # 当前训练id
    "isContinue": "true",  # 是否继续训练已保存的模型
    "isSave": "true",  # 是否每次训练保存模型
    "autoFile": true,  # 是否自动生成训练id
    "dp": true,  # 是否使用差分隐私
    "priority": 1,  # 训练id权重,同一参数训练多次使用.
    "delta": 0.001,  # 差分隐私超参数
    "epsilon": 5,  # 隐私预算
    "eps": 0.04,  # 动量更新参数1
    "beta1": 0.9,  # 动量更新参数2
    "beta2": 0.999  # 动量更新参数3
}
```