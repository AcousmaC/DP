import argparse, json
import datetime
import os
import sys
import numpy as np
import logging
import torch, random
from server import *
from client import *
from util.textToPic import makePicOne
from util.textOption import *
from util import createTime
from util import printMsg
import models, datasets

# 输出重定向
def resultMsg(T):
    current_file_path = os.path.dirname(__file__)
    result_dir = os.path.join(os.path.abspath(os.path.join(current_file_path, './result')),f'{T}{conf["fileName"]}.out')
    sys.stdout = open(result_dir, "a", buffering=1)
    sys.stderr = sys.stdout
    return

# 动态定义输出文件名称
def getConfFileName(conf):
    type_map = {"CIFAR-10": "CF","MNIST": "M","Fashion-MNIST": "FM","CIFAR-100": "CF100",}
    data_type = type_map.get(conf['type'], conf['type'])
    lambda_str = str(conf['lambda']).replace('.', '')
    lr_str = str(conf['lr']).replace('.', '')
    file_name = f"{data_type}-{conf['no_models']}-{conf['k']}-{conf['local_epochs']}-{conf['algorithm']}-{lambda_str}-{lr_str}-eps{conf['epsilon']}-{conf['model_name']}-B{conf['batch_size']}-T{conf['global_epochs']}-{conf['priority']}"
    return file_name

# 检测已存在模型
def checkContinue(T, model_dir):
    # 基本记录数据
    Acc, Loss, noAccNumList = list(), list(), list()
    maxAcc, minLoss, currentAE, currentLE, noAccNum = 0, 100, 1, 1, 0
    # 读取路径
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # 读取已保存的模型文件
    read_save_path = os.path.join(model_dir, f'{conf["fileName"]}.pt')
    start_epoch = 0
    # 检测现存模型
    if os.path.exists(read_save_path):
        try:
            saveModelMsg = torch.load(read_save_path)
        except Exception as e:
            saveModelMsg = None
        # 如果文件损坏或找不到文件，按顺序搜索其他文件
        if saveModelMsg is None:
            # 获取文件名格式为 f'{conf["fileName"]}-{e}.pt'
            max_e = -1
            for e in range(1000):
                  alt_save_path = os.path.join(model_dir, f'{conf["fileName"]}-{e}.pt')
                  if os.path.exists(alt_save_path):
                     max_e = e
            # 如果找到最大的备份文件，加载它
            if max_e >= 0:
                  try:
                     alt_save_path = os.path.join(model_dir, f'{conf["fileName"]}-{max_e}.pt')
                     saveModelMsg = torch.load(alt_save_path)
                     print(f"Successfully loaded model from backup: {alt_save_path}")
                  except Exception as e:
                     print(f"Error loading backup model from {alt_save_path}: {e}")
            else:
                  print("No valid model files found in the directory.")
        server.global_model.load_state_dict(saveModelMsg['model_state_dict'])
        start_epoch = saveModelMsg['epoch'] + 1
        T = saveModelMsg['T']
        Acc, Loss = saveModelMsg['Acc'], saveModelMsg['Loss']
        maxAcc, minLoss, currentAE, currentLE = saveModelMsg['maxAcc'], saveModelMsg['minLoss'], saveModelMsg[
            'currentAE'], saveModelMsg['currentLE']
        noAccNum, noAccNumList = saveModelMsg['noAccNum'], saveModelMsg['noAccNumList']
        resultMsg(T)
        print(f"\n=====================Exit Model=====================\n")
    else:
        resultMsg(T)
        print(f"\n=====================Init Model=====================\n")
    return server.global_model, T, start_epoch, Acc, Loss, maxAcc, minLoss, currentAE, currentLE, noAccNum, noAccNumList


# 存储模型
def checkSave(model_dir, model, T, e, Acc, Loss, maxAcc, minLoss, currentAE, currentLE, noAccNum, noAccNumList):
    save_path = os.path.join(model_dir, f'{conf["fileName"]}-{e}.pt')
    save_path_now = os.path.join(model_dir, f'{conf["fileName"]}.pt')
    save_interval = 1
    if e % save_interval == 0:
        torch.save({
            'model_state_dict': server.global_model.state_dict(),
            'T': T,
            'epoch': e,
            'Acc': Acc,
            'Loss': Loss,
            'maxAcc': maxAcc,
            'minLoss': minLoss,
            'currentAE': currentAE,
            'currentLE': currentLE,
            'noAccNum': noAccNum,
            'noAccNumList': noAccNumList,
        }, save_path_now)
        torch.save({
            'epoch': e,
            'model_state_dict': server.global_model.state_dict(),
            'Acc': Acc,
            'Loss': Loss,
            'maxAcc': maxAcc,
            'minLoss': minLoss,
            'currentAE': currentAE,
            'currentLE': currentLE,
            'noAccNum': noAccNum,
            'noAccNumList': noAccNumList,
            'T': T,
        }, save_path)


def setChange(flag, a, changeA, e, changeE):
    if flag == 1:
        if (a > changeA):
            return a, e
        else:
            return changeA, changeE
    elif flag == 2:
        if (a < changeA):
            return a, e
        else:
            return changeA, changeE


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FL')
    parser.add_argument('-c', '--conf', dest='conf')
    parser.add_argument('--auf', dest='autoFile', action='store_true', help="Enable autoFile")
    parser.add_argument('--naf', dest='autoFile', action='store_false', help="no-autoFile")
    parser.set_defaults(autoFile=None)
    args = parser.parse_args()
    # 读取配置文件，更新 fileName 字段，并将结果写回文件
    with open(args.conf, 'r+') as f:
         conf = json.load(f)
         if args.autoFile is None:
            auto_file = conf.get('autoFile', False)
         else:
            auto_file = args.autoFile
         if auto_file:
            conf["fileName"] = getConfFileName(conf)
            f.seek(0)
            json.dump(conf, f, indent=4)
            f.truncate()
    train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])
    server = Server(conf, eval_datasets)
    clients = []
    for c in range(conf["no_models"]):
        clients.append(Client(conf, server.global_model, train_datasets, c))
    global_epochs = conf["global_epochs"]
    current_file_path = os.path.dirname(__file__)
    model_dir = os.path.join(os.path.abspath(os.path.join(current_file_path, './models')), f'{conf["fileName"]}')
    T = createTime.getTime()
    # 是否继续训练已保存的模型文件
    if conf['isContinue']:
        server.global_model, T, start_epoch, Acc, Loss, maxAcc, minLoss, currentAE, currentLE, noAccNum, noAccNumList = checkContinue(T, model_dir)
    # 全局迭代
    for e in range(start_epoch, conf["global_epochs"]):
        epochBeginTime = datetime.datetime.now()
        candidates = random.sample(clients, conf["k"])
        weight_accumulator = {}
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)
        local_i = 1
        print(f"\n\n{'=' * 50} Global round ({e})/({global_epochs})-th starts {'=' * 50}")
        print("Start polymerization")
        print(f"Weight:{conf['lambda']},Learning rate:{conf['lr']}")
        for client in candidates:
            print("+" * 200)
            print(f"Round ({e})/({global_epochs}) ->Training of the {'(' * 1}{local_i}/{conf['k']}{')' * 1}-th participant starts, participant name: {client}")
            # 输出精度
            printMsg.preMsg(currentAE=currentAE, maxAcc=maxAcc, currentLE=currentLE, minLoss=minLoss,
                            global_epochs=conf["global_epochs"])
            printMsg.fileMsg(T, "Acc", conf=conf)
            diff = client.local_train(server.global_model)
            print(f"Round ({e})/({global_epochs}) ->Training of the {'(' * 1}{local_i}/{conf['k']}{')' * 1}-th participant finish, participant name: {client}")
            print("+" * 200 +"\n")
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])
            local_i += 1
        # 聚合模型以及噪声
        server.model_aggregate(weight_accumulator)
        print("    End polymerization")
        # 精度损失计算并保存详细记录
        acc, loss = server.model_eval()
        maxAcc, currentAE = setChange(1, acc, maxAcc, e, currentAE)
        minLoss, currentLE = setChange(2, loss, minLoss, e, currentLE)
        
        if (acc == 10.0 and loss == None):
            noAccNum = noAccNum + 1
            noAccNumList.append(e)
        printMsg.endMsg(e, global_epochs, acc, loss, currentAE, maxAcc, currentLE, minLoss)
        Acc.append(acc)
        Loss.append(loss)
        if conf['isSave']:
            checkSave(model_dir, server.global_model.state_dict(), T, e, Acc, Loss, maxAcc, minLoss, currentAE,
                      currentLE, noAccNum, noAccNumList)
            if len(noAccNumList) > 0:
                print(f"\t\t*******\tAs of round {e}  Gradient explosion{noAccNum}th \t List of gradient explosion rounds:{noAccNumList}")

        textSaving('true', Acc, T, "Acc", conf=conf)
        textSaving('true', Loss, T, "Loss", conf=conf)
        addList = f"--------Max Acc at {currentAE} th  :{maxAcc}\n--------Min Loss at {currentLE} th :{minLoss}"
        textSaving('add', addList, T, "All", conf=conf)
        # 每轮结束计时
        printMsg.epochTime(epochBeginTime, e, global_epochs)
        print(f"=====================The model for round {e} has been saved.=====================")
    # 结束保存数据
    textSaving('false', Acc, T, "Acc", conf=conf)
    textSaving('false', Loss, T, "Loss", conf=conf)
    x = np.arange(0, conf["global_epochs"], 1)
    # 绘图
    makePicOne(x, Acc, 'Acc Figure', T, "Acc", conf=conf)
    makePicOne(x, Loss, 'Loss Figure', T, "Loss", conf=conf)
    # 重定向结束
    sys.stdout.close()
    sys.stderr.close()
    # 重置stdout和stderr到默认值
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__