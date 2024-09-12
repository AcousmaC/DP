# -*- coding: utf-8 -*-
import datetime

def endMsg(e, global_epochs, acc, loss, currentAE, maxAcc, currentLE, minLoss):
    print(f"{'-' * 150}")
    print(f"--------The global ({e})/({global_epochs}) round ends, {'*' * 7}\t 【Acc】:{acc}, {'*' * 7}\t 【Loss】:{loss}")
    print(f"--------The max acc  is No {currentAE} round:{maxAcc}")
    print(f"--------The min loss is No {currentLE} round:{minLoss}")
    print(f"{'-' * 150}")

def preMsg(currentAE, maxAcc, currentLE, minLoss,global_epochs):
    print(f"\t"
          f"{'*' * 7}\tThe maximum acc  is No {currentAE}/{global_epochs} round:{maxAcc}"
          f"\t\t\t"
          f"{'*' * 7}\tThe maximum loss is No {currentLE}/{global_epochs} round:{minLoss}")

# save File
def fileMsg(Time=-1,Atr=-1,conf= None):
    print(f"\t"
          f"{'*' * 7}\tfileName:{Time}-{conf['fileName']}-{Atr}-{conf['global_epochs']}.txt"
          )

def epochTime(epochBeginTime, e, global_epochs):
    epochFinishTime = datetime.datetime.now()
    eTime = epochFinishTime - epochBeginTime
    label_width = 50
    print(f"{('Time taken for the current round ' + f'{e}/{global_epochs}').ljust(label_width)}: {eTime} \n"
          f"{'Time remaining until completion:'.ljust(label_width)}: {(global_epochs-e-1)*eTime} \n"
          f"{'Estimated completion time:'.ljust(label_width)}: {datetime.datetime.now() + (global_epochs-e-1)*eTime}")

