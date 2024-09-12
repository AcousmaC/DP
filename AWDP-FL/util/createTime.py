import time

def getTime():
    fileName = time.localtime()
    createTime = fileName.tm_mon, fileName.tm_mday, fileName.tm_hour, fileName.tm_min, fileName.tm_sec
    return createTime