import time
import os
# 存储文件目录
current_file_path = os.path.dirname(__file__)
text_dir = os.path.abspath(os.path.join(current_file_path, '../adaptSave/text'))
# 每轮存储
textE_dir = os.path.abspath(os.path.join(current_file_path, '../adaptSave/text/textE'))
# 每轮存储记录
record_dir = os.path.abspath(os.path.join(current_file_path, '../adaptSave/record.text'))
# 保存数据
def textSaving(eCheck,list,Time=-1,Atr=-1,conf= None):
    file_name = f"{Time}-{conf['fileName']}-{Atr}-{conf['global_epochs']}.txt"
    if eCheck == 'true':
        file_path = os.path.join(textE_dir, 'E' + file_name)
    elif eCheck == 'false':
        file_path = os.path.join(text_dir, file_name)
    else:
        # 如果eCheck不是'true'或'false'，则记录到特定文件中
        file_path = record_dir
    # 根据eCheck的值执行不同的操作
    if eCheck in ['true', 'false']:
        with open(file_path, 'w') as f:
            f.write(str(list))
        print("Data saved successfully.")
    else:
        file_name_text_find = f"{Time}-{conf['fileName']}-{Atr}-{conf['global_epochs']}"
        with open(file_path, 'r',encoding='utf-8') as file:
            lines = file.readlines()
        foundIndex = -1
        for i, line in enumerate(lines):
            if file_name_text_find in line:
                foundIndex = i
                break
        addContent = f"\n{file_name_text_find}\n" \
                     f"{str(list)}"
        replaceContent = f"{file_name_text_find}\n" \
                     f"{str(list)}\n"
        # 新增
        if foundIndex == -1:
            lines.append(addContent)
        # 替换
        elif foundIndex != -1:
            lines[foundIndex:foundIndex+3] = [replaceContent]
        # 将修改后的内容写回文件
        with open(file_path, 'w') as file:
            file.writelines(lines)
        # print("记录成功")
        print("Record successful.")


# 读取数据
def textReading(fileUrlAndFileName):
    # print(fileUrlAndFileName)
    with open(
            fileUrlAndFileName,
            'r'
    ) as f:
        data_dict = f.read()
        # 将字符转换为dict格式
        data_dict = eval(data_dict)
        # 如果字典格式为：collections.defaultdict()格式
        # data_dict = eval(data_dict [28:-1])
        # print(data_dict)
        f.close()
        return data_dict

