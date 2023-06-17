import glob2
import random

random.seed(2023)
data_path = '/home/FedTC/data/gatritis_data/'
path = ['*/IM/贲门/*.TIF','*/GA/贲门/*.TIF','*/Normal/贲门/*.TIF',
        '*/IM/胃底/*.TIF','*/GA/胃底/*.TIF','*/Normal/胃底/*.TIF',
        '*/IM/胃窦/*.TIF','*/GA/胃窦/*.TIF','*/Normal/胃窦/*.TIF',
        '*/IM/胃角/*.TIF','*/GA/胃角/*.TIF','*/Normal/胃角/*.TIF',
        '*/IM/胃体/*.TIF','*/GA/胃体/*.TIF','*/Normal/胃体/*.TIF',]
data = [[] for i in range(len(path))]
for i in range(len(path)):
    data[i]+= glob2.glob(data_path+path[i])
    random.shuffle(data[i])
d_IM, d_GA, d_Normal = [], [], []

for i in range(0,len(path),3):
    d_IM.append(data[i])
    d_GA.append(data[i+1])
    d_Normal.append(data[i+2])

data_IM = d_IM+d_Normal
data_GA = d_GA+d_Normal
IM_train,IM_test,IM_val = [],[],[]
for i in data_IM:
    IM_train.append(i[:int(len(i)*0.7)])
    IM_test.append(i[int(len(i)*0.7):int(len(i)*0.9)])
    IM_val.append(i[int(len(i)*0.9):])
GA_train,GA_test,GA_val = [],[],[]
for i in data_GA:
    GA_train.append(i[:int(len(i)*0.7)])
    GA_test.append(i[int(len(i)*0.7):int(len(i)*0.9)])
    GA_val.append(i[int(len(i)*0.9):])
# label dict
IM_train_dict,IM_test_dict,IM_val_dict = {},{},{}
for index in range(len(IM_train)):
    for item in IM_train[index]:
        IM_train_dict[item] = index
for index in range(len(IM_test)):
    for item in IM_test[index]:
        IM_test_dict[item] = index
for index in range(len(IM_val)):
    for item in IM_val[index]:
        IM_val_dict[item] = index


GA_train,GA_test,GA_val = [],[],[]
for i in data_GA:
    GA_train.append(i[:int(len(i)*0.7)])
    GA_test.append(i[int(len(i)*0.7):int(len(i)*0.9)])
    GA_val.append(i[int(len(i)*0.9):])
GA_train,GA_test,GA_val = [],[],[]
for i in data_GA:
    GA_train.append(i[:int(len(i)*0.7)])
    GA_test.append(i[int(len(i)*0.7):int(len(i)*0.9)])
    GA_val.append(i[int(len(i)*0.9):])
GA_train_dict,GA_test_dict,GA_val_dict = {},{},{}
for index in range(len(GA_train)):
    for item in GA_train[index]:
        GA_train_dict[item] = index
for index in range(len(GA_test)):
    for item in GA_test[index]:
        GA_test_dict[item] = index
for index in range(len(GA_val)):
    for item in GA_val[index]:
        GA_val_dict[item] = index
# # 制作txt训练测试文件
with open('utils/data_Txt/IM_train.txt', 'w') as f:
    for key, val in IM_train_dict.items():
        f.write(key+','+str(val)+"\n")
with open('utils/data_Txt/IM_test.txt', 'w') as f:
    for key, val in IM_test_dict.items():
        f.write(key+','+str(val)+"\n")
with open('utils/data_Txt/IM_val.txt', 'w') as f:
    for key, val in IM_val_dict.items():
        f.write(key+','+str(val)+"\n")
        
with open('utils/data_Txt/GA_train.txt', 'w') as f:
    for key, val in GA_train_dict.items():
        f.write(key+','+str(val)+"\n")
with open('utils/data_Txt/GA_test.txt', 'w') as f:
    for key, val in GA_test_dict.items():
        f.write(key+','+str(val)+"\n")
with open('utils/data_Txt/GA_val.txt', 'w') as f:
    for key, val in GA_val_dict.items():
        f.write(key+','+str(val)+"\n")

