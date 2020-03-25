import sys
import pandas as pd
import numpy as np
#from google.colab import drive 
#!gdown --id '1wNKAxQ29Gnum_featurekgpBy_asjTcZRRgmsCZRm' --output data.zip
#!unzip data.zip
# data = pd.read_csv('gdrive/My Drive/hw1-regression/train.csv', header = None, encoding = 'big5')
data = pd.read_csv('./train.csv', encoding = 'big5')
feature = ['NO','NO2','NOx','O3','PM10','PM2.5','SO2','RH','WS_HR']
data = data[data['測項'].isin(feature)]
#print(data)
data = data.iloc[:, 3:]
#data[data == 'NR'] = 0
raw_data = data.to_numpy()
num_feature = 9
for i in range(240):
    for j in range(24):
        if raw_data[i + 9][j] == -1:
            if j > 0:
                raw_data[i + 9][j] = raw_data[i + 9][j - 1]
            else:
                raw_data[i + 9][0] = raw_data[i + 9][1]
month_data = {}
for month in range(12):
    sample = np.empty([num_feature, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[num_feature * (20 * month + day) : num_feature * (20 * month + day + 1), :]
    month_data[month] = sample

x = np.empty([12 * 471, num_feature * 9], dtype = float)
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > num_feature:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:num_feature*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][5, day * 24 + hour + 9] #value

mean_x = np.mean(x, axis = 0) #num_feature * 9 
std_x = np.std(x, axis = 0) #num_feature * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #num_feature * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

import math
"""#validation set
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]
print(x_train_set)
print(y_train_set)
print(x_validation)
print(y_validation)
print(len(x_train_set))
print(len(y_train_set))
print(len(x_validation))
print(len(y_validation))"""

dim = num_feature * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)
learning_rate = 10
iter_time = 15000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)#rmse
    if(t%100==0):
        print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
np.save('weight.npy', w)

#testdata = pd.read_csv('gdrive/My Drive/hw1-regression/test.csv', header = None, encoding = 'big5')
testdata = pd.read_csv('./test.csv', header = None, encoding = 'big5')
testdata = testdata[testdata[1].isin(feature)]
test_data = testdata.iloc[:, 2:]
print(test_data)
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, num_feature*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[num_feature * i: num_feature* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

w = np.load('weight.npy')
ans_y = np.dot(test_x, w)

import csv
with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)
























