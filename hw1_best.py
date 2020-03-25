import sys
import pandas as pd
import numpy as np

feature = ['NO','NO2','NOx','O3','PM10','PM2.5','SO2','RH','WS_HR']

num_feature = 9

#testdata = pd.read_csv('gdrive/My Drive/hw1-regression/test.csv', header = None, encoding = 'big5')
mean_x = np.load('mean.npy')
std_x = np.load('std.npy')
testdata = pd.read_csv(sys.argv[1], header = None, encoding = 'big5')
testdata = testdata[testdata[1].isin(feature)]
test_data = testdata.iloc[:, 2:]
print(test_data)
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
with open(sys.argv[2], mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)






