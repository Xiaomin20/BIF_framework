import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import scipy.io
import joblib
import csv
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

data1 = scipy.io.loadmat('spec_in_train.mat')
data2 = scipy.io.loadmat('spec_in_test.mat')
data3 = scipy.io.loadmat('spec_out_train.mat')
data4 = scipy.io.loadmat('spec_out_test.mat')

x_train = data1['spec_in_train']
x_test = data2['spec_in_test']
y_train = data3['spec_out_train'] - x_train
y_test = data4['spec_out_test'] - x_test

scaler = StandardScaler()

# normalize
x_minmax = preprocessing.MinMaxScaler()
x_minmax.fit(x_train)
x_train_scaled = x_minmax.transform(x_train)
x_test_scaled = x_minmax.transform(x_test)

y_minmax = preprocessing.MinMaxScaler()
y_minmax.fit(y_train)
y_train_scaled = y_minmax.transform(y_train)
y_test_scaled = y_minmax.transform(y_test)

test = np.column_stack((x_test_scaled, y_test_scaled))
data = np.column_stack((x_train_scaled, y_train_scaled))

kernel = RBF(0.5, (1e-4, 1e4))

n_origin_data = 10

train = np.zeros((n_origin_data, 80))

for j in range(n_origin_data):
    C = np.random.randint(0, (9578 - j), 1)
    train[j] = data[C[0]]
    data = np.delete(data, C[0], axis=0)

    pass

f = open('error_GPR+AL.csv', 'w', encoding='utf-8')

csv_writer = csv.writer(f)
csv_writer.writerow(["Data size", "RMSE", "MME", "MAE"])

for i in range(500):

    # train
    reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, alpha=0.00001)
    reg.fit(train[:, :40], train[:, 40:])
    joblib.dump(reg, 'model_GPR/model_' + str(i + n_origin_data) + '.pkl')
    # select
    output, err = reg.predict(data[:, :40], return_std=True)
    C = np.random.randint(0, (9577 - n_origin_data - i), 1)
    train = np.row_stack((train, data[np.argmax(err)]))
    data = np.delete(data, np.argmax(err), axis=0)
    # calculat error
    output, err = reg.predict(test[:, :40], return_std=True)
    error = y_minmax.inverse_transform(output) - y_minmax.inverse_transform(test[:, 40:])
    count = 0
    number = 0
    for k in range(0, 1002):
        for j in range(0, 40):
            number += 1
            count = count + error[k, j] * error[k, j]
    RMSE2 = count / number
    MAE = np.mean((abs(error)))
    RMSE = np.sqrt(RMSE2)

    maxErr = np.zeros((1, 1002))
    absErr = abs(error)
    for j in range(1002):
        maxErr[:, j] = absErr[j, np.argmax(absErr[j, :])]
    MME = np.mean(maxErr)

    print(i + n_origin_data, " ", RMSE, " ", MME, " ", MAE)

    csv_writer.writerow([i + n_origin_data, RMSE, MME, MAE])

    np.save("train_GPR+AL.npy", train)  # 保存文件

f.close()