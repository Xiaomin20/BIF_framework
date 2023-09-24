import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import scipy.io
import joblib
import csv
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

database = scipy.io.loadmat('input_output.mat')

x = database['input']
y = database['gsnrall']

scaler = StandardScaler()

# normalize
x_minmax = preprocessing.MinMaxScaler()
x_minmax.fit(x)
x_scaled = x_minmax.transform(x)

y_minmax = preprocessing.MinMaxScaler()
y_minmax.fit(y)
y_scaled = y_minmax.transform(y)

data = np.column_stack((x_scaled, y_scaled))

# testing datdaset
test = np.zeros((500, 160))
for j in range(500):
    C = np.random.randint(0, (np.size(x, 0) - j), 1)
    test[j] = data[C[0]]
    data = np.delete(data, C[0], axis=0)
    pass

kernel = RBF(0.5, (1e-4, 1e4))

# origin training datasset
n_origin = 3
train = np.zeros((n_origin, 160))
for j in range(n_origin):
    C = np.random.randint(0, (np.size(x, 0) - 500 - j), 1)
    train[j] = data[C[0]]
    data = np.delete(data, C[0], axis=0)
    pass

f = open('error_GPR+AL.csv', 'w', encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow(["Data size", "RMSE", "MME", "MAE"])

for i in range(1500):
    # train
    reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, alpha=0)
    reg.fit(train[:, :80], train[:, 80:])
    joblib.dump(reg, 'model_GPR/model_' + str(i + n_origin) + '.pkl')
    # select
    output, err = reg.predict(data[:, :80], return_std=True)
    C = np.random.randint(0, (np.size(x, 0) - 500 - n_origin - i), 1)
    train = np.row_stack((train, data[np.argmax(err)]))
    data = np.delete(data, np.argmax(err), axis=0)
    # calculat error
    output, err = reg.predict(test[:, :80], return_std=True)
    error = y_minmax.inverse_transform(output) - y_minmax.inverse_transform(test[:, 80:])
    count = 0
    number = 0
    for k in range(0, 500):
        for j in range(0, 80):
            number += 1
            count = count + error[k, j] * error[k, j]
    RMSE2 = count / (number)
    MAE = np.mean((abs(error)))
    RMSE = np.sqrt(RMSE2)
    maxErr = np.zeros((1, 500))
    absErr = abs(error)
    for j in range(500):
        maxErr[:, j] = absErr[j, np.argmax(absErr[j, :])]
    MME = np.mean(maxErr)
    if ((i + n_origin) % 10 == 0):
        print(i + n_origin, RMSE, MME, MAE)
    csv_writer.writerow([i + n_origin, RMSE, MME, MAE])
    np.save("train_GPR+AL.npy", train)
f.close()