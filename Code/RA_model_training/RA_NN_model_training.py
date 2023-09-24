import numpy as np
import scipy.io
from tensorflow.compat.v1 import keras
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import csv

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
y_scaled = y_minmax.transform(y) \
    # testing datdaset
x_test_scaled = np.zeros((500, 80))
y_test_scaled = np.zeros((500, 80))

for j in range(500):
    C = np.random.randint(0, (np.size(x, 0) - j), 1)
    x_test_scaled[j] = x_scaled[C[0]]
    x_scaled = np.delete(x_scaled, C[0], axis=0)
    y_test_scaled[j] = y_scaled[C[0]]
    y_scaled = np.delete(y_scaled, C[0], axis=0)

    pass

x_train_scaled = x_scaled
y_train_scaled = y_scaled

# select training data
n = [10, 30, 50, 70, 100, 150, 220, 300, 400, 500, 750, 1000, 1250, 1500]
n_patience = [1000]
len = np.size(n)
f = open('error_NN+random.csv', 'w', encoding='utf-8')

csv_writer = csv.writer(f)

csv_writer.writerow(["Data size", "RMSE", "MME", "MAE"])

for k in range(len):
    for i in range(np.size(n_patience)):

        model = keras.models.Sequential()
        model.add(keras.layers.Dense(units=80, activation='sigmoid', input_dim=80))
        model.add(keras.layers.Dense(units=80, activation='sigmoid'))
        model.add(keras.layers.Dense(units=80))

        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mse'])
        EarlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=n_patience[i], verbose=1, mode='min')
        # seperate training and validation dataset
        xtrain = np.zeros((n[k], 80))
        ytrain = np.zeros((n[k], 80))
        for j in range(n[k]):
            C = np.random.randint(0, (np.size(x, 0) - 500 - j), 1)
            xtrain[j] = x_train_scaled[C[0]]
            ytrain[j] = y_train_scaled[C[0]]
            pass

        xvalid = np.zeros((int(n[k] * 0.2), 80))
        yvalid = np.zeros((int(n[k] * 0.2), 80))
        for j in range(int(n[k] * 0.2)):
            C = np.random.randint(0, (n[k] - j), 1)
            xvalid[j] = xtrain[C[0]]
            xtrain = np.delete(xtrain, C[0], axis=0)
            yvalid[j] = ytrain[C[0]]
            ytrain = np.delete(ytrain, C[0], axis=0)
            pass

        history = model.fit(xtrain, ytrain,
                            epochs=1000000,
                            validation_data=(xvalid, yvalid), callbacks=[EarlyStop], verbose=2)
        from matplotlib import pyplot
        pyplot.rcParams['font.sans-serif'] = ['KaiTi']
        pyplot.rcParams['axes.unicode_minus'] = False
        pyplot.plot(history.history['loss'])
        pyplot.title('Training performance')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.show()
        model.save('model_NN/nn_model_' + str(n[k]) + '.h5')
        # calculat error
        predictions = model.predict(x_test_scaled)
        y_out = y_minmax.inverse_transform(predictions)
        y_test = y_minmax.inverse_transform(y_test_scaled)
        error = y_out - y_test
        number = 0
        count = 0
        for l in range(0, 500):
            for j in range(0, 80):
                number += 1
                count = count + error[l, j] * error[l, j]
        RMSE = np.sqrt(count / number)
        MAE = np.mean((abs(error)))
        nn_err = abs(error)
        nn_maxErr = np.zeros((1, 500))
        for j in range(500):
            nn_maxErr[:, j] = nn_err[j, np.argmax(nn_err[j, :])]
        MME = np.mean(nn_maxErr)
        print(n[k], " ", RMSE, " ", MME, " ", MAE)

        csv_writer.writerow([n[k], RMSE, MME, MAE])

f.close()