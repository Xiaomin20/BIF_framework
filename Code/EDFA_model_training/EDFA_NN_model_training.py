import numpy as np
import scipy.io
from tensorflow.compat.v1 import keras
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

# select training data
n = [10, 13, 16, 20, 23, 26, 30, 32, 35, 42, 50, 60, 70, 85, 100, 120, 145, 170, 200, 230, 265, 300, 340, 380, 430, 465,
     500]
len = np.size(n)

for k in range(len):

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=40, activation='sigmoid', input_dim=40))
    model.add(keras.layers.Dense(units=40, activation='sigmoid'))
    model.add(keras.layers.Dense(units=40))

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    EarlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min')

    # seperate training and validation dataset
    xtrain = np.zeros((n[k], 40))
    ytrain = np.zeros((n[k], 40))
    for j in range(n[k]):
        C = np.random.randint(0, (9578 - j), 1)
        xtrain[j] = x_train_scaled[C[0]]
        ytrain[j] = y_train_scaled[C[0]]
        pass

    xvalid = np.zeros((int(n[k] * 0.2), 40))
    yvalid = np.zeros((int(n[k] * 0.2), 40))
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