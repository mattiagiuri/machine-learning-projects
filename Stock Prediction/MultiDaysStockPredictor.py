import math
import pandas_datareader as web
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow import keras

n_of_days = 60
n_predicted = 10
plt.style.use('fivethirtyeight')

df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2021-01-02')
data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * 0.8)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len, :]

x_train = []
y_train = []

for i in range(n_of_days, len(train_data)-10):
    x_train.append(train_data[i-n_of_days:i, 0])
    y_train.append(train_data[i:i+n_predicted, 0])


x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

test_data = scaled_data[training_data_len - n_of_days:, :]


model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(100))
model.add(Dense(10))

model.compile(optimizer='adam', loss='mean_squared_error')
checkpoint_cb = keras.callbacks.ModelCheckpoint('stockpredictor.h5', save_best_only=True)

model.fit(x_train, y_train, batch_size=1, epochs=10, callbacks=[keras.callbacks.EarlyStopping(patience=10), checkpoint_cb], validation_split=0.2)

# model.save('stockpredictor.h5')


x_test = []
y_test = []
y_tet = dataset[training_data_len:, :]

for i in range(n_of_days, len(test_data) - n_predicted):
    x_test.append(test_data[i - n_of_days:i, 0])

for i in range(len(y_tet) - n_predicted):
    y_test.append(y_tet[i:i + n_predicted, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# model = keras.models.load_model('stockpredictor.h5')


predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

l = []
final_dp = {}
a = ([a+i for i in range(n_predicted)] for a in range(1000000000))

for i in range(len(predictions)+n_predicted):
    final_dp[i] = []

for i in range(len(predictions)):
    pred = predictions[i]
    tet = y_test[i]
    error = mean_squared_error(pred, tet)
    l.append(np.sqrt(error))
    r = list(next(a)).copy()
    for j in r:
        for k in range(10):
            try:
                final_dp[j].append(pred[k])
            except:
                continue

print(np.mean(l))

final_p = []
for i in range(len(predictions)+n_predicted):
    final_dp[i] = np.mean(final_dp[i])

a = list(range(len(predictions)+n_predicted))

for i in range(len(a)):
    final_p.append(final_dp[i])

final_p = np.array(final_p)
print(y_tet)
print(y_tet.shape)
plt.plot(a, final_p, c='r')
plt.plot(a, y_tet.reshape(-1, 1), c='g')

plt.show()
