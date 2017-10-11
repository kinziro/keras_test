import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from CreateSinData import *
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from savefig import *

# 学習データ
T = 100
maxlen = 25
X, Y, f = CreateData(T, maxlen)

N_train = int(X.shape[0] * 0.9)
N_validation = X.shape[0] - N_train

X_train, X_validation, Y_train, Y_validation = \
        train_test_split(X, Y, test_size=N_validation)

n_in = len(X[0][0])
n_hidden = 20
n_out = len(Y[0])

# モデルの定義

def weight_variable(shape):
    return K.truncated_normal(shape, stddev=0.01)

model = Sequential()
model.add(SimpleRNN(n_hidden,
                    init=weight_variable,
                    input_shape=(maxlen, n_out)))
model.add(Dense(n_out, init=weight_variable))
model.add(Activation('linear'))

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss='mean_squared_error', optimizer=optimizer)

# 学習
epochs = 500
batch_size = 10
early_stopping = EarlyStopping(monitor='val_loss', patience=500, verbose=1)

hist = model.fit(X_train, Y_train, batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_validation, Y_validation),
          callbacks=[early_stopping])

# 結果確認
savepng(hist.history['val_loss'], filename='val_loss', xlabel='epochs')

tuncate =maxlen
Z = X[:1]

original = [f[i] for i in range(maxlen)]
predicted = [None for i in range(maxlen)]

for i in range(T * 2 - maxlen +1):
    z_ = Z[-1:]
    y_ = model.predict(z_)
    sequence_ = np.concatenate(
            (z_.reshape(maxlen, n_in)[1:], y_),
            axis=0).reshape(1, maxlen, n_in)
    Z = np.append(Z, sequence_, axis=0)
    predicted.append(y_.reshape(-1))

pre_plt = original + predicted[25:]

savepng(pre_plt, filename='sin_predicted_keras')
