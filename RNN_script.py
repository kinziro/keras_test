from CreateSinData import *
from RNN import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
from savefig import savepng
from EarlyStopping import *

# 学習データ、確認データ
T = 100
maxlen = 25
X, Y, f = CreateData(T, maxlen)

N_train = int(X.shape[0] * 0.9)
N_validation = X.shape[0] - N_train

X_train, X_validation, Y_train, Y_validation = \
        train_test_split(X, Y, test_size=N_validation)
        

n_in = len(X[0][0]) #1
n_hidden = 20
n_out = len(Y[0])   #1

x = tf.placeholder(tf.float32, shape=[None, maxlen, n_in])
t = tf.placeholder(tf.float32, shape=[None, n_out])
n_batch = tf.placeholder(tf.int32, shape=[])

y = inference(x, n_batch, maxlen=maxlen, n_hidden=n_hidden, n_out=n_out)
loss = loss(y, t)
train_step = training(loss)

epochs = 500
batch_size = 10

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

n_batches = N_train // batch_size

history = {'val_loss':[],
            'val_acc':[]}

early_stopping = EarlyStopping(patience=10, verbose=1)

for epoch in range(epochs):
    X_, Y_ = shuffle(X_train, Y_train)

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        sess.run(train_step, feed_dict={
            x: X_[start:end],
            t: Y_[start:end],
            n_batch: batch_size
        })

    # 検証データを用いた評価
    val_loss = loss.eval(session=sess, feed_dict={
        x: X_validation,
        t: Y_validation,
        n_batch: N_validation
    })

    # val_acc = accuracy.eval(session=sess, feed_dict={
    #     x: X_validation,
    #     t: Y_validation,
    #     n_batch: N_validation
    # })

    history['val_loss'].append(val_loss)
    # history['val_acc'].append(val_acc)
    print('epoch:', epoch, ' validation loss:', val_loss)

    # Early Stopping チェック
    if early_stopping.validate(val_loss):
        break

# 学習状況の可視化
savepng(history['val_loss'],filename='val_loss', xlabel='epochs')
# savepng(history['val_acc'],filename='val_acc', xlabel='epochs')

# 結果確認
tuncate = maxlen
Z = X[:1]   # 元データの最初の一部だけ切り出し

original = [f[i] for i in range(maxlen)]
predicted = [None for i in range(maxlen)]

for i in range(T * 2 -maxlen + 1):
    # 最後の時系列データから未来を予測
    z_ = Z[-1:]
    y_ = y.eval(session=sess, feed_dict={
        x: Z[-1:],
        n_batch: 1
    })

    # 予測結果を用いて新しい時系列データを生成
    sequence_ = np.concatenate((z_.reshape(maxlen, n_in)[1:], y_), axis=0) \
            .reshape(1, maxlen, n_in)
    
    Z = np.append(Z, sequence_, axis=0)
    predicted.append(y_.reshape(-1))

pre_plt = original + predicted[25:]

savepng(toy_problem(T, ampl=0), filename='sin_original')
savepng(pre_plt, filename='sin_predicted')



