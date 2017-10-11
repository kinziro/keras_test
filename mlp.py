from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

# Kerasに含まれるMNISTデータの取得
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 配列の整形と、色の範囲を0-255 -> 0-1に変換
X_train = X_train.reshape(60000, 784) / 255
X_test = X_test.reshape(10000, 784) / 255

# 正解データを数値からダミー変数の形式に変換
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# ネットワークの定義
model = Sequential([
        Dense(512, input_shape=(784,)),
        Activation('sigmoid'),
        Dense(10),
        Activation('softmax')
    ])

# 損失関数、最適化アルゴリズムなどを設定したモデルのコンパイルを行う
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 学習処理の実行
hist = model.fit(X_train, y_train, batch_size=200, verbose=1, epochs=20, validation_split=0.1)

# 予測
score = model.evaluate(X_test, y_test, verbose=1)
print('test accuracy : ', score[1])


# --学習状況の可視化
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

loss = hist.history['loss']
val_loss = hist.history['val_loss']

# lossのグラフ
fig = plt.figure()
fig_sub1 = fig.add_subplot(111)
fig_sub1.plot(range(20), loss, marker='.', label='loss')
fig_sub1.plot(range(20), val_loss, marker='.', label='val_loss')
fig_sub1.legend(loc='best', fontsize=10)
fig_sub1.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
#plt.show()
fig.savefig('/vagrant/share/sample1.png')

acc = hist.history['acc']
val_acc = hist.history['val_acc']

# accuracyのグラフ
fig = plt.figure()
fig_sub1 = fig.add_subplot(111)
fig_sub1.plot(range(20), acc, marker='.', label='acc')
fig_sub1.plot(range(20), val_acc, marker='.', label='val_acc')
fig_sub1.legend(loc='best', fontsize=10)
fig_sub1.grid()
plt.xlabel('epoch')
plt.ylabel('acc')
#fig_sub1.show()
fig.savefig('/vagrant/share/sample2.png')
