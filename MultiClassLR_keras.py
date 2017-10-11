import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

M = 2
K = 3
n = 100
N = n*K

X1 = np.random.randn(n, M) + np.array([0, 10])
X2 = np.random.randn(n, M) + np.array([5, 5])
X3 = np.random.randn(n, M) + np.array([10, 0])
Y1 = np.array([[1, 0, 0] for i in range(n)])
Y2 = np.array([[0, 1, 0] for i in range(n)])
Y3 = np.array([[0, 0, 1] for i in range(n)])

X = np.concatenate((X1, X2, X3), axis=0)
Y = np.concatenate((Y1, Y2, Y3), axis=0)

'''
モデルの定義
'''
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(input_dim=M, units=K))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1))

minibatch_size = 50
model.fit(X, Y, epochs=20, batch_size=minibatch_size)

X_, Y_ = shuffle(X, Y)
classes = model.predict_classes(X_[0:10], batch_size=minibatch_size)
prob = model.predict_proba(X_[0:10], batch_size=1)

print('classified:')
print(np.argmax(model.predict(X_[0:10]), axis=1) == classes)
print()
print('output probability:')
print(prob)

## データのグラフ
#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt
#fig = plt.figure()
#fig_sub1 = fig.add_subplot(111)
#fig_sub1.plot(X1[:,0], X1[:,1], 'o')
#fig_sub1.plot(X2[:,0], X2[:,1], '^')
#fig_sub1.plot(X3[:,0], X3[:,1], 'x')
#
#w11 = sess.run(W)[0, 0]
#w12 = sess.run(W)[1, 0]
#w21 = sess.run(W)[0, 1]
#w22 = sess.run(W)[1, 1]
#w31 = sess.run(W)[0, 2]
#w32 = sess.run(W)[1, 2]
#b1 = sess.run(b)[0]
#b2 = sess.run(b)[1]
#b3 = sess.run(b)[2]
#div_x = range(-2,12)
#div_y1 = ((-w11 + w21) * div_x - (b1 + b2)) / (w12 - w22)
#div_y2 = ((-w31 + w21) * div_x - (b3 + b2)) / (w32 - w22)
#fig_sub1.plot(div_x, div_y1)
#fig_sub1.plot(div_x, div_y2)
##fig_sub1.plot(range(20), val_acc, marker='.', label='val_acc')
##fig_sub1.legend(loc='best', fontsize=10)
##fig_sub1.grid()
##plt.xlabel('epoch')
##plt.ylabel('acc')
##fig_sub1.show()
#fig.savefig('/vagrant/share/sample3.png') 

