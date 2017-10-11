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

# モデルの定義
W = tf.Variable(tf.zeros([M, K]))
b = tf.Variable(tf.zeros([K]))

x = tf.placeholder(tf.float32, shape=[None, M])
t = tf.placeholder(tf.float32, shape=[None, K])
y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))

'''
モデルの学習
'''
batch_size = 50
n_batches = N // batch_size

# 初期化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 学習
for epoch in range(20):
    X_, Y_ = shuffle(X, Y)

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        sess.run(train_step, feed_dict={
            x: X_[start:end],
            t: Y_[start:end]
        })

'''
結果確認
'''
X_, Y_ = shuffle(X, Y)

classified = correct_prediction.eval(session=sess, feed_dict={
    x: X_[0:10],
    t: Y_[0:10]
})
prob = y.eval(session=sess, feed_dict={
    x: X_[0:10]
})

print('classified:')
print(classified)
print()
print('output probability:')
print(prob)

# データのグラフ
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
fig = plt.figure()
fig_sub1 = fig.add_subplot(111)
fig_sub1.plot(X1[:,0], X1[:,1], 'o')
fig_sub1.plot(X2[:,0], X2[:,1], '^')
fig_sub1.plot(X3[:,0], X3[:,1], 'x')

w11 = sess.run(W)[0, 0]
w12 = sess.run(W)[1, 0]
w21 = sess.run(W)[0, 1]
w22 = sess.run(W)[1, 1]
w31 = sess.run(W)[0, 2]
w32 = sess.run(W)[1, 2]
b1 = sess.run(b)[0]
b2 = sess.run(b)[1]
b3 = sess.run(b)[2]
div_x = range(-2,12)
div_y1 = ((-w11 + w21) * div_x - (b1 + b2)) / (w12 - w22)
div_y2 = ((-w31 + w21) * div_x - (b3 + b2)) / (w32 - w22)
fig_sub1.plot(div_x, div_y1)
fig_sub1.plot(div_x, div_y2)
#fig_sub1.plot(range(20), val_acc, marker='.', label='val_acc')
#fig_sub1.legend(loc='best', fontsize=10)
#fig_sub1.grid()
#plt.xlabel('epoch')
#plt.ylabel('acc')
#fig_sub1.show()
fig.savefig('/vagrant/share/sample3.png') 

