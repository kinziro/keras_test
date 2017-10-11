import numpy as np
from sklearn.model_selection import train_test_split

# 学習データ作成
def sin(x, T=100):
    return np.sin(2.0 * np.pi * x/T)

def toy_problem(T=100, ampl=0.05):
    x = np.arange(0, 2*T+1)
    noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))
    return sin(x) + noise

def CreateData(T, maxlen):
    # データ生成
    #T = 100
    f = toy_problem(T)
    
    # τごとにデータ分割
    length_of_sequences = 2 * T # 全時系列の長さ
    #maxlen = 25 # 1つの時系列データの長さ
    
    data = []
    target = []
    
    for i in range(0, length_of_sequences - maxlen + 1):
        data.append(f[i: i + maxlen])
        target.append(f[i + maxlen])
    
    X = np.array(data).reshape(len(data), maxlen, 1)    # (データ数N, τ個の時系列, 1次元)のデータ
    Y = np.array(target).reshape(len(data), 1)          # (データ数N, 1次元)のデータ
    
    return X, Y, f
