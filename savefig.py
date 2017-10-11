#　グラフを表示する代わりに画像ファイルとして保存
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def savepng(data, filename='sample.png', xlabel='x', ylabel='y'):
    fig = plt.figure()
    fig_sub1 = fig.add_subplot(111)
    fig_sub1.plot(data)
    fig_sub1.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    savepath = '/vagrant/share/' + filename
    fig.savefig(savepath)
