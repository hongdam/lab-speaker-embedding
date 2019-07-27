from tsnecuda import TSNE
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def save_t_sne(file_name):
    X = np.load('./infer/test_f_vec.npy')
    labels = np.load('./infer/test_f_label.npy')

    X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X)

    xs = X_embedded[:, 0]
    ys = X_embedded[:, 1]
    plt.scatter(xs, ys, c=labels, alpha=0.7)

    plt.savefig('./infer/' + file_name +'.png')
    plt.cla()
