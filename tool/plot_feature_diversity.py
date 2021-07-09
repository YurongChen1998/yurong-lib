import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import make_circles
import cmath

def standardization(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
def plot_diversity(plot_feature, n):
    plot_feature = plot_feature.detach().squeeze()
    plot_feature = plot_feature.view(plot_feature.size(0), -1)
    channel, feature = plot_feature.shape
    corr = torch.mm(plot_feature, plot_feature.permute(1, 0)) #(64, 64)
    mean_corr = torch.mean(corr, 0)
    sort_mean_corr = torch.sort(mean_corr, descending=True)
    sort_mean_corr = sort_mean_corr.indices
    
    m1_index = int(sort_mean_corr[0])
    m2_index = int(sort_mean_corr[1])

    m1 = standardization(plot_feature[m1_index,:].cpu().numpy())
    m2 = standardization(plot_feature[m2_index,:].cpu().numpy())
    
    r = standardization(np.sqrt(m1 ** 2 + m2 ** 2))
    r = r/r
    t = np.arctan2(m2, m1)
    
    for i in range(len(r)):
        cn1 = cmath.rect(r[i], t[i])
        r[i] = cn1.real
        t[i] = cn1.imag
    
    #ax = plt.subplot()
    #ax.plot(r, t, 'k.', markersize=2)
    #plt.show()
    
    ring = make_circles(15*feature, factor=0.9, noise=0.01)
    x1 = ring[0][:, 0]
    x2 = ring[0][:, 1]
 
    r = np.hstack([x1, r])
    t = np.hstack([x2, t])
    values = np.vstack([r, t])

    kernel = stats.gaussian_kde(values, bw_method=0.08)
    X, Y = np.mgrid[-1.2:1.2:1000j, -1.2:1.2:1000j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)
    
    ax = plt.subplot()
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
          extent=[-1.1, 1.1, -1.1, 1.1])

    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1,1.1)
    plt.savefig('./plot/out_{}.jpg'.format(n))
    #plt.show()

