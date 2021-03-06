import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import make_circles

def standardization(data):
    #mu = np.mean(data, axis=0)
    #sigma = np.std(data, axis=0)
    return 2*(data - np.min(data)) / (np.max(data) - np.min(data)) - 1
    
def plot_diversity(plot_feature, n):
    plot_feature = plot_feature.detach().squeeze()
    plot_feature = plot_feature.view(plot_feature.size(0), -1)
    channel, feature = plot_feature.shape

    corr = torch.mm(plot_feature, plot_feature.permute(1, 0)) #(64, 64)
    mean_corr = torch.mean(corr, 0)
    sort_mean_corr = torch.sort(mean_corr, descending=True)
    #sort_mean_corr = torch.sort(mean_corr)
    sort_mean_corr = sort_mean_corr.indices
    
    #m1_index = int(sort_mean_corr[int(channel/2)])
    #m2_index = int(sort_mean_corr[int(channel/2) -1])
    
    print(plot_feature)

    m1_index = int(sort_mean_corr[int(0)])
    m2_index = int(sort_mean_corr[int(1)])

    m1 = standardization(plot_feature[m1_index,:].cpu().numpy())
    m2 = standardization(plot_feature[m2_index,:].cpu().numpy())

    #m1 = np.cos(m1)
    #m2 = np.sin(m2)
    #for j in range(feature):
    #    len = np.random.uniform(low=0.9, high=1.0)
    #    m1[j] = m1[j] * len
    #    m2[j] = m2[j] * len
    
    r = np.sqrt(m1 ** 2 + m2 ** 2)
    t = np.arctan2(m2, m1)
    
    m1 = standardization(r)
    m1 = m1 / m1
    m2 = standardization(t)

    #print("!!!", r.min(), r.max(), t.min(), t.max())

    ring = make_circles(60*feature, factor=0.9, noise=0.01)
    x1 = 1.35 * ring[0][:, 0]
    x2 = 1.35 * ring[0][:, 1]

    #plt.figure(figsize=(10, 10))
    #plt.scatter(m1,m2,marker='*',color='r')
    #plt.show()

    m1 = np.hstack([m1, x1])
    m2 = np.hstack([m2, x2])
    xmin = m1.min()
    xmax = m1.max()
    ymin = m2.min()
    ymax = m2.max()
    
    print(xmin, xmax, ymin, ymax)
    
    values = np.vstack([m1, m2])

    kernel = stats.gaussian_kde(values, bw_method=0.08)
    X, Y = np.mgrid[-1.6:1.6:500j, -1.6:1.6:500j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)
    
    ax = plt.subplot()
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
          extent=[-1.6, 1.6, -1.6, 1.6])
          
    #ax.plot(m1, m2, 'k.', markersize=2)
    plt.xlim(-1.6,1.6)
    plt.ylim(-1.6,1.6)
    plt.savefig('./plot/out_{}.jpg'.format(n))
    #plt.show()

def test():
    import torchvision.models as models
    import torch.nn as nn
    resnet50 = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(resnet50.children())[:-2])
    model.eval()
    out = model(torch.rand(1, 3, 224, 224))
    plot_diversity(out, 0)
    
#test()
    
