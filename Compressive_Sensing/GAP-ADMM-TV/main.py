##############################################################################
####                             Yurong Chen                              ####
##############################################################################

from numpy import *
import scipy.io as sio
from model import gap_denoise, admm_denoise
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
from func import *

random.seed(5)
#----------------------- Data Configuration -----------------------#
dataset_dir = 'Dataset'
results_dir = 'results'
data_name = 'kaist_crop256_01'
matfile = dataset_dir + '/' + data_name + '.mat'
h, w, nC, step = 256, 256, 31, 1
data_truth = sio.loadmat(matfile)['img']
data_truth_shift = np.zeros((h, w + step*(nC - 1), nC))
for i in range(nC):
    data_truth_shift[:, i*step:i*step+256, i] = data_truth[:, :, i]
#------------------------------------------------------------------#


#----------------------- Mask Configuration -----------------------#
mask = np.zeros((h, w + step*(nC - 1)))
mask_3d = np.tile(mask[:, :, np.newaxis], (1, 1, nC))
mask_256 = sio.loadmat('mask/mask256.mat')['mask']
for i in range(nC):
    mask_3d[:, i*step:i*step+256, i] = mask_256
Phi = mask_3d
meas = np.sum(Phi * data_truth_shift, 2)
plt.figure()
plt.imshow(meas,cmap='gray')
plt.savefig('result/meas.png')
#------------------------------------------------------------------#



#-----------------------Opti. Configuration -----------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--method', default='GAP', help="Select GAP or ADMM")
parser.add_argument('--lambda_', default=1, help="_lambda is the regularization factor")
parser.add_argument('--denoiser', default='TV', help="Select which denoiser: Total Variation (TV) or Deep denoiser (HSICNN)")
parser.add_argument('--accelerate', default=True, help="Acclearted version of GAP or not")
parser.add_argument('--iter_max', default=20, help="Maximum number of iterations")
parser.add_argument('--tv_weight', default=6, help="TV denoising weight (larger for smoother but slower)")
parser.add_argument('--tv_iter_max', default=5, help="TV denoising maximum number of iterations each")
parser.add_argument('--x0', default=None, help="The initialization data point")
parser.add_argument('--sigma', default=[130, 100, 80, 70, 60, 90], help="The noise levels")
args = parser.parse_args()
#------------------------------------------------------------------#

begin_time = time.time()
if args.method == 'GAP':
    recon, psnr_all = gap_denoise(meas, Phi, data_truth, args)
    end_time = time.time()
    print('GAP-{} PSNR {:2.2f} dB, running time {:.1f} seconds.'.format(
        args.denoiser.upper(), mean(psnr_all), end_time - begin_time))
elif args.method == 'ADMM':
    recon, psnr_all = admm_denoise(meas, Phi, data_truth, args)
    end_time = time.time()
    print('ADMM-{} PSNR {:2.2f} dB, running time {:.1f} seconds.'.format(
        args.denoiser.upper(), mean(psnr_all), end_time - begin_time))

recon = shift_back(recon, step=1)
sio.savemat('./result/result.mat', {'img':recon})
fig = plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(recon[:,:,(i+1)*3], cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig('./result/result.png')
