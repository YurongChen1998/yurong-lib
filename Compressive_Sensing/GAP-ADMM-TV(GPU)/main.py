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
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
random.seed(5)
#----------------------- Data Configuration -----------------------#
dataset_dir = 'Dataset'
results_dir = 'results'
data_name = 'kaist_crop256_01'
matfile = dataset_dir + '/' + data_name + '.mat'
h, w, nC, step = 256, 256, 31, 1
data_truth = torch.from_numpy(sio.loadmat(matfile)['img'])
data_truth_shift = torch.zeros((h, w + step*(nC - 1), nC))
for i in range(nC):
    data_truth_shift[:, i*step:i*step+256, i] = data_truth[:, :, i]
#------------------------------------------------------------------#


#----------------------- Mask Configuration -----------------------#
mask = torch.zeros((h, w + step*(nC - 1)))
mask_3d = torch.unsqueeze(mask, 2).repeat(1, 1, nC)
mask_256 = torch.from_numpy(sio.loadmat('mask/mask256.mat')['mask'])
for i in range(nC):
    mask_3d[:, i*step:i*step+256, i] = mask_256
Phi = mask_3d
meas = torch.sum(Phi * data_truth_shift, 2)
plt.figure()
plt.imshow(meas,cmap='gray')
plt.savefig('result/meas.png')
#------------------------------------------------------------------#



#-----------------------Opti. Configuration -----------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--method', default='ADMM', help="Select GAP or ADMM")
parser.add_argument('--lambda_', default=1, help="_lambda is the regularization factor")
parser.add_argument('--denoiser', default='TV', help="Select which denoiser: Total Variation (TV) or Deep denoiser (HSICNN)")
parser.add_argument('--accelerate', default=True, help="Acclearted version of GAP or not")
parser.add_argument('--iter_max', default=20, help="Maximum number of iterations")
parser.add_argument('--tv_weight', default=3, help="TV denoising weight (larger for smoother but slower)")
parser.add_argument('--tv_iter_max', default=10, help="TV denoising maximum number of iterations each")
parser.add_argument('--x0', default=None, help="The initialization data point")
parser.add_argument('--sigma', default=[130, 100, 80, 70, 60, 90], help="The noise levels")
args = parser.parse_args()
#------------------------------------------------------------------#

begin_time = time.time()
if args.method == 'GAP':
    recon, psnr_all = gap_denoise(meas.to(device), Phi.to(device), data_truth.to(device), args)
    end_time = time.time()
    # GAP-TV PSNR 30.10 dB, running time 106.4 seconds.
    # GAP-TV PSNR 31.24 dB, running time 6.1 seconds (GPU)
    print('GAP-{} PSNR {:2.2f} dB, running time {:.1f} seconds.'.format(
        args.denoiser.upper(), torch.mean(torch.tensor(psnr_all)), end_time - begin_time))
elif args.method == 'ADMM':
    recon, psnr_all = admm_denoise(meas.to(device), Phi.to(device), data_truth.to(device), args)
    end_time = time.time()
    # ADMM-TV PSNR 31.31 dB, running time 109.2 seconds
    # ADMM-TV PSNR 31.77 dB, running time 6.5 seconds.
    print('ADMM-{} PSNR {:2.2f} dB, running time {:.1f} seconds.'.format(
        args.denoiser.upper(), torch.mean(torch.tensor(psnr_all)), end_time - begin_time))

recon = shift_back(recon, step=1)
sio.savemat('./result/result.mat', {'img':recon})
fig = plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(recon[:,:,(i+1)*3], cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig('./result/result.png')
