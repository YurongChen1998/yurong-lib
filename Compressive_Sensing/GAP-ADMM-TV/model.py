from func import *
import torch
from skimage.restoration import denoise_tv_chambolle
from bm3d import bm3d_deblurring

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def gap_denoise(meas, Phi, data_truth, args):
    #-------------- Initialization --------------#
    if args.x0 is None:
        x0 = At(meas, Phi)
    meas_1 = torch.zeros_like(meas)
    iter_max = [args.iter_max] * len(args.sigma)
    ssim_all = []
    psnr_all = []
    k = 0
    show_iqa = True
    noise_estimate = True
    Phi_sum = torch.sum(Phi, 2)
    Phi_sum[Phi_sum==0] = 1
    x = x0

    # ---------------- Iteration ----------------#
    for idx, noise_level in enumerate(args.sigma):
        for iter in range(iter_max[idx]):
            x = x.to(device)   
            meas_b = A(x, Phi)
            if args.accelerate:
                meas_1 = meas_1 + (meas - meas_b)
                x = x + args.lambda_ * (At((meas_1 - meas_b)/Phi_sum, Phi))
            else:
                x = x + args.lambda_ * (At((meas - meas_b)/Phi_sum, Phi))
            x = shift_back(x, step=1)

            if args.denoiser == 'TV':
                x = TV_denoiser(x, args.tv_weight, args.tv_iter_max)

            # --------------- Evaluation ---------------#
            if show_iqa and data_truth is not None:
                ssim_all.append(calculate_ssim(data_truth, x))
                psnr_all.append(calculate_psnr(data_truth, x))
                if (k + 1) % 1 == 0:
                    print('  GAP-{0} iteration {1: 3d}, '
                        'PSNR {2:2.2f} dB.'.format(args.denoiser.upper(), k + 1, psnr_all[k]),
                        'SSIM:{}'.format(ssim_all[k]))

            x = shift(x, step=1)
            k = k + 1
    return x, psnr_all


def admm_denoise(meas, Phi, data_truth, args):
    #-------------- Initialization --------------#
    if args.x0 is None:
        x0 = At(meas, Phi)
    iter_max = [args.iter_max] * len(args.sigma)
    ssim_all = []
    psnr_all = []
    k = 0
    show_iqa = True
    noise_estimate = True
    Phi_sum = torch.sum(Phi, 2)
    Phi_sum[Phi_sum==0] = 1
    x = x0.to(device)
    theta = x0.to(device)
    b = torch.zeros_like(x0).to(device)
    gamma = 0.01

# ---------------- Iteration ----------------#
    for idx, noise_level in enumerate(args.sigma):
        for iter in range(iter_max[idx]):
            # Euclidean Projection
            theta = theta.to(device)
            b = b.to(device)
            meas_b = A(theta+b, Phi)
            x = (theta + b) + args.lambda_*(At((meas - meas_b)/(Phi_sum + gamma), Phi))
            x1 = shift_back(x-b, step=1)

            if args.denoiser == 'TV':
                theta = TV_denoiser(x1, args.tv_weight, args.tv_iter_max)

            # --------------- Evaluation ---------------#
            if show_iqa and data_truth is not None:
                ssim_all.append(calculate_ssim(data_truth, theta))
                psnr_all.append(calculate_psnr(data_truth, theta))
                if (k + 1) % 1 == 0:
                    print('  ADMM-{0} iteration {1: 3d}, '
                          'PSNR {2:2.2f} dB.'.format(args.denoiser.upper(), k + 1, psnr_all[k]),
                          'SSIM:{}'.format(ssim_all[k]))
            theta = shift(theta, step=1)
            b = b - (x.to(device) - theta.to(device))
            k += 1
    return theta, psnr_all
