from func import *
import torch
from skimage.restoration import denoise_tv_chambolle
from bm3d import bm3d_deblurring

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def TV_denoiser(x, _lambda, n_iter_max):
    dt = 0.25
    N = x.shape
    idx = np.arange(1, N[0]+1)
    idx[-1] = N[0]-1
    iux = np.arange(-1, N[0]-1)
    iux[0] = 0
    ir = np.arange(1, N[1]+1)
    ir[-1] = N[1]-1
    il = np.arange(-1, N[1]-1)
    il[0] = 0
    p1 = np.zeros_like(x)
    p2 = np.zeros_like(x)
    divp = np.zeros_like(x)

    for i in range(n_iter_max):
        z = divp - x*_lambda
        z1 = z[:,ir,:] - z
        z2 = z[idx,:,:] - z
        denom_2d = 1 + dt*np.sqrt(np.sum(z1**2 + z2**2, 2))
        denom_3d = np.tile(denom_2d[:,:,np.newaxis], (1,1,N[2]))
        p1 = (p1+dt*z1)/denom_3d
        p2 = (p2+dt*z2)/denom_3d
        divp = p1-p1[:,il,:] + p2 - p2[iux,:,:]
    u = x - divp/_lambda;
    return u
    
def gap_denoise(meas, Phi, data_truth, args):
    #-------------- Initialization --------------#
    if args.x0 is None:
        x0 = At(meas, Phi)
    meas_1 = np.zeros_like(meas)
    iter_max = [args.iter_max] * len(args.sigma)
    ssim_all = []
    psnr_all = []
    k = 0
    show_iqa = True
    noise_estimate = True
    Phi_sum = np.sum(Phi, 2)
    Phi_sum[Phi_sum==0] = 1
    x = x0

    # ---------------- Iteration ----------------#
    for idx, noise_level in enumerate(args.sigma):
        for iter in range(iter_max[idx]):
            meas_b = A(x, Phi)
            if args.accelerate:
                meas_1 = meas_1 + (meas - meas_b)
                x = x + args.lambda_ * (At((meas_1 - meas_b)/Phi_sum, Phi))
            else:
                x = x + args.lambda_ * (At((meas - meas_b)/Phi_sum, Phi))
            x = shift_back(x, step=1)

            if args.denoiser == 'TV':
                #x = denoise_tv_chambolle(x, noise_level / 255, n_iter_max=args.tv_iter_max, multichannel=True)
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
    Phi_sum = np.sum(Phi, 2)
    Phi_sum[Phi_sum==0] = 1
    x = x0
    theta = x0
    b = np.zeros_like(x0)
    gamma = 0.01

# ---------------- Iteration ----------------#
    for idx, noise_level in enumerate(args.sigma):
        for iter in range(iter_max[idx]):
            # Euclidean Projection
            meas_b = A(theta+b, Phi)
            x = (theta + b) + args.lambda_*(At((meas - meas_b)/(Phi_sum + gamma), Phi))
            x1 = shift_back(x-b, step=1)

            if args.denoiser == 'TV':
                theta = denoise_tv_chambolle(x1, noise_level / 255, n_iter_max=args.tv_iter_max, multichannel=True)

            # --------------- Evaluation ---------------#
            if show_iqa and data_truth is not None:
                ssim_all.append(calculate_ssim(data_truth, theta))
                psnr_all.append(calculate_psnr(data_truth, theta))
                if (k + 1) % 1 == 0:
                    print('  ADMM-{0} iteration {1: 3d}, '
                          'PSNR {2:2.2f} dB.'.format(args.denoiser.upper(), k + 1, psnr_all[k]),
                          'SSIM:{}'.format(ssim_all[k]))
            theta = shift(theta, step=1)
            b = b - (x - theta)
            k += 1
    return theta, psnr_all
