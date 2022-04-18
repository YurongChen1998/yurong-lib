import numpy as np
import cv2
import math

def A(data, Phi):
    '''
    :param data: [h, w, nC]
    :param Phi: [h, w, nC(repeat)]
    :return: [h, w] Element-wise product
    '''
    return np.sum(data * Phi, axis=2)

def At(meas, Phi):
    '''
    :param meas: [h, w]
    :param Phi: [h, w, nC]
    :return: [h, w, nC] Element-wise product
    '''
    return np.multiply(np.repeat(meas[:, :, np.newaxis], Phi.shape[2], axis=2), Phi)

def shift(inputs, step):
    [h, w, nC] = inputs.shape
    output = np.zeros((h, w+(nC - 1)*step, nC))
    for i in range(nC):
        output[:, i*step : i*step + w, i] = inputs[:, :, i]
    return output

def shift_back(inputs, step):
    [h, w, nC] = inputs.shape
    for i in range(nC):
        inputs[:, :, i] = np.roll(inputs[:, :, i], (-1)*step*i, axis=1)
    output = inputs[:, 0 : w - step*(nC - 1), :]
    return output

def ssim(data, recon):
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2
    data = data.astype(np.float64)
    recon = recon.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(data, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(recon, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(data ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(recon ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(data * recon, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(data, recon, border=0):
    if not data.shape == recon.shape:
        raise ValueError('Data size must have the same dimensions!')
    h, w = data.shape[:2]
    data = data[border:h - border, border:w - border]
    recon = recon[border:h - border, border:w - border]
    if data.ndim == 2:
        return ssim(data, recon)
    elif data.ndim == 3:
        ssims = []
        for i in range(data.shape[2]):
            ssims.append(ssim(data[:, :, i], recon[:, :, i]))
        return np.array(ssims).mean()

def calculate_psnr(data, recon):
    mse = np.mean((recon - data)**2)
    if mse == 0:
        return 100
    Pixel_max = 1.
    return 20 * math.log10(Pixel_max / math.sqrt(mse))