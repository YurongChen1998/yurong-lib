#### Yurong Chen
import os
import sys

import torch
from torch.optim import Adam, SGD

from .pspnet import *

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(args, class_num, mode):

    # Device Init
    net = pspnet_res50()
    ckpt =torch.load('models/checkpoint_450.pth')
    ckpt = ckpt['autoencoder']
    del ckpt['aux_classifier.weight']
    del ckpt['aux_classifier.bias']
    net.feats.load_state_dict(ckpt)
        
    # Optimizer Init
    if mode == 'train':
        resume = args.resume
        #optimizer = Adam(net.parameters(), lr=args.lr)
        optimizer = SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif mode == 'test':
        resume = True
        optimizer = None
    else:
        raise ValueError('load_model mode ERROR')

    # Model Load
    if resume:
        checkpoint = Checkpoint(net, optimizer)
        checkpoint.load(os.path.join(args.ckpt_root, args.model+'.tar'))
        print("~~~~~~~~~~~~~~~~~~~~Load Checkpoint~~~~~~~~~~~~~~~~~")
        best_score = checkpoint.best_score
        start_epoch = checkpoint.epoch+1
    else:
        best_score = 0
        start_epoch = 1

    if device == 'cuda':
        net.cuda()
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark=True

    return net, optimizer, best_score, start_epoch
