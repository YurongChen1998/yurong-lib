###### Yurong Chen 
import argparse
import logging
import pdb

import torch
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image

from dataset import *
from models import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(args):

    # Variables and logger Init
    cudnn.benchmark = True
    get_logger()

    # Data Load
    validloader = data_loader(args, mode='valid')

    # Model Load
    net, optimizer, best_score, start_epoch =\
        load_model(args, class_num=args.class_num, mode='train')
    log_msg = '\n'.join(['%s Train Start'%(args.model)])
    logging.info(log_msg)
    net = net.to(device)
    
    net.train()
    for module in net.modules():
        if isinstance(module, torch.nn.modules.Dropout2d):
            module.train(True)
        elif isinstance(module, torch.nn.modules.Dropout):
            module.train(True)
        else:
            pass
            
    for idx, (inputs, _, _, _) in enumerate(validloader):
        inputs = inputs.to(device)
        _ = net(inputs)
    print(">>>>>>>>>>>>>> Adjusted BN >>>>>>>>>>>>>>>>>")

    for epoch in range(1):

        print('\n\n<Validation>')
        net.eval()
        for module in net.modules():
            if isinstance(module, torch.nn.modules.Dropout2d):
                module.train(True)
            elif isinstance(module, torch.nn.modules.Dropout):
                module.train(True)
            else:
                pass
        loss = 0
        torch.set_grad_enabled(False)
        for idx, (inputs, targets_left, targets_right, paths) in enumerate(validloader):
            inputs, targets_left, targets_right  = inputs.to(device), targets_left.to(device), targets_right.to(device)
            #targets = targets_left + targets_right
            targets = targets_left
            outputs = net(inputs)
            if type(outputs) == tuple:
                outputs = outputs[0]
            #batch_loss = dice_coef(outputs.cpu(), targets.cpu(), backprop=False)
            batch_loss = dice_coef(outputs, targets)
            loss += float(batch_loss)
            
            save_image(outputs[:, 0, :, :], 'fig/ouput0_{}.png'.format(idx))
            save_image(targets[:, 0, :, :], 'fig/target0_{}.png'.format(idx))
            
            progress_bar(idx, len(validloader), 'Loss: %.5f, Dice-Coef: %.5f'
                         %((loss/(idx+1)), (1-(loss/(idx+1)))))
        log_msg = '\n'.join(['Epoch: %d  Loss: %.5f,  Dice-Coef:  %.5f'
                        %(epoch, loss/(idx+1), 1-(loss/(idx+1)))])
        logging.info(log_msg)

           

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class_num", type=bool, default=2)
    parser.add_argument("--model", type=str, default='pspnet_res50',
                        help="Model Name (unet, pspnet_squeeze, pspnet_res50,\
                        pspnet_res34, pspnet_res50, deeplab)")
    parser.add_argument("--resume", type=bool, default=True,
                        help="Model Trianing resume.")
    parser.add_argument("--in_channel", type=int, default=3,
                        help="A number of images to use for input")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="The batch size to load the data")
    parser.add_argument("--epochs", type=int, default=500,
                        help="The training epochs to run.")
    parser.add_argument("--drop_rate", type=float, default=0.1,
                        help="Drop-out Rate")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate to use in training")


    parser.add_argument("--val_img_root", type=str, default="../../../Data/S_Data/val/images",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--val_label_left_root", type=str, default="../../../Data/S_Data/val/mask",
                        help="The directory containing the training label datgaset")
    parser.add_argument("--val_label_right_root", type=str, default="../../../Data/S_Data/val/mask",
                        help="The directory containing the training label datgaset")
                        
    parser.add_argument("--output_root", type=str, default="./output/prediction",
                        help="The directory containing the result predictions")
    parser.add_argument("--ckpt_root", type=str, default="./checkpoint",
                        help="The directory containing the checkpoint files")
    args = parser.parse_args()
    test(args)
