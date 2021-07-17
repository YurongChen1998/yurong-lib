import argparse
import logging
import pdb

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image

from dataset import *
from models import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):

    # Variables and logger Init
    cudnn.benchmark = True
    get_logger()
    bce_loss = nn.BCELoss().to(device)
    recon_loss = torch.nn.L1Loss().to(device)

    # Data Load
    trainloader = data_loader(args, mode='train')
    validloader = data_loader(args, mode='valid')

    # Model Load
    net, optimizer, best_score, start_epoch =\
        load_model(args, class_num=args.class_num, mode='train')
    log_msg = '\n'.join(['%s Train Start'%(args.model)])
    logging.info(log_msg)
    net = net.to(device)

    for epoch in range(start_epoch, start_epoch+args.epochs):

        # Train Model
        print('\n\n\nEpoch: {}\n<Train>'.format(epoch))
        net.train(True)
        loss = 0
        c_loss = 0
        recontruction_loss = 0
        w_loss = 0
        lr = args.lr * (0.5 ** (epoch // 4))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        torch.set_grad_enabled(True)
        for idx, (inputs, targets_left, targets_right, paths) in enumerate(trainloader):
            inputs, targets_left, targets_right  = inputs.to(device), targets_left.to(device), targets_right.to(device)
            targets = targets_left + targets_right
            outputs, feature, recon_x = net(inputs)
            if type(outputs) == tuple:
                outputs = outputs[0]
            batch_loss = dice_coef(outputs, targets)
            class_loss = bce_loss(outputs, targets)
            
            sum_recon_loss = F.mse_loss(recon_x, inputs) + recon_loss(recon_x, inputs)
            #swd_loss = compute_swd(feature)
            
            total_loss = class_loss + batch_loss + 0.1*sum_recon_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            loss += float(batch_loss)
            c_loss += float(class_loss)
            recontruction_loss += float(sum_recon_loss)
            #w_loss += float(swd_loss)
            
            progress_bar(idx, len(trainloader), 'Class_Loss: %.5f, Recon_Error: %.5f, SWD_Loss: %.5f, Loss: %.5f, Dice-Coef: %.5f'
                         %((c_loss/(idx+1)), (recontruction_loss/(idx+1)), (w_loss/(idx+1)), (loss/(idx+1)), (1-(loss/(idx+1)))))
        log_msg = '\n'.join(['Epoch: %d  Loss: %.5f,  Dice-Coef:  %.5f'\
                         %(epoch, loss/(idx+1), 1-(loss/(idx+1)))])
        logging.info(log_msg)

        # Validate Model
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
            #targets = torch.cat([targets_left, targets_right], dim=1)
            targets = targets_left + targets_right
            outputs,_,recon_x = net(inputs)
            if type(outputs) == tuple:
                outputs = outputs[0]
            #batch_loss = dice_coef(outputs.cpu(), targets.cpu(), backprop=False)
            batch_loss = dice_coef(outputs, targets)
            loss += float(batch_loss)
            progress_bar(idx, len(validloader), 'Loss: %.5f, Dice-Coef: %.5f'
                         %((loss/(idx+1)), (1-(loss/(idx+1)))))
        log_msg = '\n'.join(['Epoch: %d  Loss: %.5f,  Dice-Coef:  %.5f'
                        %(epoch, loss/(idx+1), 1-(loss/(idx+1)))])
        logging.info(log_msg)

        save_image(outputs[:, 0, :, :], 'fig/ouput0_{}.png'.format(idx))
        save_image(recon_x[:, 0, :, :], 'fig/recon_{}.png'.format(idx))
        save_image(targets[:, 0, :, :], 'fig/target0_{}.png'.format(idx))
        
        
        # Save Model
        loss /= (idx+1)
        score = 1 - loss
        if score > best_score:
            checkpoint = Checkpoint(net, optimizer, epoch, score)
            checkpoint.save(os.path.join(args.ckpt_root, args.model+'.tar'))
            best_score = score
            print("Saving...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class_num", type=bool, default=1)
    parser.add_argument("--model", type=str, default='pspnet_res50',
                        help="Model Name (unet, pspnet_squeeze, pspnet_res50,\
                        pspnet_res34, pspnet_res50, deeplab)")
    parser.add_argument("--resume", type=bool, default=False,
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
    parser.add_argument("--img_root", type=str, default="../../../Data/J_Data/train/images",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--label_left_root", type=str, default="../../../Data/J_Data/train/mask/left",
                        help="The directory containing the training label datgaset")
    parser.add_argument("--label_right_root", type=str, default="../../../Data/J_Data/train/mask/right",
                        help="The directory containing the training label datgaset")
    parser.add_argument("--val_img_root", type=str, default="../../../Data/J_Data/val/images",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--val_label_left_root", type=str, default="../../../Data/J_Data/val/mask/left",
                        help="The directory containing the training label datgaset")
    parser.add_argument("--val_label_right_root", type=str, default="../../../Data/J_Data/val/mask/right",
                        help="The directory containing the training label datgaset")
    parser.add_argument("--output_root", type=str, default="./output/prediction",
                        help="The directory containing the result predictions")
    parser.add_argument("--ckpt_root", type=str, default="./checkpoint",
                        help="The directory containing the checkpoint files")
    args = parser.parse_args()
    train(args)
