import math
import argparse, yaml

import numpy as np

import utils
import os
from tqdm import tqdm
import logging
import sys
import time
import importlib
import glob
import cv2
import imageio

parser = argparse.ArgumentParser(description='MBAN')
## yaml configuration files
parser.add_argument('--config', type=str, default=None, help = 'pre-config file for training')
parser.add_argument('--resume', type=str, default=None, help = 'resume training or not')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.config:
       opt = vars(args)
       yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
       opt.update(yaml_args)
    ## set visibel gpu
    gpu_ids_str = str(args.gpu_ids).replace('[','').replace(']','')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_ids_str)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import MultiStepLR, StepLR
    from datas.utils import create_datasets


    ## select active gpu devices
    device = None
    if args.gpu_ids is not None and torch.cuda.is_available():
        print('use cuda & cudnn for acceleration!')
        print('the gpu id is: {}'.format(args.gpu_ids))
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        print('use cpu for training!')
        device = torch.device('cpu')
    torch.set_num_threads(args.threads)

    ## create dataset for training and validating
    train_dataloader, valid_dataloaders = create_datasets(args)

    ## definitions of model
    try:
        model = utils.import_module('models.{}_network'.format(args.model, args.model)).create_model(args)
    except Exception:
        raise ValueError('not supported model type! or something')
    model = nn.DataParallel(model).to(device)

    ## load pretrain
    if args.pretrain is not None:
        print('load pretrained model: {}!'.format(args.pretrain))
        ckpt = torch.load(args.pretrain)
        model.load_state_dict(ckpt['model_state_dict'])

    experiment_name = None
    timestamp = utils.cur_timestamp_str()
    if args.log_name is None:
        experiment_name = 'test-{}-{}-x{}-{}'.format(args.model, 'fp32', args.scale, timestamp)
    else:
        experiment_name = '{}-{}'.format(args.log_name, timestamp)
    experiment_path = os.path.join(args.log_path, experiment_name)
    log_name = os.path.join(experiment_path, 'log.txt')

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    time.sleep(3)  # sleep 3 seconds
    sys.stdout = utils.ExperimentLogger(log_name, sys.stdout)

    ## print architecture of model
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)
    sys.stdout.flush()

    ## test
    torch.set_grad_enabled(False)
    test_log = ''
    model = model.eval()
    for valid_dataloader in valid_dataloaders:
        avg_psnr, avg_ssim, avg_time = 0.0, 0.0, 0.0
        name = valid_dataloader['name']
        loader = valid_dataloader['dataloader']
        i=1
        output_folder = os.path.join(experiment_path, args.output_folder, name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for lr, hr in tqdm(loader, ncols=80):
            lr, hr = lr.to(device), hr.to(device)
            start.record()
            sr = model(lr)
            end.record()
            torch.cuda.synchronize()
            avg_time += start.elapsed_time(end)  # milliseconds
            # out_img_1 = utils.tensor2np(sr.detach()[0])
            # quantize output to [0, 255]
            hr = hr.clamp(0, 255)
            sr = sr.clamp(0, 255)
            out_img_2=sr.detach()[0].float().cpu().numpy()
            out_img_2 =np.transpose(out_img_2,(1,2,0))
            # conver to ycbcr
            if args.colors == 3:
                hr_ycbcr = utils.rgb_to_ycbcr(hr)
                sr_ycbcr = utils.rgb_to_ycbcr(sr)
                hr = hr_ycbcr[:, 0:1, :, :]
                sr = sr_ycbcr[:, 0:1, :, :]
            # crop image for evaluation
            hr = hr[:, :, args.scale:-args.scale, args.scale:-args.scale]
            sr = sr[:, :, args.scale:-args.scale, args.scale:-args.scale]
            # calculate psnr and ssim
            psnr = utils.calc_psnr(sr, hr)
            ssim = utils.calc_ssim(sr, hr)

            # hr = utils.tensor2np(hr.detach()[0])
            # sr = utils.tensor2np(sr.detach()[0])
            # hr = utils.shave(hr, args.scale)
            # sr = utils.shave(sr, args.scale)
            # # # calculate psnr and ssim
            # psnr = utils.compute_psnr(sr, hr)
            # ssim = utils.compute_ssim(sr, hr)

            avg_psnr += psnr
            avg_ssim += ssim

            # ## save output image
            # output_path_1 = os.path.join(output_folder, '1_'+str(i) + '_x' + str(args.scale) + '.png')
            # # sr = np.array(sr)
            # # cv2.imwrite(output_path, out_img)
            # imageio.imwrite(output_path_1, out_img_1)

            ## save output image
            output_path_2 = os.path.join(output_folder, str(i) + '_x' + str(args.scale) + '.png')
            # sr = np.array(sr)
            # cv2.imwrite(output_path, out_img)
            imageio.imwrite(output_path_2, out_img_2)

            # ## save output image
            # output_path_3 = os.path.join(output_folder, '3_'+str(i) + '_x' + str(args.scale) + '.png')
            # # sr = np.array(sr)
            # cv2.imwrite(output_path_3, out_img_2)
            # # imageio.imwrite(output_path_2, out_img_2)
            i += 1

        avg_psnr = round(avg_psnr / len(loader) + 5e-3, 2)
        avg_ssim = round(avg_ssim / len(loader) + 5e-5, 4)
        avg_time = round(avg_psnr / len(loader) + 5e-3, 4)
        test_log += '[{}-X{}], PSNR/SSIM/Time: {:.2f}/{:.4f}/{} \n'.format(
            name, args.scale, float(avg_psnr), float(avg_ssim), avg_time)
    # print log & flush out
    print(test_log)
    sys.stdout.flush()