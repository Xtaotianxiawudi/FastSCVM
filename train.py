from datetime import datetime
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import os.path as osp

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import yaml
from train_process import Trainer
from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr
from torch.backends import cudnn

############################### networks ############################
# from Remove_BMF import *
# from xmq import *
# from LightM_UNet import *
# from U_New3 import *
# from UltraLight_VM_UNet import *
# from edgenext import *
# from deeplabv3plus import *
# from U_New3_remove_mamba import *
from FastSCVM3_New import *


cudnn.benchmark = False
cudnn.deterministic = True

here = osp.dirname(osp.abspath(__file__))


def main():
    # Add default values to all parameters
    print("MFM10")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--resume', default=None, help='checkpoint path')
    parser.add_argument(
        '--coefficient', type=float, default=0.01, help='balance coefficient'
    )
    parser.add_argument(
        '--boundary-exist', type=bool, default=True, help='whether or not using boundary branch'
    )
    #############################需要修改的地方#############################
    parser.add_argument(
        '--dataset', type=str, default='refuge', help='folder id contain images ROIs to train or validation'
    )  # gdrishtiGS
    parser.add_argument(
        '--aux', type=str, default=False, help='folder id contain images ROIs to train or validation'
    )
    #############################需要修改的地方#############################
    parser.add_argument(
        '--batch-size', type=int, default=2, help='batch size for training the model'
    )
    parser.add_argument(
        '--max-epoch', type=int, default=800, help='max epoch'
    )
    parser.add_argument(
        '--stop-epoch', type=int, default=800, help='stop epoch'
    )
    parser.add_argument(
        '--warmup-epoch', type=int, default=-1, help='warmup epoch begin train GAN'
    )
    parser.add_argument(
        '--interval-validate', type=int, default=1, help='interval epoch number to valide the model'
    )
    parser.add_argument(
        '--lr-gen', type=float, default=1e-3, help='learning rate',
    )
    parser.add_argument(
        '--lr-dis', type=float, default=2.5e-5, help='learning rate',
    )
    parser.add_argument(
        '--lr-decrease-rate', type=float, default=0.2, help='ratio multiplied to initial lr',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.9, help='momentum',
    )
    parser.add_argument(
        '--data-dir',
        default=r'D:\upload\datasets\fundus',
        help='data root path'
    )

    args = parser.parse_args()
    now = datetime.now()
    args.out = osp.join(here, 'logs', args.dataset, now.strftime('%Y%m%d_%H%M%S.%f'))
    os.makedirs(args.out)

    # save training hyperparameters or/and settings
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(2020)
    if cuda:
        torch.cuda.manual_seed(2020)

    import random
    import numpy as np
    random.seed(2020)
    np.random.seed(2020)

    # 1. loading data
    composed_transforms_train = transforms.Compose([
        tr.Resize(256),
        tr.RandomRotate(),
        tr.RandomFlip(),
        tr.elastic_transform(),
        tr.add_salt_pepper_noise(),
        tr.adjust_light(),
        tr.eraser(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    composed_transforms_val = transforms.Compose([
        tr.Resize(256),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    split_data = 'val'
    if args.dataset == 'refuge':
        split_data = 'testval'
    data_train = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='train',
                                       transform=composed_transforms_train)
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                  pin_memory=True)
    data_val = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split=split_data,
                                     transform=composed_transforms_val)
    dataloader_val = DataLoader(data_val, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    #############################需要修改的地方#############################
    # model_gen = get_fast_scnn().to('cuda')
    # model_gen = DeepLab(2, backbone="mobilenet", pretrained=False, downsample_factor=16).to('cuda')
    # model_gen = edgenext_small(classifier_dropout=0.5).to('cuda')
    model_gen = FastSCVM().to('cuda')
    # model_gen = LightMUNet(
    # spatial_dims = 2,
    # init_filters = 32,
    # in_channels=3,
    # out_channels=2,
    # blocks_down=[1,2, 4, 4,8],
    # blocks_up=[8,8,8,8],).cuda()
    start_epoch = 0
    start_iteration = 0

    # 3. optimizer
    optim_gen = torch.optim.Adam(
        model_gen.parameters(),
        lr=args.lr_gen,
        betas=(0.9, 0.99)
    )

    trainer = Trainer.Trainer(
        cuda=cuda,
        model_gen=model_gen,
        optimizer_gen=optim_gen,
        lr_gen=args.lr_gen,
        lr_decrease_rate=args.lr_decrease_rate,
        train_loader=dataloader_train,
        validation_loader=dataloader_val,
        out=args.out,
        max_epoch=args.max_epoch,
        stop_epoch=args.stop_epoch,
        interval_validate=args.interval_validate,
        batch_size=args.batch_size,
        warmup_epoch=args.warmup_epoch,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
