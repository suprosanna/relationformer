import os
import yaml
import sys
import json
from argparse import ArgumentParser
import numpy as np
from shutil import copyfile
import logging
import ignite
import torch
import torch.nn as nn
from monai.data import DataLoader, DistributedSampler
from dataset_vessel3d import build_vessel_data
from evaluator import build_evaluator
from trainer import build_trainer
from models import build_model
from utils import image_graph_collate
from models.matcher import build_matcher
from losses import SetCriterion
import torch.distributed as dist
import ignite.distributed as igdist
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from torch.utils.tensorboard import SummaryWriter

parser = ArgumentParser()
parser.add_argument('--config',
                    default=None,
                    help='config file (.yml) containing the hyper-parameters for training. '
                         'If None, use the nnU-Net config. See /config for examples.')
parser.add_argument('--resume', default=None, help='checkpoint of the last epoch of the model')
parser.add_argument('--device', default='cuda',
                        help='device to use for training')
parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[0,1],
                        help='list of index where skip conn will be made')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--nproc_per_node", default=None, type=int)

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

def main(rank, args):
    
    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config['log']['exp_name'])
    config = dict2obj(config)

    exp_path = os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED))
    if os.path.exists(exp_path) and args.resume == None:
        print('ERROR: Experiment folder exist, please change exp name in config file')
    else:
        try:
            os.makedirs(exp_path)
            copyfile(args.config, os.path.join(exp_path, "config.yaml"))
        except:
            pass


    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    # device = torch.device("cuda") if args.device=='cuda' else torch.device("cpu")
    args.distributed = False
    args.rank = rank  # args.rank = int(os.environ["RANK"])
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.gpu = int(os.environ["LOCAL_RANK"])  # args.gpu = 'cuda:%d' % args.local_rank
        args.world_size = int(os.environ['WORLD_SIZE'])  # igdist.get_world_size()
        print('Running Distributed:',args.distributed, '; GPU:', args.gpu, '; RANK:', args.rank)

    if igdist.get_local_rank() > 0:
        # Ensure that only local rank 0 download the dataset
        # Thus each node will download a copy of the dataset
        igdist.barrier()

    train_ds, val_ds = build_vessel_data(config,
                                         mode='split',
                                         )

    if igdist.get_local_rank() == 0:
        # Ensure that only local rank 0 download the dataset
        igdist.barrier()

    train_loader = igdist.auto_dataloader(train_ds,
                              batch_size=config.DATA.BATCH_SIZE,
                              shuffle=True,
                              num_workers=config.DATA.NUM_WORKERS,
                              collate_fn=image_graph_collate,
                              pin_memory=True)
    
    val_loader = igdist.auto_dataloader(val_ds,
                            batch_size=config.DATA.BATCH_SIZE,
                            shuffle=False,
                            num_workers=config.DATA.NUM_WORKERS,
                            collate_fn=image_graph_collate,
                            pin_memory=True)

    device = torch.device(args.device)
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        # dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)
        args.rank = igdist.get_rank()
        device = torch.device(f"cuda:{args.rank}")

    net = build_model(config)

    net_wo_dist = net.to(device)
    relation_embed = net.relation_embed.to(device)

    net = igdist.auto_model(net)
    relation_embed = igdist.auto_model(relation_embed)

    if args.distributed:
        net_wo_dist = net.module

    matcher = build_matcher(config)
    loss = SetCriterion(config, matcher, relation_embed)

    optimizer = torch.optim.AdamW(
        net_wo_dist.parameters(), lr=float(config.TRAIN.BASE_LR), weight_decay=float(config.TRAIN.WEIGHT_DECAY)
    )
    optimizer = igdist.auto_optim(optimizer)

    # LR schedular
    iter_per_epoch = len(train_loader)
    num_warmup_epoch = float(config.TRAIN.WARMUP_EPOCHS)
    warm_lr_init = float(config.TRAIN.WARMUP_LR)
    warm_lr_final = float(config.TRAIN.BASE_LR)
    num_warmup_iter = num_warmup_epoch * iter_per_epoch
    num_after_warmup_iter = config.TRAIN.EPOCHS * iter_per_epoch
    def lr_lambda_polynomial(iter: int):
        if iter < num_warmup_epoch * iter_per_epoch:
            lr_lamda0 = warm_lr_init / warm_lr_final
            return lr_lamda0 + (1 - lr_lamda0) * iter / num_warmup_iter
        else:
            # The total number of epochs is num_warmup_epoch + max_epochs
            return (1 - (iter - num_warmup_iter) / num_after_warmup_iter) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_polynomial)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net_wo_dist.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    writer = SummaryWriter(
        log_dir=os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED)),
    )

    evaluator = build_evaluator(
        val_loader,
        net,
        optimizer,
        scheduler,
        writer,
        config,
        device
    )
    trainer = build_trainer(
        train_loader,
        net,
        loss,
        optimizer,
        scheduler,
        writer,
        evaluator,
        config,
        device,
        # fp16=args.fp16,
    )

    if args.resume:
        last_epoch = int(scheduler.last_epoch/trainer.state.epoch_length)
        evaluator.state.epoch = last_epoch
        trainer.state.epoch = last_epoch
        trainer.state.iteration = trainer.state.epoch_length * last_epoch
    if dist.get_rank()==0:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform= lambda x: {'loss': x["loss"]["total"]})
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    trainer.run()


if __name__ == '__main__':
    args = parser.parse_args()
    if args.cuda_visible_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))

    with igdist.Parallel(backend='nccl', nproc_per_node=args.nproc_per_node) as parallel:
        parallel.run(main, args)