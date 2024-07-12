# Basic Libraries
import os
import numpy as np
import math
import time

# PyTorch Libraries
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Config
from utils.arguments import cfgs

# Local Libraries
from models.economicgrasp import economicgrasp
from models.loss_economicgrasp import get_loss as get_loss_economicgrasp
from dataset.graspnet_dataset import GraspNetDataset, collate_fn

# ----------- GLOBAL CONFIG ------------

# Epoch
EPOCH_CNT = 0

# Checkpoint path
CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None and cfgs.resume else None

# Logging
if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)
LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')
LOG_FOUT.write(str(cfgs) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


# Create Dataset and Dataloader
TRAIN_DATASET = GraspNetDataset(cfgs.dataset_root, camera=cfgs.camera, split='train',
                                voxel_size=cfgs.voxel_size, num_points=cfgs.num_point, remove_outlier=True,
                                augment=True)
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
                              num_workers=2, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)

# Init the model and optimzier
net = economicgrasp(seed_feat_dim=512, is_training=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate, weight_decay=cfgs.weight_decay)

# Load checkpoint if there is any
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))


# cosine learning rate decay
def get_current_lr(epoch):
    lr = cfgs.learning_rate
    lr = lr * (math.cos(epoch / cfgs.max_epoch * math.pi) + 1) * 0.5
    return lr


def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------TRAINING BEGIN  ------------
def train_one_epoch():
    stat_dict = {}  # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    # set model to training mode
    net.train()
    batch_start_time = time.time()
    data_start_time = time.time()
    num_batches = len(TRAIN_DATALOADER)
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            elif 'graph' in key:
                for i in range(len(batch_data_label[key])):
                    batch_data_label[key][i] = batch_data_label[key][i].to(device)
            else:
                batch_data_label[key] = batch_data_label[key].to(device)
        data_end_time = time.time()
        stat_dict['C: Data Time'] = data_end_time - data_start_time

        model_start_time = time.time()
        end_points = net(batch_data_label)
        model_end_time = time.time()
        stat_dict['C: Model Time'] = model_end_time - model_start_time
        end_points['epoch'] = EPOCH_CNT

        loss_start_time = time.time()
        # Compute loss and gradients, update parameters.
        loss, end_points = get_loss_economicgrasp(end_points)

        loss.backward()
        if (batch_idx + 1) % 1 == 0:
            optimizer.step()
            optimizer.zero_grad()
        loss_end_time = time.time()
        stat_dict['C: Loss Time'] = loss_end_time - loss_start_time

        # Accumulate statistics and print out
        for key in end_points:
            if 'A' in key or 'B' in key or 'C' in key or 'D' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 20

        if (batch_idx + 1) % batch_interval == 0:
            remain_batches = (cfgs.max_epoch - EPOCH_CNT) * num_batches - batch_idx - 1
            batch_time = time.time() - batch_start_time
            batch_start_time = time.time()
            stat_dict['C: Remain Time (h)'] = remain_batches * batch_time / 3600
            log_string(f' ---- epoch: {EPOCH_CNT},  batch: {batch_idx + 1} ----')
            for key in sorted(stat_dict.keys()):
                log_string(f'{key:<20}: {round(stat_dict[key] / batch_interval, 4):0<8}')
                stat_dict[key] = 0

        data_start_time = time.time()


def train(start_epoch):
    global EPOCH_CNT
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        log_string(f'**** EPOCH {epoch:<3} ****')
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))

        np.random.seed()
        train_one_epoch()

        # Save checkpoint
        save_dict = {'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
                     'optimizer_state_dict': optimizer.state_dict(),
                     }
        try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        torch.save(save_dict, os.path.join(cfgs.log_dir, cfgs.model + '_epoch' + str(epoch + 1).zfill(2) + '.tar'))


if __name__ == '__main__':
    train(start_epoch)
