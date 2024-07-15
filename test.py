import os
import numpy as np
import time

import torch
from torch.utils.data import DataLoader
from graspnetAPI import GraspGroup, GraspNetEval

from utils.collision_detector import ModelFreeCollisionDetector
from utils.arguments import cfgs

from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from models.economicgrasp import economicgrasp, pred_decode

# ------------ GLOBAL CONFIG ------------
if not os.path.exists(cfgs.save_dir):
    os.mkdir(cfgs.save_dir)


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


# Create dataset and dataloader
if cfgs.test_mode == 'seen':
    TEST_DATASET = GraspNetDataset(cfgs.dataset_root, split='test_seen',
                                   camera=cfgs.camera, num_points=cfgs.num_point, remove_outlier=True, augment=False,
                                   load_label=False)
elif cfgs.test_mode == 'similar':
    TEST_DATASET = GraspNetDataset(cfgs.dataset_root, split='test_similar',
                                   camera=cfgs.camera, num_points=cfgs.num_point, remove_outlier=True, augment=False,
                                   load_label=False)
elif cfgs.test_mode == 'novel':
    TEST_DATASET = GraspNetDataset(cfgs.dataset_root, split='test_novel',
                                   camera=cfgs.camera, num_points=cfgs.num_point, remove_outlier=True, augment=False,
                                   load_label=False)

SCENE_LIST = TEST_DATASET.scene_list()
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
                             num_workers=2, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)

# Init the model
net = economicgrasp(seed_feat_dim=512, is_training=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# Load checkpoint
checkpoint = torch.load(cfgs.checkpoint_path)
net.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))


# ------ Testing ------------
def inference():
    batch_interval = 20
    stat_dict = {}  # collect statistics
    # set model to eval mode (for bn and dp)
    net.eval()
    tic = time.time()
    for batch_idx, batch_data in enumerate(TEST_DATALOADER):
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            elif 'graph' in key:
                for i in range(len(batch_data[key])):
                    batch_data[key][i] = batch_data[key][i].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)

        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points)

        # Save results for evaluation
        for i in range(cfgs.batch_size):
            data_idx = batch_idx * cfgs.batch_size + i
            preds = grasp_preds[i].detach().cpu().numpy()
            gg = GraspGroup(preds)

            # collision detection
            if cfgs.collision_thresh > 0:
                cloud, _ = TEST_DATASET.get_data(data_idx, return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
                gg = gg[~collision_mask]

            # save grasps
            save_dir = os.path.join(cfgs.save_dir, SCENE_LIST[data_idx], cfgs.camera)
            save_path = os.path.join(save_dir, str(data_idx % 256).zfill(4) + '.npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gg.save_npy(save_path)

        if batch_idx % batch_interval == 0:
            toc = time.time()
            print('Eval batch: %d, time: %fs' % (batch_idx, (toc - tic) / batch_interval))
            tic = time.time()


def evaluate_seen():
    ge = GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split='test')
    # In test time, we will select top-10 grasps for each objects (sorted by our predicted score).
    # Then, for all the grasp, we will further select the top-50 grasps for evaluation.
    res, ap = ge.eval_seen(cfgs.save_dir, proc=6)
    save_dir = os.path.join(cfgs.save_dir, 'ap_{}_seen.npy'.format(cfgs.camera))
    np.save(save_dir, res)
    print(f"seen testing, AP 0.8={np.mean(res[:, :, :, 3])}, AP 0.4={np.mean(res[:, :, :, 1])}")


def evaluate_similar():
    ge = GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split='test')
    # In test time, we will select top-10 grasps for each objects (sorted by our predicted score).
    # Then, for all the grasp, we will further select the top-50 grasps for evaluation.
    res, ap = ge.eval_similar(cfgs.save_dir, proc=6)
    save_dir = os.path.join(cfgs.save_dir, 'ap_{}_similar.npy'.format(cfgs.camera))
    np.save(save_dir, res)
    print(f"similar testing, AP 0.8={np.mean(res[:, :, :, 3])}, AP 0.4={np.mean(res[:, :, :, 1])}")


def evaluate_novel():
    ge = GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split='test')
    # In test time, we will select top-10 grasps for each objects (sorted by our predicted score).
    # Then, for all the grasp, we will further select the top-50 grasps for evaluation.
    res, ap = ge.eval_novel(cfgs.save_dir, proc=6)
    save_dir = os.path.join(cfgs.save_dir, 'ap_{}_novel.npy'.format(cfgs.camera))
    np.save(save_dir, res)
    print(f"novel testing, AP 0.8={np.mean(res[:, :, :, 3])}, AP 0.4={np.mean(res[:, :, :, 1])}")


if __name__ == '__main__':
    if cfgs.inference:
        inference()
    if cfgs.test_mode == 'seen':
        evaluate_seen()
    elif cfgs.test_mode == 'similar':
        evaluate_similar()
    elif cfgs.test_mode == 'novel':
        evaluate_novel()
