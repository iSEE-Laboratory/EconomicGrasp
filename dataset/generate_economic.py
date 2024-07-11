import os
import sys
import torch
import numpy as np
import scipy.io as scio
from PIL import Image
import pdb
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/home/xiaoming/dataset/graspnet', help='the root of the GraspNet dataset')
parser.add_argument('--camera_type', default='kinect', help='Camera split [realsense/kinect]')

cfgs = parser.parse_args()

obj_data_folders = os.path.join(cfgs.dataset_root, "grasp_label")
scenes_data_folders = os.path.join(cfgs.dataset_root, "scenes")
collision_data_folders = os.path.join(cfgs.dataset_root, "collision_label")


if __name__ == "__main__":
    keeping_views_numbers = 300

    save_data_folders = os.path.join(cfgs.dataset_root, f"economic_grasp_label_{keeping_views_numbers}views")
    if not os.path.exists(save_data_folders):
        os.makedirs(save_data_folders)

    # collect the labels from object-level to scene-level
    number = 0
    for label_path in os.listdir(scenes_data_folders):
        print(f"---------The {number} scenes----------")
        meta = scio.loadmat(os.path.join(scenes_data_folders, label_path, cfgs.camera_type, 'meta', '0000.mat'))
        obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
        object_list = open(os.path.join(scenes_data_folders, label_path, 'object_id_list.txt'), "r")
        scene_collision = np.load(os.path.join(collision_data_folders, label_path, 'collision_labels.npz'))
        scene_points = []
        scene_pointid = []
        scene_scores = []
        scene_width = []
        for i, obj_idx in enumerate(obj_idxs):
            object_labels = np.load(os.path.join(obj_data_folders, f"{str(obj_idx - 1).zfill(3)}_labels.npz"))
            points = torch.from_numpy(object_labels['points'])
            pointid = torch.ones(points.shape[0]) * i
            width = torch.from_numpy(object_labels['offsets'][:, :, :, :, 2])
            scores = torch.from_numpy(object_labels['scores'])
            collision = torch.from_numpy(scene_collision[f'arr_{i}'])
            scores[collision] = 0
            scores[scores < 0] = 0
            scene_points.append(points)
            scene_pointid.append(pointid)
            scene_scores.append(scores)
            scene_width.append(width)
        scene_points = torch.cat(scene_points, dim=0)
        scene_pointid = torch.cat(scene_pointid, dim=0)
        scene_scores = torch.cat(scene_scores, dim=0)
        scene_width = torch.cat(scene_width, dim=0)

        # filtering labels in bad points
        threshold = 0.4
        Ns, V, A, D = scene_scores.size()
        grasp_num = V * A * D
        grasp_mask = (scene_scores <= threshold) & (scene_scores > 0)
        grasp_mask = grasp_mask.float()
        grasp_mask = grasp_mask.view(Ns, -1)
        graspness = torch.sum(grasp_mask, dim=-1).float() / grasp_num  # [objects points, 1]
        filter_mask = (graspness > 0)
        ori_number = scene_points.shape[0]
        scene_points = scene_points[filter_mask]
        scene_pointid = scene_pointid[filter_mask]
        scene_scores = scene_scores[filter_mask]
        scene_width = scene_width[filter_mask]
        result_number = scene_points.shape[0]
        print(result_number, ori_number)

        # compute view graspness
        view_u_threshold = 0.6
        grasp_view_valid_mask = (scene_scores <= view_u_threshold) & (scene_scores > 0)
        grasp_view_valid = grasp_view_valid_mask.float()
        grasp_view_graspness = torch.sum(torch.sum(grasp_view_valid, dim=-1), dim=-1) / 48  # (Ns, V)
        view_graspness_min, _ = torch.min(grasp_view_graspness, dim=-1)  # (Ns)
        view_graspness_max, _ = torch.max(grasp_view_graspness, dim=-1)
        view_graspness_max = view_graspness_max.unsqueeze(-1).expand(-1, 300)  # (Ns, V)
        view_graspness_min = view_graspness_min.unsqueeze(-1).expand(-1,
                                                                     300)  # same shape as batch_grasp_view_graspness
        grasp_view_graspness = (grasp_view_graspness - view_graspness_min) / (
                view_graspness_max - view_graspness_min + 1e-5)  # (Ns, V)

        # nomalize the score
        label_mask = (scene_scores > 0) & (scene_width <= 0.1)
        scene_scores[~label_mask] = 0
        po_mask = scene_scores > 0
        scene_scores[po_mask] = 1.1 - scene_scores[po_mask]

        # only keeping the views
        grasp_score_label = scene_scores
        grasp_width_label = scene_width
        grasp_score_label_max_depth, grasp_score_label_max_depth_idx = grasp_score_label.max(-1)
        grasp_width_label = grasp_width_label.gather(-1, grasp_score_label_max_depth_idx.unsqueeze(-1)).squeeze(-1)
        grasp_score_label_max_angle, grasp_score_label_max_angle_idx = grasp_score_label_max_depth.max(-1)
        scene_depth = grasp_score_label_max_depth_idx.gather(-1, grasp_score_label_max_angle_idx.unsqueeze(-1)).squeeze(
            -1)
        scene_rotations = grasp_score_label_max_angle_idx
        scene_scores = grasp_score_label_max_angle
        scene_width = grasp_width_label.gather(-1, grasp_score_label_max_angle_idx.unsqueeze(-1)).squeeze(-1)

        # further view filtering
        values, index = torch.topk(grasp_view_graspness, k=keeping_views_numbers)
        scene_rotations = torch.gather(scene_rotations, 1, index)
        scene_depth = torch.gather(scene_depth, 1, index)
        scene_scores = torch.gather(scene_scores, 1, index)
        scene_width = torch.gather(scene_width, 1, index)
        scene_top_view_index = index

        # save the results
        scene_points = scene_points.numpy()
        grasp_rotations = scene_rotations.numpy().astype(np.uint8)
        grasp_depth = scene_depth.numpy().astype(np.uint8)
        grasp_scores = (scene_scores.numpy() * 10).astype(np.uint8)
        grasp_widths = (scene_width.numpy() * 1000).astype(np.uint8)
        scene_pointid = scene_pointid.numpy().astype(np.uint8)
        grasp_view_graspness = grasp_view_graspness.numpy()
        grasp_top_view_index = scene_top_view_index.numpy().astype(np.uint16)

        np.savez(os.path.join(save_data_folders, f"{label_path}_labels.npz"),
                 points=scene_points,
                 rotations=grasp_rotations,
                 depth=grasp_depth,
                 scores=grasp_scores,
                 widths=grasp_widths,
                 pointid=scene_pointid,
                 vgraspness=grasp_view_graspness,
                 topview=grasp_top_view_index)

        number = number + 1

    print(f"---------Finishing----------")

