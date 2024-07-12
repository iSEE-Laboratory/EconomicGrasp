import os
import sys
import time
import pdb

import torch
import os
import sys
import torch

from libs.knn.knn_modules import knn
from utils.loss_utils import (batch_viewpoint_params_to_matrix, transform_point_cloud,
                              generate_grasp_views, compute_pointwise_dists)
from utils.arguments import cfgs


def process_grasp_labels(end_points):
    """ Process labels according to scene points and object poses. """
    seed_xyzs = end_points['xyz_graspable']  # [B (batch size), 1024 (scene points after sample), 3]
    pred_top_view_inds = end_points['grasp_top_view_inds']  # [B (batch size), 1024 (scene points after sample)]
    batch_size, num_samples, _ = seed_xyzs.size()

    valid_points_count = 0
    valid_views_count = 0

    batch_grasp_points = []
    batch_grasp_views_rot = []
    batch_view_graspness = []
    batch_grasp_rotations = []
    batch_grasp_depth = []
    batch_grasp_scores = []
    batch_grasp_widths = []
    batch_valid_mask = []
    for i in range(batch_size):
        seed_xyz = seed_xyzs[i]  # [1024 (scene points after sample), 3]
        pred_top_view = pred_top_view_inds[i]  # [1024 (scene points after sample)]
        poses = end_points['object_poses_list'][i]  # a list with length of object amount, each has size [3, 4]

        # get merged grasp points for label computation
        # transform the view from object coordinate system to scene coordinate system
        grasp_points_merged = []
        grasp_views_rot_merged = []
        grasp_rotations_merged = []
        grasp_depth_merged = []
        grasp_scores_merged = []
        grasp_widths_merged = []
        view_graspness_merged = []
        top_view_index_merged = []
        for obj_idx, pose in enumerate(poses):
            grasp_points = end_points['grasp_points_list'][i][obj_idx]  # [objects points, 3]
            grasp_rotations = end_points['grasp_rotations_list'][i][obj_idx]  # [objects points, num_of_view]
            grasp_depth = end_points['grasp_depth_list'][i][obj_idx]  # [objects points, num_of_view]
            grasp_scores = end_points['grasp_scores_list'][i][obj_idx]  # [objects points, num_of_view]
            grasp_widths = end_points['grasp_widths_list'][i][obj_idx]  # [objects points, num_of_view]
            view_graspness = end_points['view_graspness_list'][i][obj_idx]  # [objects points, 300]
            top_view_index = end_points['top_view_index_list'][i][obj_idx]  # [objects points, num_of_view]
            num_grasp_points = grasp_points.size(0)

            # generate and transform template grasp views
            grasp_views = generate_grasp_views(cfgs.num_view).to(pose.device)  # [300 (views), 3 (coordinate)]
            grasp_points_trans = transform_point_cloud(grasp_points, pose, '3x4')
            grasp_views_trans = transform_point_cloud(grasp_views, pose[:3, :3], '3x3')
            # [300 (views), 3 (coordinate)], after translation to scene coordinate system

            # generate and transform template grasp view rotation
            angles = torch.zeros(grasp_views.size(0), dtype=grasp_views.dtype, device=grasp_views.device)
            grasp_views_rot = batch_viewpoint_params_to_matrix(-grasp_views, angles)
            grasp_views_rot_trans = torch.matmul(pose[:3, :3], grasp_views_rot)
            # [300 (views), 3, 3 (the rotation matrix)]

            # assign views after transform (the view will not exactly match)
            grasp_views_ = grasp_views.transpose(0, 1).contiguous().unsqueeze(0)
            grasp_views_trans_ = grasp_views_trans.transpose(0, 1).contiguous().unsqueeze(0)
            view_inds = knn(grasp_views_trans_, grasp_views_, k=1).squeeze() - 1  # [300]
            view_graspness_trans = torch.index_select(view_graspness, 1, view_inds)  # [object points, 300]
            grasp_views_rot_trans = torch.index_select(grasp_views_rot_trans, 0, view_inds)
            grasp_views_rot_trans = grasp_views_rot_trans.unsqueeze(0).expand(num_grasp_points, -1, -1, -1)
            # [object points, 300, 3, 3]

            # -1 means that when we transform the top 60 views into the scene coordinate,
            # some views will have no matching
            # It means that two views in the object coordinate match to one view in the scene coordinate
            top_view_index_trans = (-1 * torch.ones((num_grasp_points, grasp_rotations.shape[1]), dtype=torch.long)
                                    .to(seed_xyz.device))
            tpid, tvip, tids = torch.where(view_inds == top_view_index.unsqueeze(-1))
            top_view_index_trans[tpid, tvip] = tids  # [objects points, num_of_view]

            # add to list
            grasp_points_merged.append(grasp_points_trans)
            view_graspness_merged.append(view_graspness_trans)
            top_view_index_merged.append(top_view_index_trans)
            grasp_rotations_merged.append(grasp_rotations)
            grasp_depth_merged.append(grasp_depth)
            grasp_scores_merged.append(grasp_scores)
            grasp_widths_merged.append(grasp_widths)
            grasp_views_rot_merged.append(grasp_views_rot_trans)

        grasp_points_merged = torch.cat(grasp_points_merged, dim=0)  # [all object points, 3]
        view_graspness_merged = torch.cat(view_graspness_merged, dim=0)  # [all object points, 300]
        top_view_index_merged = torch.cat(top_view_index_merged, dim=0)  # [all object points, num_of_view]
        grasp_rotations_merged = torch.cat(grasp_rotations_merged, dim=0)  # [all object points, num_of_view]
        grasp_depth_merged = torch.cat(grasp_depth_merged, dim=0)  # [all object points, num_of_view]
        grasp_scores_merged = torch.cat(grasp_scores_merged, dim=0)  # [all object points, num_of_view]
        grasp_widths_merged = torch.cat(grasp_widths_merged, dim=0)  # [all object points, num_of_view]
        grasp_views_rot_merged = torch.cat(grasp_views_rot_merged, dim=0)  # [all object points, 300, 3, 3]

        # compute nearest neighbors
        seed_xyz_ = seed_xyz.transpose(0, 1).contiguous().unsqueeze(0)
        grasp_points_merged_ = grasp_points_merged.transpose(0, 1).contiguous().unsqueeze(0)
        nn_inds = knn(grasp_points_merged_, seed_xyz_, k=1).squeeze() - 1

        # assign anchor points to real points
        grasp_points_merged = torch.index_select(grasp_points_merged, 0, nn_inds)
        # [1024 (scene points after sample), 3]
        grasp_views_rot_merged = torch.index_select(grasp_views_rot_merged, 0, nn_inds)
        # [1024 (scene points after sample), 300, 3, 3]
        view_graspness_merged = torch.index_select(view_graspness_merged, 0, nn_inds)
        # [1024 (scene points after sample), 300]
        top_view_index_merged = torch.index_select(top_view_index_merged, 0, nn_inds)
        # [1024 (scene points after sample), num_of_view]
        grasp_rotations_merged = torch.index_select(grasp_rotations_merged, 0, nn_inds)
        # [1024 (scene points after sample), num_of_view]
        grasp_depth_merged = torch.index_select(grasp_depth_merged, 0, nn_inds)
        # [1024 (scene points after sample), num_of_view]
        grasp_scores_merged = torch.index_select(grasp_scores_merged, 0, nn_inds)
        # [1024 (scene points after sample), num_of_view]
        grasp_widths_merged = torch.index_select(grasp_widths_merged, 0, nn_inds)
        # [1024 (scene points after sample), num_of_view]

        # select top view's rot, score and width
        # we only assign labels when the pred view is in the pre-defined 60 top view, others are zero
        pred_top_view_ = pred_top_view.view(num_samples, 1, 1, 1).expand(-1, -1, 3, 3)
        # [1024 (points after sample), 1, 3, 3]
        top_grasp_views_rot = torch.gather(grasp_views_rot_merged, 1, pred_top_view_).squeeze(1)
        # [1024 (points after sample), 3, 3]
        pid, vid = torch.where(pred_top_view.unsqueeze(-1) == top_view_index_merged)
        # both pid and vid are [true numbers], where(condition) equals to nonzero(condition)
        top_grasp_rotations = 12 * torch.ones(num_samples, dtype=torch.int32).to(seed_xyz.device)
        # [1024 (points after sample)]
        top_grasp_depth = 4 * torch.ones(num_samples, dtype=torch.int32).to(seed_xyz.device)
        # [1024 (points after sample)]
        top_grasp_scores = torch.zeros(num_samples, dtype=torch.float32).to(seed_xyz.device)
        # [1024 (points after sample)]
        top_grasp_widths = 0.1 * torch.ones(num_samples, dtype=torch.float32).to(seed_xyz.device)
        # [1024 (points after sample)]
        top_grasp_rotations[pid] = torch.gather(grasp_rotations_merged[pid], 1, vid.view(-1, 1)).squeeze(1)
        top_grasp_depth[pid] = torch.gather(grasp_depth_merged[pid], 1, vid.view(-1, 1)).squeeze(1)
        top_grasp_scores[pid] = torch.gather(grasp_scores_merged[pid], 1, vid.view(-1, 1)).squeeze(1)
        top_grasp_widths[pid] = torch.gather(grasp_widths_merged[pid], 1, vid.view(-1, 1)).squeeze(1)

        # only compute loss in the points with correct matching (so compute the mask first)
        dist = compute_pointwise_dists(seed_xyz, grasp_points_merged)
        valid_point_mask = dist < 0.005
        valid_view_mask = torch.zeros(num_samples, dtype=torch.bool).to(seed_xyz.device)
        valid_view_mask[pid] = True
        valid_points_count = valid_points_count + torch.sum(valid_point_mask)
        valid_views_count = valid_views_count + torch.sum(valid_view_mask)
        valid_mask = valid_point_mask & valid_view_mask

        # add to batch
        batch_grasp_points.append(grasp_points_merged)
        batch_grasp_views_rot.append(top_grasp_views_rot)
        batch_view_graspness.append(view_graspness_merged)
        batch_grasp_rotations.append(top_grasp_rotations)
        batch_grasp_depth.append(top_grasp_depth)
        batch_grasp_scores.append(top_grasp_scores)
        batch_grasp_widths.append(top_grasp_widths)
        batch_valid_mask.append(valid_mask)

    batch_grasp_points = torch.stack(batch_grasp_points, 0)
    # [B (batch size), 1024 (scene points after sample), 3]
    batch_grasp_views_rot = torch.stack(batch_grasp_views_rot, 0)
    # [B (batch size), 1024 (scene points after sample), 3, 3]
    batch_view_graspness = torch.stack(batch_view_graspness, 0)
    # [B (batch size), 1024 (scene points after sample), 300]
    batch_grasp_rotations = torch.stack(batch_grasp_rotations, 0)
    # [B (batch size), 1024 (scene points after sample)]
    batch_grasp_depth = torch.stack(batch_grasp_depth, 0)
    # [B (batch size), 1024 (scene points after sample)]
    batch_grasp_scores = torch.stack(batch_grasp_scores, 0)
    # [B (batch size), 1024 (scene points after sample)]
    batch_grasp_widths = torch.stack(batch_grasp_widths, 0)
    # [B (batch size), 1024 (scene points after sample)]
    batch_valid_mask = torch.stack(batch_valid_mask, 0)
    # [B (batch size), 1024 (scene points after sample)]

    end_points['batch_grasp_point'] = batch_grasp_points
    end_points['batch_grasp_rotations'] = batch_grasp_rotations
    end_points['batch_grasp_depth'] = batch_grasp_depth
    end_points['batch_grasp_score'] = batch_grasp_scores
    end_points['batch_grasp_width'] = batch_grasp_widths
    end_points['batch_grasp_view_graspness'] = batch_view_graspness
    end_points['batch_valid_mask'] = batch_valid_mask
    end_points['C: Valid Points'] = valid_points_count / batch_size
    return batch_grasp_views_rot, end_points



