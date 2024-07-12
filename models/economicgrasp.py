import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME

from models.backbone import TDUnet
from models.modules_economicgrasp import GraspableNet, RotationNet, Cylinder_Grouping_Local_Interaction, Grasp_Head_Globle_Interaction
from utils.label_generation import process_grasp_labels
from libs.pointnet2.pointnet2_utils import furthest_point_sample, gather_operation
from utils.arguments import cfgs


class economicgrasp(nn.Module):
    def __init__(self, cylinder_radius=0.05, seed_feat_dim=512, is_training=True, voxel_size=0.005):
        super().__init__()
        self.is_training = is_training
        self.seed_feature_dim = seed_feat_dim
        self.num_depth = cfgs.num_depth
        self.num_angle = cfgs.num_angle
        self.M_points = cfgs.m_point
        self.num_view = cfgs.num_view
        self.voxel_size = voxel_size

        # Backbone
        self.backbone = TDUnet(in_channels=3, out_channels=self.seed_feature_dim, D=3)

        # Objectness and graspness
        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)

        # View Selection
        self.rotation = RotationNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)

        # Cylinder Grouping
        self.cy_group = Cylinder_Grouping_Local_Interaction(nsample=16, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim)

        # Depth and Score searching
        self.grasp_head = Grasp_Head_Globle_Interaction(num_angle=self.num_angle, num_depth=self.num_depth)

    def forward(self, end_points):
        seed_xyz = end_points['point_clouds']  # use all sampled point cloud, [B, point_num (15000)， 3]
        B, point_num, _ = seed_xyz.shape  # batch _size

        # Generate input to meet the Minkowski Engine
        coordinates_batch, features_batch = ME.utils.sparse_collate(
                                             [coord for coord in end_points['coordinates_for_voxel']],
                                             [feat for feat in np.ones_like(seed_xyz.cpu()).astype(np.float32)])
        coordinates_batch, features_batch, _, end_points['quantize2original'] = \
            ME.utils.sparse_quantize(coordinates_batch, features_batch, return_index=True, return_inverse=True)

        coordinates_batch = coordinates_batch.cuda()
        features_batch = features_batch.cuda()

        end_points['coors'] = coordinates_batch  # [points of the whole scenes after quantize, 3(coors) + 1(index)]
        end_points['feats'] = features_batch  # [points of the whole scenes after quantize, 3 (input feature dim)]
        mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)

        # Minkowski Backbone
        seed_features = self.backbone(mink_input).F
        seed_features = seed_features[end_points['quantize2original']].view(B, point_num, -1).transpose(1, 2)  # [B (batch size), 512 (feature dim), 20000 (points in a scene)]

        # Generate the masks of the objectness and the graspness
        end_points = self.graspable(seed_features, end_points)
        seed_features_flipped = seed_features.transpose(1, 2)
        objectness_score = end_points['objectness_score']  # [B (batch size), 2 (object classification), 20000 (points in a scene)]
        graspness_score = end_points['graspness_score'].squeeze(1)  # [B (batch size), 20000 (points in a scene)]
        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = (objectness_pred == 1)
        graspness_mask = graspness_score > cfgs.graspness_threshold
        graspable_mask = objectness_mask & graspness_mask

        # Generate the downsample point (1024 per scene) using the furthest point sampling
        seed_features_graspable = []
        seed_xyz_graspable = []
        graspable_num_batch = 0.
        for i in range(B):
            cur_mask = graspable_mask[i]
            graspable_num_batch += cur_mask.sum()
            cur_feat = seed_features_flipped[i][cur_mask]
            cur_seed_xyz = seed_xyz[i][cur_mask]

            cur_seed_xyz = cur_seed_xyz.unsqueeze(0)
            fps_idxs = furthest_point_sample(cur_seed_xyz, self.M_points)
            cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()
            cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous()
            cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()
            cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous()

            seed_features_graspable.append(cur_feat)
            seed_xyz_graspable.append(cur_seed_xyz)
        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)  # [B (batch size), 512 (feature dim), 1024 (points after sample)]
        seed_features_graspable = torch.stack(seed_features_graspable)
        end_points['xyz_graspable'] = seed_xyz_graspable
        end_points['D: Graspable Points'] = graspable_num_batch / B

        # Select the view for each point
        end_points, res_feat = self.rotation(seed_features_graspable, end_points)  # [B (batch size), 512 (feature dim), 1024 (points after sample)]
        seed_features_graspable = seed_features_graspable + res_feat  # [B (batch size), 512 (feature dim), 1024 (points after sample)]

        # Generate the labels
        if self.is_training:
            # generate the scene-level grasp labels from the object-level grasp label and the object poses
            # map the scene sampled points to the labeled object points
            # (note that the labeled object points and the sampled points may not 100% match due to the sampling and argumentation)
            grasp_top_views_rot, end_points = process_grasp_labels(end_points)
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']

        # Cylinder grouping
        group_features = self.cy_group(seed_xyz_graspable.contiguous(),
                                   seed_features_graspable.contiguous(),
                                   grasp_top_views_rot)

        # Width and score predicting
        end_points = self.grasp_head(group_features, end_points)

        return end_points


# score cls
def pred_decode(end_points):
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    for i in range(batch_size):
        grasp_center = end_points['xyz_graspable'][i].float()

        # composite score estimation
        grasp_score_prob = end_points['grasp_score_pred'][i].float()
        score = torch.tensor([0, 0.2, 0.4, 0.6, 0.8, 1]).view(-1, 1).expand(-1, grasp_score_prob.shape[1]).to(grasp_score_prob)
        score = torch.sum(score * grasp_score_prob, dim=0)
        grasp_score = score.view(-1, 1)

        grasp_angle_pred = end_points['grasp_angle_pred'][i].float()
        grasp_angle, grasp_angle_indxs = torch.max(grasp_angle_pred.squeeze(0), 0)
        grasp_angle = grasp_angle_indxs * np.pi / 12

        grasp_depth_pred = end_points['grasp_depth_pred'][i].float()
        grasp_depth, grasp_depth_indxs = torch.max(grasp_depth_pred.squeeze(0), 0)
        grasp_depth = (grasp_depth_indxs + 1) * 0.01
        grasp_depth = grasp_depth.view(-1, 1)

        grasp_width = 1.2 * end_points['grasp_width_pred'][i] / 10.
        grasp_width = torch.clamp(grasp_width, min=0., max=cfgs.grasp_max_width)
        grasp_width = grasp_width.view(-1, 1)

        approaching = -end_points['grasp_top_view_xyz'][i].float()
        grasp_rot = batch_viewpoint_params_to_matrix(approaching, grasp_angle)
        grasp_rot = grasp_rot.view(cfgs.m_point, 9)

        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(
            torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, grasp_rot, grasp_center, obj_ids], axis=-1))
    return grasp_preds