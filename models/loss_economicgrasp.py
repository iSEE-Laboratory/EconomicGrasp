import torch.nn as nn
import torch
from utils.arguments import cfgs
import pdb


def get_loss(end_points):
    # graspness loss and objectness loss
    objectness_loss, end_points = compute_objectness_loss(end_points)
    graspness_loss, end_points = compute_graspness_loss(end_points)

    # view selecting loss
    view_loss, end_points = compute_view_graspness_loss(end_points)

    # grasp match loss
    angle_loss, end_points = compute_angle_loss(end_points)
    depth_loss, end_points = compute_depth_loss(end_points)
    score_loss, end_points = compute_score_loss_cls(end_points)  # use classification to learn score
    width_loss, end_points = compute_width_loss(end_points)

    loss = cfgs.objectness_loss_weight * objectness_loss + \
           cfgs.graspness_loss_weight * graspness_loss + \
           cfgs.view_loss_weight * view_loss + \
           cfgs.angle_loss_weight * angle_loss + \
           cfgs.depth_loss_weight * depth_loss + \
           cfgs.score_loss_weight * score_loss + \
           cfgs.width_loss_weight * width_loss

    end_points['A: Overall Loss'] = loss

    return loss, end_points


def compute_objectness_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    objectness_score = end_points['objectness_score']
    objectness_label = end_points['objectness_label']
    loss = criterion(objectness_score, objectness_label)
    end_points['B: Objectness Loss'] = loss

    objectness_pred = torch.argmax(objectness_score, 1)
    end_points['D: Objectness Acc'] = (objectness_pred == objectness_label.long()).float().mean()
    return loss, end_points


def compute_graspness_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    graspness_score = end_points['graspness_score'].squeeze(1)
    graspness_label = end_points['graspness_label'].squeeze(-1)
    loss_mask = end_points['objectness_label'].bool()
    loss = criterion(graspness_score, graspness_label)
    loss = loss[loss_mask]
    loss = loss.mean()

    end_points['B: Graspness Loss'] = loss
    return loss, end_points


def compute_view_graspness_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='mean')
    view_score = end_points['view_score']
    view_label = end_points['batch_grasp_view_graspness']
    loss = criterion(view_score, view_label)
    end_points['B: View Loss'] = loss
    return loss, end_points


def compute_angle_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction='none')
    grasp_angle_pred = end_points['grasp_angle_pred']
    grasp_angle_label = end_points['batch_grasp_rotations'].long()
    valid_mask = end_points['batch_valid_mask']
    loss = criterion(grasp_angle_pred, grasp_angle_label)
    if torch.sum(valid_mask) == 0:
        loss = 0 * torch.sum(loss)
    else:
        loss = loss[valid_mask].mean()
    end_points['B: Angle Loss'] = loss
    end_points['D: Angle Acc'] = (torch.argmax(grasp_angle_pred, 1) == grasp_angle_label)[valid_mask].float().mean()

    return loss, end_points


def compute_depth_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction='none')
    grasp_depth_pred = end_points['grasp_depth_pred']
    grasp_depth_label = end_points['batch_grasp_depth'].long()
    valid_mask = end_points['batch_valid_mask']
    loss = criterion(grasp_depth_pred, grasp_depth_label)
    if torch.sum(valid_mask) == 0:
        loss = 0 * torch.sum(loss)
    else:
        loss = loss[valid_mask].mean()
    end_points['B: Depth Loss'] = loss
    end_points['D: Depth Acc'] = (torch.argmax(grasp_depth_pred, 1) == grasp_depth_label)[valid_mask].float().mean()
    return loss, end_points


def compute_score_loss_cls(end_points):
    criterion = nn.CrossEntropyLoss(reduction='none')
    grasp_score_pred = end_points['grasp_score_pred']
    grasp_score_label = (end_points['batch_grasp_score'] * 10 / 2).long()
    valid_mask = end_points['batch_valid_mask']
    loss = criterion(grasp_score_pred.squeeze(1), grasp_score_label)
    if torch.sum(valid_mask) == 0:
        loss = 0 * torch.sum(loss)
    else:
        loss = loss[valid_mask].mean()
    end_points['B: Score Loss'] = loss
    end_points['D: Score Acc'] = (torch.argmax(grasp_score_pred, 1) == grasp_score_label)[valid_mask].float().mean()
    return loss, end_points


def compute_width_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    grasp_width_pred = end_points['grasp_width_pred']
    grasp_width_label = end_points['batch_grasp_width'] * 10
    valid_mask = end_points['batch_valid_mask']
    loss = criterion(grasp_width_pred.squeeze(1), grasp_width_label)
    if torch.sum(valid_mask) == 0:
        loss = 0 * torch.sum(loss)
    else:
        loss = loss[valid_mask].mean()
    end_points['B: Width Loss'] = loss
    return loss, end_points
