import argparse

parser = argparse.ArgumentParser()

# data relevant
parser.add_argument('--dataset_root', required=True, help='Dataset root')
parser.add_argument('--camera', required=True, help='Camera split [realsense/kinect]')

# log
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')

# model setting
parser.add_argument('--model', type=str, default='graspness', help='graspnessp')
parser.add_argument('--num_point', type=int, default=20000, help='Point number [default: 20000]')

# graspness setting
parser.add_argument('--grasp_max_width', type=float, default=0.1, help='The max width of the grasp [default: 0.1]')
parser.add_argument('--graspness_threshold', type=float, default=0.1,
                    help='Threshold of the graspness [default: 0.1], less than the threshold is unacceptable')
parser.add_argument('--num_view', type=int, default=300, help='Loss weight of the objectness term')
parser.add_argument('--num_angle', type=int, default=12, help='Loss weight of the objectness term')
parser.add_argument('--num_depth', type=int, default=4, help='Loss weight of the objectness term')
parser.add_argument('--m_point', type=int, default=1024, help='Point number after graspness [default: 1024]')

# loss setting
parser.add_argument('--objectness_loss_weight', type=float, default=1, help='Loss weight of the objectness term')
parser.add_argument('--graspness_loss_weight', type=float, default=10, help='Loss weight of the graspness term')
parser.add_argument('--view_loss_weight', type=float, default=100, help='Loss weight of the view term')
parser.add_argument('--score_loss_weight', type=float, default=15, help='Loss weight of the score term')
parser.add_argument('--width_loss_weight', type=float, default=10, help='Loss weight of the width term')

# training setting
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--resume', action='store_true', help='Whether to resume from checkpoint')
parser.add_argument('--max_epoch', type=int, default=20, help='Epoch to run [default: 18]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel size (in metters)')

# testing setting
parser.add_argument('--save_dir', type=str, help='Dir to save outputs')
parser.add_argument('--test_mode', type=str, help='Mode of the testing (seen, similar, novel)')
parser.add_argument('--collision_thresh', type=float, default=0,
                    help='Collision threshold in collision detection [default: 0], if used, set to 0.01')
parser.add_argument('--inference', action='store_true', help='Whether to inference')


cfgs = parser.parse_args()