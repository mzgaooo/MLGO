import argparse

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')
    
parser = argparse.ArgumentParser(description='MLGO')

# general settings
parser.add_argument('--device', type=str, default='cuda:0', help='device to run')
parser.add_argument('--filename', type=str, default='pems-bay')
parser.add_argument('--save', type=str, default='./save/', help='model save path')
parser.add_argument('--save_preds', type=str_to_bool, default=False, help='whether to save prediction results')
parser.add_argument('--save_preds_path', type=str, default='./results/', help='predictions save path')
parser.add_argument('--num_nodes', type=int, default=325, help='number of nodes/variables')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')

# training related
parser.add_argument('--epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--lr', type=float, default=0.0004, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.00002, help='weight decay rate')
parser.add_argument('--lr_decay', type=str_to_bool, default=True, help='whether to decrease lr during training')
parser.add_argument('--lr_decay_steps', type=str, default='30,60', help='lr decay at this step')
parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='how much lr will decay')
parser.add_argument('--step_size', type=int, default=100, help='control the node permutation')

# model related
parser.add_argument('--sigma1', type=float, default=0.1, help='sigma for the semantic matrix')
parser.add_argument('--sigma2', type=float, default=10, help='sigma for the spatial matrix')
parser.add_argument('--thres1', type=float, default=0.6, help='the threshold for the semantic matrix')
parser.add_argument('--thres2', type=float, default=0.5, help='the threshold for the spatial matrix')
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
parser.add_argument('--subgraph_size', type=int, default=20, help='learned adj top-k sparse')
parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
parser.add_argument('--tanhalpha', type=float, default=3, help='saturation ratio in graph construction')
parser.add_argument('--conv_channels', type=int, default=64, help='convolution channels')
parser.add_argument('--end_channels', type=int, default=64, help='end channels')
parser.add_argument('--num_hidden_layers', type=int, default=4, help='num_hidden_layers')
parser.add_argument('--solver_1', type=str, default='euler', help='interlayer Solver')
parser.add_argument('--time_1', type=float, default=1.0, help='interlayer integration time')
parser.add_argument('--step_1', type=float, default=0.25, help='interlayer step size')
parser.add_argument('--solver_2', type=str, default='euler', help='inlayer Solver')
parser.add_argument('--time_2', type=float, default=1.0, help='inlayer integration time')
parser.add_argument('--step_2', type=float, default=0.5, help='inlayer step size')
parser.add_argument('--alpha', type=float, default=1.0, help='eigen normalization')
parser.add_argument('--rtol', type=float, default=1e-4, help='rtol')
parser.add_argument('--atol', type=float, default=1e-3, help='atol')
parser.add_argument('--train-ratio', type=float, default=0.6, help='the ratio of training dataset')
parser.add_argument('--valid-ratio', type=float, default=0.2, help='the ratio of validating dataset')

args = parser.parse_args()
