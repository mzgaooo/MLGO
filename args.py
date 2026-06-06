import argparse
import configparser

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser(description='MPGODE')
parser.add_argument('--config', type=str, default='./configs/pems04.conf', help='Path to configuration file')
args, _ = parser.parse_known_args()

config = configparser.ConfigParser()
config.read(args.config)

# general settings
parser.add_argument('--device', type=str, default=config['general']['device'], help='device to run')
parser.add_argument('--filename', type=str, default=config['general']['filename'], help='dataset name')
parser.add_argument('--save', type=str, default=config['general']['save'], help='model save path')
parser.add_argument('--seq_len', type=int, default=int(config['general']['seq_len']), help='input sequence length')
parser.add_argument('--num_nodes', type=int, default=int(config['general']['num_nodes']), help='number of nodes/variables')
parser.add_argument('--train_ratio', type=float, default=float(config['general']['train_ratio']), help='the ratio of training dataset')
parser.add_argument('--valid_ratio', type=float, default=float(config['general']['valid_ratio']), help='the ratio of validating dataset')

# training related
parser.add_argument('--epochs', type=int, default=int(config['train']['epochs']), help='train epochs')
parser.add_argument('--batch_size', type=int, default=int(config['train']['batch_size']), help='batch size')
parser.add_argument('--lr', type=float, default=float(config['train']['lr']), help='learning rate')
parser.add_argument('--weight_decay', type=float, default=float(config['train']['weight_decay']), help='weight decay rate')
parser.add_argument('--lr_decay', type=str_to_bool, default=str_to_bool(config['train']['lr_decay']), help='whether to decrease lr during training')
parser.add_argument('--lr_decay_steps', type=str, default=config['train']['lr_decay_steps'], help='lr decay at this step')
parser.add_argument('--lr_decay_rate', type=float, default=float(config['train']['lr_decay_rate']), help='how much lr will decay')
parser.add_argument('--dropout', type=float, default=float(config['train']['dropout']), help='dropout rate')
parser.add_argument('--step_size', type=int, default=int(config['train']['step_size']), help='control the node permutation')

# model related
parser.add_argument('--sigma1', type=float, default=float(config['model']['sigma1']), help='sigma for the semantic matrix')
parser.add_argument('--sigma2', type=float, default=float(config['model']['sigma2']), help='sigma for the spatial matrix')
parser.add_argument('--thres1', type=float, default=float(config['model']['thres1']), help='the threshold for the semantic matrix')
parser.add_argument('--thres2', type=float, default=float(config['model']['thres2']), help='the threshold for the spatial matrix')

parser.add_argument('--node_dim', type=int, default=int(config['model']['node_dim']), help='dim of nodes')
parser.add_argument('--subgraph_size', type=int, default=int(config['model']['subgraph_size']), help='learned adj top-k sparse')
parser.add_argument('--num_split', type=int, default=int(config['model']['num_split']), help='number of splits for graphs')
parser.add_argument('--tanhalpha', type=float, default=float(config['model']['tanhalpha']), help='saturation ratio in graph construction')

parser.add_argument('--in_dim', type=int, default=int(config['model']['in_dim']), help='inputs dimension')
parser.add_argument('--conv_channels', type=int, default=int(config['model']['conv_channels']), help='convolution channels')
parser.add_argument('--end_channels', type=int, default=int(config['model']['end_channels']), help='end channels')
parser.add_argument('--num_hidden_layers', type=int, default=int(config['model']['num_hidden_layers']), help='num_hidden_layers')

parser.add_argument('--solver_1', type=str, default=config['model']['solver_1'], help='interlayer Solver')
parser.add_argument('--time_1', type=float, default=float(config['model']['time_1']), help='interlayer integration time')
parser.add_argument('--step_1', type=float, default=float(config['model']['step_1']), help='interlayer step size')
parser.add_argument('--solver_2', type=str, default=config['model']['solver_2'], help='inlayer Solver')
parser.add_argument('--time_2', type=float, default=float(config['model']['time_2']), help='inlayer integration time')
parser.add_argument('--step_2', type=float, default=float(config['model']['step_2']), help='inlayer step size')
parser.add_argument('--alpha', type=float, default=float(config['model']['alpha']), help='eigen normalization')
parser.add_argument('--rtol', type=float, default=float(config['model']['rtol']), help='rtol')
parser.add_argument('--atol', type=float, default=float(config['model']['atol']), help='atol')

args = parser.parse_args()