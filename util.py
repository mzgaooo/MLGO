import os
import csv
import numpy as np
from fastdtw import fastdtw
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch


files = {
    'pems03': ['PEMS03/PEMS03.npz', 'PEMS03/PEMS03.csv'],
    'pems04': ['PEMS04/PEMS04.npz', 'PEMS04/PEMS04.csv'],
    'pems07': ['PEMS07/PEMS07.npz', 'PEMS07/PEMS07.csv'],
    'pems08': ['PEMS08/PEMS08.npz', 'PEMS08/PEMS08.csv'],
    'pems-bay': ['PEMS-bay/pems-bay.npz', 'PEMS-bay/pems-bay.csv'],
}

def read_data(args):
    filename = args.filename
    file = files[filename]
    filepath = "./data/"
    data = np.load(filepath + file[0])['data']
    num_node = data.shape[1]
    mean_value = np.mean(data, axis=(0, 1)).reshape(1, 1, -1)
    std_value = np.std(data, axis=(0, 1)).reshape(1, 1, -1)
    data = (data - mean_value) / std_value
    mean_value = mean_value.reshape(-1)[0]
    std_value = std_value.reshape(-1)[0]

    split_size = int(args.time_1 / args.step_1)
    print('split_size is',split_size)
    dtw = []
    if not os.path.exists(f'data/{filename}_dtw_distance.npy'):
        split_data = data[:int(data.shape[0]*0.6)]
        split_data = np.array_split(split_data, split_size)
        for k in range(split_size):
            data_mean = np.mean([split_data[k][:, :, 0][24*12*i: 24*12*(i+1)] for i in range(split_data[k].shape[0]//(24*12))], axis=0)
            data_mean = data_mean.squeeze().T 
            dtw_distance = np.zeros((num_node, num_node))
            print(k)
            for i in tqdm(range(num_node)):
                for j in range(i, num_node):
                    dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
            for i in range(num_node):
                for j in range(i):
                    dtw_distance[i][j] = dtw_distance[j][i]
            dtw.append(dtw_distance)
        np.save(f'data/{filename}_dtw_distance.npy', dtw)


    dist_matrix = np.load(f'data/{filename}_dtw_distance.npy')

    mean = np.mean(dist_matrix, axis=(1, 2))
    std = np.std(dist_matrix, axis=(1, 2))
    dist_matrix = (dist_matrix - mean[:, np.newaxis, np.newaxis]) / std[:, np.newaxis, np.newaxis]
    sigma = args.sigma1
    dist_matrix = np.exp(-dist_matrix ** 2 / sigma ** 2)
    dtw_matrix = np.zeros_like(dist_matrix)
    dtw_matrix[dist_matrix > args.thres1] = 1

    
    if not os.path.exists(f'data/{filename}_spatial_distance.npy'):
        with open(filepath + file[1], 'r') as fp:
            dist_matrix = np.zeros((num_node, num_node)) + np.float('inf')
            file = csv.reader(fp)
            for line in file:
                break
            for line in file:
                start = int(line[0])
                end = int(line[1])
                dist_matrix[start][end] = float(line[2])
                dist_matrix[end][start] = float(line[2])
            np.save(f'data/{filename}_spatial_distance.npy', dist_matrix)

    dist_matrix = np.load(f'data/{filename}_spatial_distance.npy')
   
    std = np.std(dist_matrix[dist_matrix != np.float('inf')])
    mean = np.mean(dist_matrix[dist_matrix != np.float('inf')])
    dist_matrix = (dist_matrix - mean) / std
    sigma = args.sigma2
    sp_matrix = np.exp(- dist_matrix**2 / sigma**2)
    sp_matrix[sp_matrix < args.thres2] = 0 

    return torch.from_numpy(data[:,:,0].astype(np.float32)), mean_value, std_value, dtw_matrix, sp_matrix


class MyDataset(Dataset):
    def __init__(self, data, split_start, split_end, his_length, pred_length):
        split_start = int(split_start)
        split_end = int(split_end)
        self.data = data[split_start: split_end]
        self.his_length = his_length     #history
        self.pred_length = pred_length   #predict
    
    def __getitem__(self, index):
        x = self.data[index: index + self.his_length]
        x = torch.unsqueeze(x, dim=0).transpose(1, 2)
        y = self.data[index + self.his_length: index + self.his_length + self.pred_length]
        y = torch.unsqueeze(y, dim=0).transpose(1, 2)
        return torch.Tensor(x), torch.Tensor(y)
    def __len__(self):
        return self.data.shape[0] - self.his_length - self.pred_length + 1


def generate_dataset(data, args):
    #shape：batch * features * N * his_length
    batch_size = args.batch_size
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    his_length = args.seq_len
    pred_length = args.seq_len
    train_dataset = MyDataset(data, 0, data.shape[0] * train_ratio, his_length, pred_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = MyDataset(data, data.shape[0]*train_ratio, data.shape[0]*(train_ratio+valid_ratio), his_length, pred_length)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MyDataset(data, data.shape[0]*(train_ratio+valid_ratio), data.shape[0], his_length, pred_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader

