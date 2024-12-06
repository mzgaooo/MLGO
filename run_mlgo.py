import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm

from args import args
from model import MLGO
from util import generate_dataset, read_data
from eval import masked_mae_torch, masked_mape_torch, masked_rmse_torch


def train(loader, model, optimizer, criterion, step_size, num_nodes, num_split, device):
    batch_loss = 0
    for idx, (inputs, targets) in enumerate(tqdm(loader)):
        model.train()
        optimizer.zero_grad()

        inputs = inputs.to(device)
        targets = targets.to(device)

        if idx % step_size == 0:
            perm = np.random.permutation(range(num_nodes))
        num_sub = int(num_nodes / num_split)
        for j in range(num_split):
            if j != num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]

        id = torch.tensor(id).to(device)
        inputs = inputs[:, :, id, :]   
        targets = targets[:, :, id, :]  
        outputs = model(inputs, id)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_loss += loss.detach().cpu().item() 
    return batch_loss / (idx + 1)


@torch.no_grad()
def eval(loader, model, std, mean, device):
    batch_rmse_loss = 0  
    batch_mae_loss = 0
    batch_mape_loss = 0
    for idx, (inputs, targets) in enumerate(tqdm(loader)):
        model.eval()

        inputs = inputs.to(device)
        targets = targets.to(device)
        output = model(inputs)
        
        out_unnorm = output.detach().cpu()*std + mean
        target_unnorm = targets.detach().cpu()*std + mean

        mae_loss = masked_mae_torch(out_unnorm, target_unnorm, 0)
        rmse_loss = masked_rmse_torch(out_unnorm, target_unnorm, 0)
        mape_loss = masked_mape_torch(out_unnorm, target_unnorm, 0)

        batch_rmse_loss += rmse_loss
        batch_mae_loss += mae_loss
        batch_mape_loss += mape_loss

    return batch_rmse_loss / (idx + 1), batch_mae_loss / (idx + 1), batch_mape_loss / (idx + 1)

@torch.no_grad()
def test(loader, model, std, mean, device):
    batch_rmse_loss = 0  
    batch_mae_loss = 0
    batch_mape_loss = 0

    batch_rmse = [0]*12
    batch_mae = [0]*12
    batch_mape = [0]*12
    for idx, (inputs, targets) in enumerate(tqdm(loader)):
        model.eval()

        inputs = inputs.to(device)
        targets = targets.to(device)
        output = model(inputs)
        
        out_unnorm = output.detach().cpu()*std + mean
        target_unnorm = targets.detach().cpu()*std + mean

        mae_loss = masked_mae_torch(out_unnorm, target_unnorm, 0)  #Multi-step
        rmse_loss = masked_rmse_torch(out_unnorm, target_unnorm, 0)
        mape_loss = masked_mape_torch(out_unnorm, target_unnorm, 0)
        batch_rmse_loss += rmse_loss
        batch_mae_loss += mae_loss
        batch_mape_loss += mape_loss

        for t in range(target_unnorm.shape[3]):  #Single-step
            mae = masked_mae_torch(out_unnorm[:,:,:,t], target_unnorm[:,:,:,t], 0)
            rmse = masked_rmse_torch(out_unnorm[:,:,:,t], target_unnorm[:,:,:,t], 0)
            mape = masked_mape_torch(out_unnorm[:,:,:,t], target_unnorm[:,:,:,t], 0)
            batch_rmse[t] += rmse
            batch_mae[t] += mae
            batch_mape[t] += mape

    return batch_rmse_loss / (idx + 1), batch_mae_loss / (idx + 1), batch_mape_loss / (idx + 1), [x / (idx + 1) for x in batch_rmse], [x / (idx + 1) for x in batch_mae], [x / (idx + 1) for x in batch_mape]


torch.set_num_threads(4)

def main(args):
    # random seed
    
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    
    device = torch.device(args.device)

    data, mean, std, dtw_matrix, sp_matrix = read_data(args)
    train_loader, valid_loader, test_loader = generate_dataset(data, args)

    model = MLGO(num_nodes=args.num_nodes, device=device, predefined_A=[dtw_matrix, sp_matrix], 
                 num_hidden_layers=args.num_hidden_layers, subgraph_size=args.subgraph_size,
                 node_dim=args.node_dim, conv_channels=args.conv_channels, end_channels=args.end_channels, 
                 seq_length=args.seq_len, in_dim=args.in_dim, tanhalpha=args.tanhalpha,
                 method_1=args.solver_1, time_1=args.time_1, step_size_1=args.step_1, 
                 method_2=args.solver_2, time_2=args.time_2, step_size_2=args.step_2,
                 alpha=args.alpha, rtol=args.rtol, atol=args.atol)
    model = model.to(device)

    lr=args.lr
    weight_decay=args.weight_decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if args.lr_decay:
        lr_decay_steps = args.lr_decay_steps.split(',')
        lr_decay_steps = [int(i) for i in lr_decay_steps]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_steps, gamma=args.lr_decay_rate)
    
    criterion = nn.L1Loss()

    print(args)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)
    
    best_valid_rmse = 1000 
    
    for epoch in range(1, args.epochs+1):
        print("=====Epoch {}=====".format(epoch))
        print('Training...')
        loss = train(train_loader, model, optimizer, criterion, args.step_size, args.num_nodes, args.num_split, device)
        print('Evaluating...')
        train_rmse, train_mae, train_mape = eval(train_loader, model, std, mean, device)
        valid_rmse, valid_mae, valid_mape = eval(valid_loader, model, std, mean, device)

        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            print('New best results!')
            torch.save(model.state_dict(), args.save + args.filename + ".pkl")

        print(f'\n##on train data## loss: {loss}\n' + 
            f'##on train data## rmse loss: {train_rmse}, mae loss: {train_mae}, mape loss: {train_mape}\n' +
            f'##on valid data## rmse loss: {valid_rmse}, mae loss: {valid_mae}, mape loss: {valid_mape}\n')
        
        if args.lr_decay:
            scheduler.step() 
    
    model.load_state_dict(torch.load(args.save + args.filename + ".pkl"))
    test_rmse_loss, test_mae_loss, test_mape_loss, test_rmse, test_mae, test_mape = test(test_loader, model, std, mean, device)
    print(f'##on test data## rmse loss: {test_rmse_loss}, mae loss: {test_mae_loss}, mape loss: {test_mape_loss}')
    for t in range(args.seq_len):
        print(f'on test data horizon: {t+1}, rmse: {test_rmse[t]}, mae: {test_mae[t]}, mape: {test_mape[t]}')
    
    if args.save_preds:
        torch.save(model.state_dict(), args.save_preds_path + args.filename + "_seq_len" + str(args.seq_len) + ".pkl")


if __name__ == '__main__':
    main(args)

