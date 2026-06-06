import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from datetime import timedelta

from args import args
from model import MPGODE
from util import generate_dataset, read_data
from eval import masked_mae_torch, masked_mape_torch, masked_rmse_torch


def train(loader, model, optimizer, criterion, step_size, num_nodes, num_split, device):
    batch_loss = 0
    total_batches = len(loader)
    print_interval = max(total_batches // 5, 1)  # Print 5 times per epoch
    
    for idx, (inputs, targets) in enumerate(loader):
        model.train()
        optimizer.zero_grad()

        inputs = inputs.to(device)
        targets = targets.to(device)

        if idx % step_size == 0:
            id = np.random.permutation(range(num_nodes))
            id = torch.from_numpy(id).to(device)

        inputs = inputs[:, :, id, :]   
        targets = targets[:, :, id, :]  
        outputs = model(inputs, id)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_loss += loss.detach().cpu().item() 
        
        if (idx + 1) % print_interval == 0 or (idx + 1) == total_batches:
            avg_loss = batch_loss / (idx + 1)
            print(f'  Batch {idx+1}/{total_batches}, Avg Loss: {avg_loss:.4f}')
            
    return batch_loss / (idx + 1)


@torch.no_grad()
def eval(loader, model, std, mean, device):
    batch_rmse_loss = 0  
    batch_mae_loss = 0
    batch_mape_loss = 0
    for idx, (inputs, targets) in enumerate(loader):
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

    #pred = []
    #true = []
    for idx, (inputs, targets) in enumerate(loader):
        model.eval()

        inputs = inputs.to(device)
        targets = targets.to(device)
        output = model(inputs)
        
        out_unnorm = output.detach().cpu()*std + mean
        target_unnorm = targets.detach().cpu()*std + mean
        #pred.append(out_unnorm)
        #true.append(target_unnorm)
        

        mae_loss = masked_mae_torch(out_unnorm, target_unnorm, 0)  
        rmse_loss = masked_rmse_torch(out_unnorm, target_unnorm, 0)
        mape_loss = masked_mape_torch(out_unnorm, target_unnorm, 0)
        batch_rmse_loss += rmse_loss
        batch_mae_loss += mae_loss
        batch_mape_loss += mape_loss

        for t in range(target_unnorm.shape[3]):  #每个时间步
            mae = masked_mae_torch(out_unnorm[:,:,:,t], target_unnorm[:,:,:,t], 0)
            rmse = masked_rmse_torch(out_unnorm[:,:,:,t], target_unnorm[:,:,:,t], 0)
            mape = masked_mape_torch(out_unnorm[:,:,:,t], target_unnorm[:,:,:,t], 0)
            batch_rmse[t] += rmse
            batch_mae[t] += mae
            batch_mape[t] += mape

    #torch.save(pred,'pred.pkl')
    #torch.save(true,'true.pkl')

    return batch_rmse_loss / (idx + 1), batch_mae_loss / (idx + 1), batch_mape_loss / (idx + 1), [x / (idx + 1) for x in batch_rmse], [x / (idx + 1) for x in batch_mae], [x / (idx + 1) for x in batch_mape]


torch.set_num_threads(4)

def main(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device(args.device)

    data, mean, std, dtw_matrix, sp_matrix = read_data(args)
    train_loader, valid_loader, test_loader = generate_dataset(data, args)

    model = MPGODE(num_nodes=args.num_nodes, device=device, predefined_A=[dtw_matrix, sp_matrix], 
                   dropout=args.dropout, num_hidden_layers=args.num_hidden_layers, subgraph_size=args.subgraph_size,
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
    '''
    best_valid_rmse = 1000 
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        print("====================Epoch {}====================".format(epoch))
        print('Training...')
        loss = train(train_loader, model, optimizer, criterion, args.step_size, args.num_nodes, args.num_split, device)
        epoch_time = time.time() - epoch_start_time

        print('Evaluating...')
        train_rmse, train_mae, train_mape = eval(train_loader, model, std, mean, device)
        valid_rmse, valid_mae, valid_mape = eval(valid_loader, model, std, mean, device)

        print(f'##on train data## loss: {loss}\n' + 
            f'##on train data## rmse loss: {train_rmse}, mae loss: {train_mae}, mape loss: {train_mape}\n' +
            f'##on valid data## rmse loss: {valid_rmse}, mae loss: {valid_mae}, mape loss: {valid_mape}\n' +
            f'Training Time : {str(timedelta(seconds=int(epoch_time)))}')

        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            print('New best results!!!\n')
            best_model = f"{args.filename}_epoch{epoch}_rmse{valid_rmse:.3f}_mae{valid_mae:.3f}_mape{valid_mape:.3f}.pkl"
            torch.save(model.state_dict(), args.save + best_model)
            #torch.save(model.ODE.odefunc.graph3,'adj.pt')

            if epoch > 70:
                test_rmse_, test_mae_, test_mape_, _, _, _ = test(test_loader, model, std, mean, device)
                print(f'##on test data## rmse loss: {test_rmse_}, mae loss: {test_mae_}, mape loss: {test_mape_}\n')

        if args.lr_decay:
            scheduler.step() 
    '''
    print('Testing...')
    model.load_state_dict(torch.load('pemsbay.pkl'))
    start_inference_time = time.time()
    test_rmse, test_mae, test_mape, test_rmse_loss, test_mae_loss, test_mape_loss = test(test_loader, model, std, mean, device)
    inference_time = time.time() - start_inference_time
    print(f'##on test data## rmse loss: {test_rmse}, mae loss: {test_mae}, mape loss: {test_mape}')
    print(f'Inference Time : {str(timedelta(seconds=int(inference_time)))}')
    for t in range(args.seq_len):
        print(f'on test data horizon: {t+1}, rmse: {test_rmse_loss[t]}, mae: {test_mae_loss[t]}, mape: {test_mape_loss[t]}')

if __name__ == '__main__':
    main(args)