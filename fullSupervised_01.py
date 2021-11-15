from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from process import *
from utils import *
from models import *

import uuid
import sys

from torch.profiler import profile, record_function, ProfilerActivity

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=64, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')

parser.add_argument('--model', default='GCNII_BASE', help='Model to use.')
parser.add_argument('--mode', default='GCNII', help='Mode to use. GCNII | FDGATII')

parser.add_argument('--heads', type=int, default=1, help='Number of attention heads for multihead attention')
parser.add_argument('--iterations', type=int, default=10, help='Iterations')
parser.add_argument('--support', type=int, default=0, help='0:No support. 1:GCNII 2:GCNII*')
parser.add_argument('--verbosity', type=int, default=0, help='0:only result. 1:checkpoint, scores..., 2:epoch info')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

cudaid = "cuda:"+str(args.dev) if torch.cuda.is_available() else "cpu"
device = torch.device(cudaid)
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'
if args.verbosity > 0: print(cudaid,checkpt_file)

def train_step(model,optimizer,features,labels,adj,idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(features,adj)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()


def validate_step(model,features,labels,adj,idx_val):
    model.eval()
    with torch.no_grad():
        output = model(features,adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

def test_step(model,features,labels,adj,idx_test):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_test.item(),acc_test.item()
  
def train(datastr,splitstr):
    adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = full_load_data(datastr,splitstr)
    
    features = features.to(device)
    adj = adj.to(device)

    _model = getattr(sys.modules[__name__], args.model)
    #print(f'Model : {_model}')

    model = _model(nfeat=num_features,
                nlayers=args.layer,
                nhidden=args.hidden,
                nclass=num_labels,
                dropout=args.dropout,
                lamda = args.lamda, 
                alpha=args.alpha,
                support=args.support,
                mode=args.mode, 
                heads=args.heads,
              ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    bad_counter = 0
    best = 999999999

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True) 

    starter.record()
    for epoch in range(args.epochs):
        loss_tra,acc_tra = train_step(model,optimizer,features,labels,adj,idx_train)
        loss_val,acc_val = validate_step(model,features,labels,adj,idx_val)

        if args.verbosity > 1:
            if(epoch+1)%1 == 0:            
                print('Epoch:{:04d}'.format(epoch+1),
                    'train', 'loss:{:.3f}'.format(loss_tra), 'acc:{:.2f}'.format(acc_tra*100),
                    '| val', 'loss:{:.3f}'.format(loss_val), 'acc:{:.2f}'.format(acc_val*100),  
                    flush=True)
        if loss_val < best:
            best = loss_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break
    ender.record()
    torch.cuda.synchronize()
    train_time = starter.elapsed_time(ender)

    starter.record()
    acc = test_step(model,features,labels,adj,idx_test)[1]
    ender.record()
    torch.cuda.synchronize()
    inf_time = starter.elapsed_time(ender)

    starter.record()
    for _ in range(1000):
      _ = test_step(model,features,labels,adj,idx_test)[1]
    ender.record()
    torch.cuda.synchronize()
    infN_time = starter.elapsed_time(ender)/1000

    return acc*100, epoch+1, train_time, inf_time, infN_time

print('\nSD', args)
print('Timing code - Using warmup and sync. time in ms', flush=True)  ##https://deci.ai/resources/blog/measure-inference-time-deep-neural-networks/
t_total = time.time()
acc_list = []
epoch_list = []
train_time_list, inf_time_list, infN_time_list = [], [], []


#GPU-WARM-UP
for _ in range(1):
    _ = train(args.data,  'splits/'+args.data+'_split_0.6_0.2_0.npz')

for i in range(args.iterations):
    datastr = args.data
    splitstr = 'splits/'+args.data+'_split_0.6_0.2_'+str(i)+'.npz'
    acc, epoch, train_time, inf_time, infN_time = train(datastr,splitstr)
    acc_list.append(acc)
    epoch_list.append(epoch)
    train_time_list.append(train_time)
    inf_time_list.append(inf_time)
    infN_time_list.append(infN_time)
    if args.verbosity > 0: print(i,": {:7.4f}".format(acc_list[-1]))


print(f'SS {args.data:10}, {args.mode:10}, S{args.support}, D{args.hidden}, H{args.heads}, L{args.layer}, {time.time() - t_total:7.4f}s, {np.mean(acc_list):7.4f}, {np.std(acc_list):7.4f}, :, ', end='')
[print("{:7.4f}, ".format(acc_list[i]), end='') for i in range(args.iterations)]
print(f'ep|, {np.mean(epoch_list):7.4f}, {np.std(epoch_list):7.4f}, :, ', end='')
[print("{:4}, ".format(epoch_list[i]), end='') for i in range(args.iterations)]
print(f'Tr|, {np.mean(train_time_list):7.4f}, {np.std(train_time_list):7.4f}, :, ', end='')
print(f'in|, {np.mean(inf_time_list):7.4f}, {np.std(inf_time_list):7.4f}, :, ', end='')
print(f'iN|, {np.mean(infN_time_list):7.4f}, {np.std(infN_time_list):7.4f}, :, ', end='')
print(flush=True)

print('------------------------------------------------------------------------')







