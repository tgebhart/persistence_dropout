import os
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import dionysus as dion
import networkx as nx

import numpy as np
import pandas as pd

from persistence_dropout.functions.filtration import conv_filtration, linear_filtration, conv_layer_as_matrix, conv_filtration_static, linear_filtration_static, conv_filtration_fast, linear_filtration_fast
from persistence_dropout.functions.persistence_dropout import StaticPersistenceDropout, InducedPersistenceDropout

import cProfile, pstats, io


class CFF(nn.Module):
    def __init__(self, filters=5, kernel_size=5, fc1_size=50, dropout_type='static'):
        super(CFF, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.fc1_size = fc1_size
        self.stride = 1
        self.activation = 'relu'
        self.conv1 = nn.Conv2d(1, self.filters, kernel_size=self.kernel_size, bias=False, stride=1)
        self.fc1 = nn.Linear(((28-self.kernel_size+1)**2)*self.filters, self.fc1_size, bias=False)
        self.fc2 = nn.Linear(self.fc1_size, 10, bias=False)
        self.input_size = 28
        self.layer_types = ['convolution', 'fully-connected', 'fully-connected']
        self.dropout_type = dropout_type
        self.dropout_function = dropout_function(dropout_type)

    def forward(self, x, muls=None, hiddens=False, dropout=False):
        if dropout:
            h1_m = F.relu(self.conv1(x))
            h1 = h1_m.view(-1, (28-self.kernel_size+1)**2*self.filters)
            h1_d = self.dropout_function(h1, muls, self.train, 0)
            h2 = F.relu(self.fc1(h1_d))
            h2_d = self.dropout_function(h2, muls, self.train, 1)
            y = self.fc2(h2_d)
            if hiddens:
                return F.log_softmax(y, dim=1), [h1, h2, y]
            return F.log_softmax(y, dim=1)
        else:
            h1_m = F.relu(self.conv1(x))
            h1 = h1_m.view(-1, (28-self.kernel_size+1)**2*self.filters)
            h2 = F.relu(self.fc1(h1))
            y = self.fc2(h2)
            if hiddens:
                return F.log_softmax(y, dim=1), [h1, h2, y]
            return F.log_softmax(y, dim=1)


    def hidden_forward(self,x):
        h1_m = F.relu(self.conv1(x))
        h1 = h1_m.view(-1, (28-self.kernel_size+1)**2*self.filters)
        h2 = F.relu(self.fc1(h1))
        y = self.fc2(h2)
        return F.log_softmax(y, dim=1), [h1, h2, y]


    def save_string(self, dropout_type='', p=''):
        return "cff_relu_{}_{}.pt".format(dropout_type, p)

    def layerwise_ids(self, input_size=28*28):
        l1_size = (28-self.kernel_size+1)**2*self.filters
        l1_end = input_size+l1_size
        l2_end = l1_end+self.fc1_size
        l3_end = l2_end + 10
        return [range(input_size), range(input_size, l1_end), range(l1_end, l2_end), range(l2_end, l3_end)]

    def compute_static_filtration(self, x, hiddens, percentile=None):
        f = dion.Filtration()

        h1_id_start = x.cpu().detach().numpy().reshape(-1).shape[0]
        f, h1_births = conv_filtration_static(f, x[0], self.conv1.weight.data[:,0,:,:], 0, h1_id_start, percentile=percentile)

        h2_id_start = h1_id_start + hiddens[0].cpu().detach().numpy().shape[0]
        f, h2_births = linear_filtration_static(f, hiddens[0], self.fc1, h1_births, h1_id_start, h2_id_start, percentile=percentile, last=False)

        h3_id_start = h2_id_start + hiddens[1].cpu().detach().numpy().shape[0]
        f = linear_filtration_static(f, hiddens[1], self.fc2, h2_births, h2_id_start, h3_id_start, percentile=percentile, last=True)

        # print('filtration size', len(f))
        f.sort(reverse=True)
        return f

    def compute_induced_filtration_batch(self, x, hiddens, percentile=None):
        '''Generally too memory intensive to store entire batch of filtrations.
        Instead iterate over each example input, compute diagram, then save.
        '''
        filtrations = []
        for s in range(x.shape[0]):
            # check if this makes sense
            this_hiddens = [hiddens[0][s], hiddens[1][s], hiddens[2][s]]
            print('Filtration: {}'.format(s))
            print(hiddens[0].shape, hiddens[1].shape, hiddens[2].shape)
            f = self.compute_induced_filtration(x[s,0], this_hiddens, percentile=percentile)
            filtrations.append(f)
        return filtrations

    def compute_induced_filtration(self, x, hiddens, percentile=None, mat=None):

        if mat is None:
            mat = conv_layer_as_matrix(self.conv1.weight.data[:,0,:,:], x[0], self.stride)

        # pr = cProfile.Profile()
        # pr.enable()
        h1_id_start = x.cpu().detach().numpy().reshape(-1).shape[0]
        m1, h0_births, h1_births, percentile_1 = conv_filtration_fast(x[0], mat, 0, h1_id_start, percentile=percentile)
        enums = m1
        enums += [([i], h0_births[i]) for i in np.argwhere(h0_births > percentile_1)]

        h2_id_start = h1_id_start + hiddens[0].cpu().detach().numpy().shape[0]
        m2, h1_births_2, h2_births, percentile_2 = linear_filtration_fast(hiddens[0], self.fc1, h1_id_start, h2_id_start, percentile=percentile)
        enums += m2

        max1 = np.maximum.reduce([h1_births, h1_births_2])
        comp_percentile = percentile_1 if percentile_1 < percentile_2 else percentile_2
        enums += [([i+h1_id_start], max1[i]) for i in np.argwhere(max1 > comp_percentile)]

        h3_id_start = h2_id_start + hiddens[1].cpu().detach().numpy().shape[0]
        m3, h2_births_2, h3_births, percentile_3 = linear_filtration_fast(hiddens[1], self.fc2, h2_id_start, h3_id_start, percentile=percentile)
        enums += m3

        max2 = np.maximum.reduce([h2_births, h2_births_2])
        comp_percentile = percentile_2 if percentile_2 < percentile_3 else percentile_3
        enums += [([i+h2_id_start], max2[i]) for i in np.argwhere(max2 > comp_percentile)]

        enums += [([i+h3_id_start], h3_births[i]) for i in np.argwhere(h3_births > percentile_3)]

        f = dion.Filtration(enums)

        # print('filtration size', len(f))
        f.sort(reverse=True)
        # pr.disable()
        # s = io.StringIO()
        # sortby = 'cumulative'
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())

        return f


    def compute_dropouts(self, x, hiddens, thru=2, p=1, dropout_type='static', percentile=0, muls=None, invert=False, mat=None):
        if dropout_type == 'static':
            f = self.compute_static_filtration(x, hiddens, percentile=percentile)
        if dropout_type == 'induced':
            f = self.compute_induced_filtration(x, hiddens, percentile=percentile, mat=mat)
        m = dion.homology_persistence(f)
        dgms = dion.init_diagrams(m,f)
        subgraphs = {}

        # compute ones tensors of same hidden dimensions
        muls = []
        if invert:
            for h in hiddens:
                muls.append(torch.zeros(h.shape))
        else:
            for h in hiddens:
                muls.append(torch.ones(h.shape))

        fac = 1.0 if invert else 0.0

        for i,c in enumerate(m):
            if len(c) == 2:
                if f[c[0].index][0] in subgraphs:
                    subgraphs[f[c[0].index][0]].add_edge(f[c[0].index][0],f[c[1].index][0],weight=f[i].data)
                else:
                    eaten = False
                    for k, v in subgraphs.items():
                        if v.has_node(f[c[0].index][0]):
                            v.add_edge(f[c[0].index][0], f[c[1].index][0], weight=f[i].data)
                            eaten = True
                            break
                    if not eaten:
                        g = nx.Graph()
                        g.add_edge(f[c[0].index][0], f[c[1].index][0], weight=f[i].data)
                        subgraphs[f[c[0].index][0]] = g

        #  I don't think we need this composition. Disjoint subgraphs are fine.
        # subgraph = nx.compose_all([subgraphs[k] for k in list(subgraphs.keys())[:thru]])

        sub_keys = list(subgraphs.keys())[:thru]
        lifetimes = np.empty(thru)
        t = 0
        for pt in dgms[0]:
            if pt.death < float('inf'):
                lifetimes[t] = pt.birth - pt.death
                t += 1
            if t >= thru:
                break
        max_lifetime = max(lifetimes)
        min_lifetime = min(lifetimes)
        ids = self.layerwise_ids()
        layer_types = self.layer_types

        for i in range(len(sub_keys)):
            k = sub_keys[i]
            lifetime = lifetimes[i]
            lim = p*lifetime/max_lifetime
            for e in subgraphs[k].edges(data=True):
                r = np.random.uniform()
                if r < lim:
                    for l in range(len(ids)-1):
                        if e[0] in ids[l] and e[1] in ids[l+1]:
                            # drop out this edge connection
                            # if layer_types[l] == 'convolution':
                            #     num_filters = self.filters
                            #     inp_size = self.input_size
                            #     stride = self.stride
                            #     kernel_size = self.kernel_size
                            #     next_ids = list(ids[l+1])
                            #     # muls[l] = (28-self.kernel_size+1)**2*self.filters
                            #     filt_num = math.floor((e[1]-ids[l][-1])/((inp_size-kernel_size+1)**2))
                            #     print('filtnum', filt_num)
                            #     # filters are 0-indexed, so add 1 and 2 to get ids
                            #     start_id = (inp_size-kernel_size+stride)**2*(filt_num+1)
                            #     end_id = (inp_size-kernel_size+stride)**2*(filt_num+2)
                            #     print('s min end', start_id, end_id, e[0], e[1])
                            #     # find which element of the kernel produced this edge
                            #     index_no = (e[1]-start_id) % (kernel_size*kernel_size)
                            #     image_no = (e[1]-start_id)//(kernel_size*kernel_size)

                            muls[l][e[1]-ids[l+1][0]] = fac

        return muls


def dropout_function(t):
    if t == 'static':
        return StaticPersistenceDropout.apply
    if t == 'induced':
        return InducedPersistenceDropout.apply


def train(args, model, device, train_loader, optimizer, epoch, dropout=False, dropout_type='static', p=1.0, percentile=0.0):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        if dropout:
            if dropout_type == 'static':
                _, hiddens = model.hidden_forward(data)
                this_hiddens = [hiddens[i][0] for i in range(len(hiddens))]
                muls = model.compute_dropouts(data[0], this_hiddens, dropout_type=dropout_type, p=p, percentile=percentile)
                optimizer.zero_grad()
                output = model(data, muls=muls, dropout=True)
            if dropout_type == 'induced':
                _, hiddens = model.hidden_forward(data)
                this_muls = [torch.ones((data.shape[0], hiddens[i][0].size()[0])) for i in range(len(hiddens))]
                sum_hiddens = [torch.sum(hidden,dim=0) for hidden in hiddens]
                sum_x = torch.sum(data,dim=0)
                mat = conv_layer_as_matrix(model.conv1.weight.data[:,0,:,:], data[0][0], model.stride)
                cd = model.compute_dropouts(sum_x, sum_hiddens, dropout_type=dropout_type, p=p, percentile=percentile, mat=mat)
                # for s in range(data.shape[0]):
                #     this_hiddens = [hiddens[i][s] for i in range(len(hiddens))]
                #     cd = model.compute_dropouts(data[s], this_hiddens, dropout_type=dropout_type, p=p, percentile=percentile)
                #     for l in range(len(cd)):
                #         this_muls[l][s] = cd[l]
                for l in range(len(cd)):
                    this_muls[l] = cd[l]
                optimizer.zero_grad()
                output = model(data, muls=this_muls, dropout=True)
        else:
            optimizer.zero_grad()
            output = model(data, dropout=False)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('-m', '--model-directory', type=str, required=True,
                        help='location to store trained model')
    parser.add_argument('-d', '--diagram-directory', type=str, required=False,
                        help='location to store homology info')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--up-to', type=int, default=500, metavar='N',
                        help='How many testing exmaples for creating diagrams')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-da', '--dataset', type=str, required=True,
                        help='which dataset to train on (mnist or fashionmnist)')
    parser.add_argument('-do', '--dropout', type=str, required=False,
                        help='which dropout to use')
    parser.add_argument('-dp', '--drop_probability', type=float, default=1.0,
                        help='Persistence dropout probability')
    parser.add_argument('-p', '--percentile', type=float, default=0,
                        help='Filtration threshold percentile')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])), batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])), batch_size=args.test_batch_size, shuffle=False, **kwargs)


    if args.dataset == 'fashion':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data/fashion', train=True, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])), batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data/fashion', train=False, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])), batch_size=args.test_batch_size, shuffle=False, **kwargs)

    dropout_type = args.dropout
    if dropout_type is not None:
        model = CFF(dropout_type=dropout_type).to(device)
        dropout = True
    else:
        model = CFF().to(device)
        dropout = False
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    res_df = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, dropout=dropout, dropout_type=dropout_type, p=args.drop_probability, percentile=args.percentile)
        test(args, model, device, test_loader)


    save_path = os.path.join(args.model_directory, model.save_string(dropout_type=dropout_type, p=args.drop_probability))
    torch.save(model.state_dict(), save_path)

    if args.diagram_directory is not None and args.create_diagrams:
        create_diagrams(args, model)

if __name__ == '__main__':
    main()
