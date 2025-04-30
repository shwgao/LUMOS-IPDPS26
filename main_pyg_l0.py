import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from applications.test_graph.gnn_l0 import GNN
import os

from tqdm.auto import tqdm
import argparse
import time
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

# for dataset mol
cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

# for dataset ppa
multicls_criterion = torch.nn.CrossEntropyLoss()

reg = True

def train(model, device, loader, optimizer, evaluator, writer, epoch, task_type=False):
    model.train()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batched_data = batch.to(device)
        x, edge_index, edge_attr, batchs = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        batched_data = (x, edge_index, edge_attr, batchs)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batched_data)
            optimizer.zero_grad()
            
            if not task_type:
                loss = multicls_criterion(pred.to(torch.float32), batch.y.view(-1,))
            else:
                is_labeled = batch.y == batch.y
                if "classification" in task_type: 
                    loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                else:
                    loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            
            if reg:
                l0_loss = model.regularization()
            else:
                l0_loss = torch.tensor(0.0)
            total_loss = loss + l0_loss
            writer.add_scalar('loss/train', loss.item(), step+(epoch-1)*len(loader))
            writer.add_scalar('loss/train_l0', l0_loss, step+(epoch-1)*len(loader))
            
            total_loss.backward()
            optimizer.step()
            
            if not task_type:
                y_true.append(batch.y.view(-1,1).detach().cpu())
                y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())
            else:
                y_true.append(batch.y.view(pred.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    
    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

def eval(model, device, loader, evaluator, dataset=None):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batched_data = batch.to(device)
        x, edge_index, edge_attr, batchs = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        batched_data = (x, edge_index, edge_attr, batchs)

        if x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batched_data)

            if dataset=='ogbg-ppa':
                y_true.append(batch.y.view(-1,1).detach().cpu())
                y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())
            else:
                y_true.append(batch.y.view(pred.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-ppa data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gcn',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-ppa",
                        help='dataset name (default: ogbg-ppa, ogbg-molhiv)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default=".",
                        help='filename to output result (default: )')
    params = parser.parse_args()
    
    with open(f'./applications/test_graph/settings.json', 'r') as f:
        settings = json.load(f)
        for key, value in settings.items():
            # if argument already exists, replace it
            if key in params:
                setattr(params, key, value)
            else:
                parser.add_argument(f'--{key}', type=type(value), default=value)
    
    args = parser.parse_args()
    args.application = args.gnn + '_' + args.dataset + '_l0' + str(args.use_reg)
    
    reg = args.use_reg

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    if args.dataset == 'ogbg-ppa':
        dataset = PygGraphPropPredDataset(name = args.dataset, transform = add_zeros)
    else:
        dataset = PygGraphPropPredDataset(name = args.dataset)
        if args.feature == 'full':
            pass 
        elif args.feature == 'simple':
            print('using simple feature')
            # only retain the top two node/edge features
            dataset.data.x = dataset.data.x[:,:2]
            dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    
    num_class = dataset.num_classes if args.dataset=='ogbg-ppa' else dataset.num_tasks

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_class = num_class, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False, FLAG = args.dataset).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_class = num_class, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True, FLAG = args.dataset).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_class = num_class, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False, FLAG = args.dataset).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_class = num_class, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True, FLAG = args.dataset).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    time_str = time.strftime("%d%H%M")
    
    writer = SummaryWriter(f'./runs/{args.dataset}-{args.gnn}/l0{reg}-{time_str}')
    writer.add_text('hyperparameters', json.dumps(vars(args), indent=4, sort_keys=True))

    valid_curve = []
    test_curve = []
    train_curve = []
    
    best_score = 0.
    os.makedirs(f'./checkpoints/{args.dataset}-{args.gnn}/l0{reg}-{time_str}', exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        if args.dataset == 'ogbg-ppa':
            train_perf = train(model, device, train_loader, optimizer, evaluator, writer, epoch)
        else:
            train_perf = train(model, device, train_loader, optimizer, evaluator, writer, epoch, task_type = dataset.task_type)
        print('Evaluating...')
        # train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator, dataset = args.dataset)
        # test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': 0})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        # test_curve.append(test_perf[dataset.eval_metric])
        writer.add_scalar('quality/train', train_perf[dataset.eval_metric], epoch)
        writer.add_scalar('quality/val', valid_perf[dataset.eval_metric], epoch)
        # writer.add_scalar('quality/test', test_perf[dataset.eval_metric], epoch)
        
        if reg:
            alive, total = 0, 0
            for k, layer in enumerate(model.layers):
                if hasattr(layer, 'qz_loga') and layer.qz_loga is not None:
                    mode_z = layer.sample_z(1, sample=0).view(-1)
                    alive += torch.sum(mode_z != 0).item()
                    total += mode_z.nelement()

                    writer.add_scalar('logitsq0/{}'.format(k), layer.qz_loga[0].item(), epoch)
                    writer.add_scalar('logitsq3/{}'.format(k), layer.qz_loga[3].item(), epoch)
                    writer.add_scalar('logitslast/{}'.format(k), layer.qz_loga[-1].item(), epoch)

                    writer.add_histogram('mode_z/layer{}'.format(k), mode_z.cpu().data.numpy(), epoch)
                    writer.add_scalar('mode_n/layer{}'.format(k), torch.sum(mode_z != 0).item(), epoch)
                    writer.add_scalar('prune_ratio/layer{}'.format(k), torch.sum(mode_z == 0).item() / mode_z.nelement(), epoch)
            prune_ratio = 1 - alive / total
            writer.add_scalar('prune_ratio/global', prune_ratio, epoch)
        
        # save the best model checkpoints
        if valid_perf[dataset.eval_metric] > best_score:
            best_score = valid_perf[dataset.eval_metric]
            torch.save(model.state_dict(), f'./checkpoints/{args.dataset}-{args.gnn}/l0{reg}-{time_str}/best_model.pth')
    
    # save the final model checkpoints
    torch.save(model.state_dict(), f'./checkpoints/{args.dataset}-{args.gnn}/l0{reg}-{time_str}/final_model.pth')
        
    if args.dataset == 'ogbg-ppa' or 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)
    
    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    # print('Test score: {}'.format(test_curve[best_val_epoch]))

    # if not args.filename == '':
    #     torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)


if __name__ == "__main__":
    main()