import torch
import argparse
from torch_pruning.utils.benchmark import measure_memory, measure_latency
from utils import measure_parameters, measure_flops
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from main_pyg_l0 import add_zeros
from torch_geometric.loader import DataLoader
from applications.test_graph.gnn_l0 import GNN
from main_pyg_l0 import eval


def performance_test(model, device, example_inputs, repeat=1):
    model.eval()
    model.to(device)

    with torch.no_grad():
        p_conv, p_lin = measure_parameters(model)
        print('Calculating FLOPs...')

        f_conv, f_lin = measure_flops(model)
        print(f"Conv Params: {p_conv}   Linear Params: {p_lin} ")
        print(f"Conv Flops: {f_conv}   Linear Flops: {f_lin} ")

        torch.cuda.empty_cache()

        model.to(device)

        memory = measure_memory(model, example_inputs, device=device) if device != 'cpu' else 0
        base_latency, std_time = measure_latency(
            model, example_inputs, 100, 10)
        print("Base Latency: {:.4f}+-({:.4f}) ms, Peak Memory: {:.4f}M\n"
              .format(base_latency, std_time, memory / (1024*1024)))
        

def inference(args):
    val_loader = None
    using_reg = args.model != 'original'
    
    ### prepare dataset
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
    
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    
    evaluator = Evaluator(args.dataset)
    
    # prepare model
    num_class = dataset.num_classes if args.dataset=='ogbg-ppa' else dataset.num_tasks

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_class = num_class, virtual_node = False)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_class = num_class, virtual_node = True)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_class = num_class, virtual_node = False)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_class = num_class, virtual_node = True)
    else:
        raise ValueError('Invalid GNN type')


    print('Preparation done.')

    model_s = 'original' if args.model == 'original' else 'pruned'
    model_pth = f"./checkpoints/test_graph/{args.gnn}_{args.dataset}_l0{using_reg}/final_model.pth"
    print(f"\nTesting {model_s} model\'s {args.task} on device {args.device}: \n")

    state = torch.load(model_pth, map_location='cpu')
    model.load_state_dict(state, strict=False)
    
    print('before pruning, model has parameters:', count_parameters(model))

    if model_s == 'pruned':
        print('Testing pruned model, pruning...')
        model.prune_model()
        print('Pruning done')
    else:
        print('Testing original model...')
    
    print('after pruning, model has parameters:', count_parameters(model))
    
    device = args.device
    model = model.to(device)
    
    # valid_perf = eval(model, device, valid_loader, evaluator, dataset = args.dataset)
    # print(valid_perf)
    
    for _, batched_data in enumerate(train_loader):
        batched_data = batched_data.to(device)
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        batched_data = (x, edge_index, edge_attr, batch)

        performance_test(model, device, batched_data, repeat=1)
        
        break
    
    # save as onnx model
    torch.onnx.export(model, (batched_data, ),
                    f"./whole_model/test_graph/{args.gnn}_{args.dataset}_l0{using_reg}.onnx", verbose=False)


def count_parameters(model):
    # return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # count and print the number of every layer's parameters
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.numel())
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument("--application", type=str, default="cifar10",
                        help="CFD or fluidanimation or puremd or cosmoflow or EMDenoise or minist "
                        "or DMS or optical or stemdl, slstr or synthetic or cifar10")
    parser.add_argument("--state_dir", type=str, default="../checkpoints-v0/")
    parser.add_argument("--model", type=str, default="original", help="original or pruned")
    parser.add_argument("--task", type=str, default="performance", help="performance or quality")
    
    # Training settings
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gcn',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=45, # ppa: 50, molhiv: 2400
                        help='batch size')
    parser.add_argument('--repeat', type=int, default=10, help='repeat the input data')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-ppa",
                        help='dataset name (default: ogbg-ppa, ogbg-molhiv)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default=".",
                        help='filename to output result (default: )')
    params = parser.parse_args()
    
    args = parser.parse_args()
    # args.application = 'fluidanimation'
    # args.task = 'quality'
    args.model = 'pruned'
    inference(args)
