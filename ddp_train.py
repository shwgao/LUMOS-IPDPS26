import os
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import json
import argparse
from tqdm import tqdm
import utils
import time
from utils import AverageMeter

torch.set_float32_matmul_precision('high')


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_demo(demo_fn, world_size, writer_info=None):
    torch.multiprocessing.spawn(demo_fn,
                                args=(args, world_size, writer_info),
                                nprocs=world_size,
                                join=True)


def train_step(train_loader, model, criterion, optimizer, epoch, measure_quality, print_detail=False, reg=False, rank=0):
    """Train for one epoch on the training set"""

    losses = AverageMeter()
    l0_losses = AverageMeter()
    quality = AverageMeter()

    # switch to train mode
    model.train()

    for i, (data) in enumerate(train_loader):
        input_, target = data
        target = target.to(model.device)
        input_ = input_.to(model.device)

        # compute output
        output = model(input_)

        loss, l0_loss = criterion(output, target, model.module, reg=reg)
        total_loss = loss + l0_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # clamp the parameters
        model.module.constrain_parameters()

        # measure accuracy and record loss
        measure_quality(output, target, quality, input_.size(0))

        losses.update(loss.item(), 1)
        l0_losses.update(l0_loss.item(), 1)

        if model.module.beta_ema > 0.:
            model.module.update_ema()

        if print_detail and i % 100 == 0 and rank == 0:
            print(' Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                  'L0Loss {l0_loss.val:.6f} ({l0_loss.avg:.6f})\t'
                  'Acc {acc.val:.6f} ({acc.avg:.6f})'.format(epoch, i, len(train_loader), loss=losses,
                                                             l0_loss=l0_losses, acc=quality))
    return quality.avg, losses.avg, l0_losses.avg


def validate(val_loader, model, criterion, measure_quality, print_detail=False, inference=False):
    """Perform validation on the validation set"""
    losses = AverageMeter()
    quality = AverageMeter()
    old_params = None

    # switch to evaluate mode
    model.eval()
    if model.module.beta_ema > 0 and not inference:
        old_params = model.module.get_params()
        model.module.load_ema_params()

    with torch.no_grad():
        for i, (data) in enumerate(val_loader):
            input_, target = data
            target = target.to(model.device)
            input_ = input_.to(model.device)

            # compute output
            output = model(input_)
            loss, _ = criterion(output, target, model, reg=False)

            # measure accuracy and record loss
            measure_quality(output, target, quality, input_.size(0))

            losses.update(loss.item())

    if print_detail:
        print('Test: [{0}]\t'
              'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
              'Acc {acc.val:.6f} ({acc.avg:.6f})'.format(len(val_loader), loss=losses, acc=quality))

    if model.module.beta_ema > 0 and not inference:
        model.module.load_params(old_params)

    return quality.avg, losses.avg


def train(rank, arg, world_size, writer_info=None):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)
    
    # Check if CUDA is available and set device accordingly
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    init_model, dataset, launch, input_shape, val_quality, val_loss, scheduler = None, None, None, None, None, None, None

    if arg.application == "minist":
        from applications.minist import model, dataset, launch, input_shape
        init_model = model.MLP()
    elif arg.application == "cifar10":
        from applications.cifar10 import model, dataset, launch, input_shape
        init_model = model.L0LeNet5()
    elif arg.application == "puremd":
        from applications.puremd import model, dataset, launch, input_shape
        init_model = model.MLP()
    elif arg.application == "CFD":
        from applications.CFD import model, dataset, launch, input_shape
        init_model = model.MLP()
    elif arg.application == "fluidanimation":
        from applications.fluidanimation import model, dataset, launch, input_shape
        init_model = model.MLP()
    elif arg.application == "cosmoflow":
        from applications.cosmoflow import model, dataset, launch, input_shape
        init_model = model.CosmoFlow()
    elif arg.application == "EMDenoise":
        from applications.EMDenoise import model, dataset, launch, input_shape
        init_model = model.EMDenoiseNet()
    elif arg.application == "DMS":
        from applications.DMS import model, dataset, launch, input_shape
        init_model = model.DMSNet()
    elif arg.application == "optical":
        from applications.optical import model, dataset, launch, input_shape
        init_model = model.Autoencoder()
    elif arg.application == "stemdl":
        from applications.stemdl import model, dataset, launch, input_shape
        init_model = model.VGG11()
    elif arg.application == "slstr":
        from applications.slstr import model, dataset, launch, input_shape
        init_model = model.UNet()
    elif arg.application == "synthetic":
        from applications.synthetic import model, dataset, launch, input_shape
        init_model = model.MLP()
    else:
        print("Application not found")

    # print details of the task
    print(f"Application: {arg.application}......")
    print(f"Input shape: {input_shape}......")

    train_set, test_set = dataset.get_loader(batch_size=arg.batch_size, data_only=True, rank=rank)
    
    # Move model to appropriate device
    init_model = init_model.to(device)
    init_model.device = device
    init_model = DDP(init_model, 
                     device_ids=[rank] if torch.cuda.is_available() else None,
                     output_device=rank if torch.cuda.is_available() else None,
                     find_unused_parameters=True,
                     static_graph=True)
    
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_set, batch_size=arg.batch_size, sampler=sampler)
    if rank == 0:
        test_loader = DataLoader(test_set, batch_size=arg.batch_size)
    
    # define the optimizer and scheduler
    optimizer = optim.SGD(init_model.parameters(), lr=arg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=arg.step_size, gamma=arg.gamma) if arg.scheduler else None

    # creat tqdm progress bar
    if rank == 0:
        pbar = tqdm(range(arg.epochs))
    else:
        pbar = range(arg.epochs)

    # save the model by the best quality
    best_quality = 0 if arg.higher_better else 100

    # Create writer only for rank 0
    writer = None
    if rank == 0 and writer_info and writer_info['use_tensorboard']:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_info['path'])
        writer.add_text('hyperparameters', json.dumps(vars(arg), indent=4, sort_keys=True))

    # train the model
    for ite in pbar:
        sampler.set_epoch(ite)
        # train step
        train_quality, loss, l0_loss = train_step(train_loader, init_model, launch.loss_fn, optimizer, ite,
                                                  measure_quality=launch.measure_quality, print_detail=True,
                                                  reg=arg.use_reg, rank=rank)
        # validate the model every arg.val_freq
        if ite % arg.val_freq == 0 and rank == 0:
            val_quality, val_loss = validate(test_loader, init_model, launch.loss_fn, print_detail=True,
                                             inference=False, measure_quality=launch.measure_quality)

        # log to TensorBoard
        if writer and rank == 0:
            writer.add_scalar('loss/val', val_loss, ite)
            writer.add_scalar('quality/val', val_quality, ite)
            writer.add_scalar('loss/train', loss, ite)
            writer.add_scalar('loss/train_l0', l0_loss, ite)
            writer.add_scalar('quality/train', train_quality, ite)

        # update the scheduler
        if scheduler:
            scheduler.step()

        # update budget
        if arg.task == 'prune':
            utils.update_budget(arg, ite, init_model)

        # update the progress bar and save the model by the best quality
        if rank == 0:
            pbar.set_description(f'Training: Quality {train_quality:.6f}, Loss {loss:.6f}, L0 Loss {l0_loss:.6f}, '
                                f'Validation: Quality {val_quality:.6f}, Loss {val_loss:.6f}')

            
            if arg.higher_better:
                if val_quality > best_quality:
                    best_quality = val_quality
                    torch.save(init_model.state_dict(), f'./checkpoints/{arg.application}-ddp-{arg.world_size}/{arg.name}-{time.strftime("%m%d%H%M")}.pth')
            else:
                if val_quality < best_quality:
                    best_quality = val_quality
                    torch.save(init_model.state_dict(), f'./checkpoints/{arg.application}-ddp-{arg.world_size}/{arg.name}-{time.strftime("%m%d%H%M")}.pth')

    # save the model by the last iteration
    if rank == 0:
        torch.save(init_model.state_dict(), f'./checkpoints/{arg.application}-ddp-{arg.world_size}/{arg.name}_last-{time.strftime("%m%d%H%M")}.pth')
    
    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument("--application", type=str, default="cosmoflow",
                        help="CFD or fluidanimation or puremd or cosmoflow or EMDenoise or minist "
                        "or DMS or optical or stemdl, slstr or synthetic or cifar10")
    parser.add_argument("--device", type=str, default='0', help="0, 1, ...")
    parser.add_argument("--tensorboard", action='store_false',
                        help='whether to use tensorboard (default: True)')
    parser.add_argument("--task", type=str, default='ordinary', help="prune or ordinary")
    parser.add_argument("--val_freq", type=int, default=200)
    parser.add_argument("--world_size", type=int, default=4)
    params = parser.parse_args()

    # Add CUDA availability check before setting environment variables
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = params.device

    # add additional arguments from settings.json
    with open(f'./applications/{params.application}/settings.json', 'r') as f:
        settings = json.load(f)
        for key, value in settings.items():
            # if argument already exists, replace it
            if key in params:
                setattr(params, key, value)
            else:
                parser.add_argument(f'--{key}', type=type(value), default=value)

    args = parser.parse_args()

    # set the name of the experiment
    args.name = 'pruned' if args.use_reg else 'original'
    args.task = 'prune' if args.use_reg else 'ordinary'
    
    # Create tensorboard writer only in the main process
    writer = None
    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer_path = f'./runs/{args.application}/{args.name}'
        # if path exists, add timestamp
        if os.path.exists(writer_path):
            writer_path += f'_{time.strftime("%m%d%H%M")}'
        
        # Move writer creation to the train function instead
        writer_info = {'path': writer_path, 'use_tensorboard': args.tensorboard}
    else:
        writer_info = None

    # create the directories to save the checkpoints
    os.makedirs(f'./checkpoints/{args.application}-ddp-{args.world_size}', exist_ok=True)

    # train(args)

    run_demo(train, args.world_size, writer_info)  # Pass writer_info instead of writer
