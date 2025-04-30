import json
import os
import torch
import argparse
from tqdm import tqdm
import utils
import time
from utils import AverageMeter

torch.set_float32_matmul_precision('high')


def train_step(train_loader, model, criterion, optimizer, epoch, measure_quality, writer=None, print_detail=False, reg=False):
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

        loss, l0_loss = criterion(output, target, model, reg=reg)
        total_loss = loss + l0_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # clamp the parameters
        model.constrain_parameters()

        # measure accuracy and record loss
        measure_quality(output, target, quality, input_.size(0))

        losses.update(loss.item(), 1)
        l0_losses.update(l0_loss.item(), 1)
        
        # log to TensorBoard
        if writer is not None:
            writer.add_scalar('loss_detail/train', loss, epoch*len(train_loader)+i)
            writer.add_scalar('loss_detail/train_l0', l0_loss, epoch*len(train_loader)+i)

        if model.beta_ema > 0.:
            model.update_ema()

        if print_detail and i % 100 == 0:
            print(' Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                  'L0Loss {l0_loss.val:.6f} ({l0_loss.avg:.6f})\t'
                  'Acc {acc.val:.6f} ({acc.avg:.6f})'.format(epoch, i, len(train_loader), loss=losses,
                                                             l0_loss=l0_losses, acc=quality))

    return quality.avg, losses.avg, l0_losses.avg


def validate(val_loader, model, criterion, measure_quality, writer=None, print_detail=False, inference=False):
    """Perform validation on the validation set"""
    losses = AverageMeter()
    quality = AverageMeter()
    old_params = None

    # switch to evaluate mode
    model.eval()
    if model.beta_ema > 0 and not inference:
        old_params = model.get_params()
        model.load_ema_params()

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

    if model.beta_ema > 0 and not inference:
        model.load_params(old_params)

    return quality.avg, losses.avg


def train(arg):
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

    train_loader, val_loader = dataset.get_loader(batch_size=arg.batch_size)

    # move model to device
    init_model = init_model.cuda()

    # define the optimizer and scheduler
    optimizer = torch.optim.Adam(init_model.parameters(), lr=arg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=arg.step_size, gamma=arg.gamma) if arg.scheduler else None

    # creat tqdm progress bar
    pbar = tqdm(range(arg.epochs))

    # create tensorboard writer
    if arg.tensorboard:
        from tensorboardX import SummaryWriter
        writer_path = f'./runs/{arg.application}/{arg.name}'
        # if path exists, add timestamp
        if os.path.exists(writer_path):
            writer_path += f'_{time.strftime("%m%d%H%M")}'
                    
        writer = SummaryWriter(writer_path)
        # copy hyperparameters to tensorboard
        writer.add_text('hyperparameters', json.dumps(vars(arg), indent=4, sort_keys=True))
    else:
        writer = None

    # save the model by the best quality
    best_quality = 0 if arg.higher_better else 100

    # train the model
    for ite in pbar:
        # train step
        train_quality, loss, l0_loss = train_step(train_loader, init_model, launch.loss_fn, optimizer, ite,
                                                  measure_quality=launch.measure_quality, print_detail=False,
                                                  reg=arg.use_reg, writer=writer)
        # validate the model every arg.val_freq
        if (ite+1) % arg.val_freq == 0:
            val_quality, val_loss = validate(val_loader, init_model, launch.loss_fn, print_detail=False,
                                             inference=False, measure_quality=launch.measure_quality)

        # log to TensorBoard
        if writer is not None:
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

        # update the progress bar
        pbar.set_description(f'Training: Quality {train_quality:.6f}, Loss {loss:.6f}, L0 Loss {l0_loss:.6f}, '
                             f'Validation: Quality {val_quality:.6f}, Loss {val_loss:.6f}')
        
        epoch = ite
        if arg.use_reg and writer is not None:
            alive, total = 0, 0
            for k, layer in enumerate(init_model.layers):
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
        

        # save the model by the best quality
        if arg.higher_better:
            if val_quality > best_quality:
                best_quality = val_quality
                torch.save(init_model.state_dict(), f'./checkpoints/{arg.application}/{arg.name}-{time.strftime("%m%d%H%M")}.pth')
        else:
            if val_quality < best_quality:
                best_quality = val_quality
                torch.save(init_model.state_dict(), f'./checkpoints/{arg.application}/{arg.name}-{time.strftime("%m%d%H%M")}.pth')

    # save the model by the last iteration
    torch.save(init_model.state_dict(), f'./checkpoints/{arg.application}/{arg.name}_last-{time.strftime("%m%d%H%M")}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument("--application", type=str, default="puremd",
                        help="CFD or fluidanimation or puremd or cosmoflow or EMDenoise or minist "
                        "or DMS or optical or stemdl, slstr or synthetic or cifar10")
    parser.add_argument("--device", type=str, default='0', help="0, 1, ...")
    parser.add_argument("--tensorboard", action='store_false',
                        help='whether to use tensorboard (default: True)')
    parser.add_argument("--task", type=str, default='ordinary', help="prune or ordinary")
    parser.add_argument("--val_freq", type=int, default=1)
    params = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = params.device

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

    # create the directories to save the checkpoints
    os.makedirs(f'./checkpoints/{args.application}', exist_ok=True)

    train(args)
