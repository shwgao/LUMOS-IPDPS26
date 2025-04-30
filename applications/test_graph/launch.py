import torch
from utils import accuracy
from ogb.graphproppred import Evaluator

multicls_criterion = torch.nn.CrossEntropyLoss()


def loss_fn(output, target, model, reg=False):
    loss = multicls_criterion(output.to(torch.float32), target.view(-1,))
    l0_loss = torch.tensor(0).cuda()

    if reg:
        l0_loss = model.regularization()

    return loss, l0_loss


def measure_quality(output, target, quality, input_size):
    y_true = append(target.view(-1,1).detach().cpu())
    y_pred = append(torch.argmax(output.detach(), dim = 1).view(-1,1).cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    
    quality.update(corrects, input_size)
