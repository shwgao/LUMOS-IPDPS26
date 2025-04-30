import torch

criteria = torch.nn.BCELoss()


def loss_fn(output, target, model, reg=False):
    # merge the first and the second dimension of output
    target = target.view(-1, target.size(2), target.size(3), target.size(4))
    loss = criteria(output, target)
    l0_loss = torch.tensor(0).to(output.device)

    if reg:
        l0_loss = model.regularization()

    return loss, l0_loss


def measure_quality(output, target, quality, input_size):
    target = target.view(-1, target.size(2), target.size(3), target.size(4))
    mse = criteria(output, target)
    quality.update(mse.item(), 1)
