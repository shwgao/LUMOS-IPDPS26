import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from ogb.graphproppred import PygGraphPropPredDataset


def get_loader(batch_size=100, augment=True, val_only=False, num_workers=16):
    dataset = PygGraphPropPredDataset(name = "ogbg-ppa", transform = add_zeros)

    split_idx = dataset.get_idx_split()

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True, num_workers = num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False, num_workers = num_workers)
    # test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    return train_loader, valid_loader