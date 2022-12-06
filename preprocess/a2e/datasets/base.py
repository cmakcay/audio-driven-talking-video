from torch.utils.data import DataLoader
from .custom_datasets import ds_demo

def build_dataset(args):
    dataset = ds_demo(args)
    return dataset

def build_dataloader(args, drop_last=False):
    dataset = build_dataset(args)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True, drop_last=drop_last)
    return dataloader