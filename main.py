import torch
import numpy as np
import random
import datetime
import json
import time
from pathlib import Path

from torch.utils.data import DataLoader

from model.table_detection_model import TableDetectionModel
from model.loss import SetCriterion
from args import MainArgs
from data.table_bank import build
import util.misc as utils
from training import train_one_epoch, evaluate


def main(args):
    """set seed and specify divice"""
    device = torch.device(args.device)
    print('current device: ', device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    """model and loss configuration"""
    model_cfg = args.model_cfg
    crit_cfg = args.criterion_cfg
    model = TableDetectionModel(model_cfg)
    model.to(device)
    criterion = SetCriterion(num_classes=crit_cfg.num_classes,
                             matcher=crit_cfg.matcher,
                             weight_dict=crit_cfg.weight_dict,
                             eos_coef=crit_cfg.eos_coef,
                             losses=crit_cfg.losses)
    criterion.to(device)
    
    """specify which parameters in the model to train"""
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    param_dict = [{"params": [p for n, p in model.named_parameters() if "encoder" not in n and p.requires_grad]}]
    optimizer = torch.optim.AdamW(param_dict, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    """load the data"""
    if args.only_data_subset:
        dataset_train = build('subset', args.data_dir)
        dataset_val = build('subset', args.data_dir)
    else:
        dataset_train = build('train', args.data_dir)
        dataset_val = build('test', args.data_dir)

    print(f"training size = {len(dataset_train)}, validation size = {len(dataset_val)}")

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.my_collate, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.my_collate, num_workers=args.num_workers)

    output_dir = Path(args.output_dir)
    """resume checkpoint if .pth file is specified"""
    if args.resume:
        checkpoint_file = f"{args.output_dir}{args.resume}"
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    """only evaluate the model if in eval mode"""
    if args.eval:
        test_stats = evaluate(model, criterion, postprocessors=None,
                              data_loader=data_loader_val, base_ds=None, device=device,
                              output_dir=output_dir)
        #now save eval
        return

    """the training loop"""
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.checkpoint_freq == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        """now do test stats"""
        test_stats = evaluate(model, criterion, postprocessors=None,
                              data_loader=data_loader_val, base_ds=None, device=device,
                              output_dir=output_dir)

        """save stats"""
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    #args = MainArgs()
    #args = MainArgs(resume='checkpoint.pth')
    args = MainArgs(data_dir='/home/ubuntu/ds-volume/semantic-td/data/',
                    output_dir='/home/ubuntu/ds-volume/semantic-td/data/ckpnt/',
                    only_data_subset=False,
                    batch_size=8,
                    num_workers=6,
                    epochs=10,
                    checkpoint_freq=1,
                    lr=1e-3,
                   )
    args = MainArgs(data_dir='/home/ubuntu/ds-volume/semantic-td/data/',
                    output_dir='/home/ubuntu/ds-volume/semantic-td/data/ckpnt/testing/',
                    only_data_subset=True,
                    batch_size=2,
                    num_workers=0,
                    epochs=300,
                    checkpoint_freq=100,
                    lr=1e-4,
                    weight_decay=0,
                   )    
    main(args)
