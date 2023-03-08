# -*- coding: utf-8 -*-
import os
import sys
import logging

import numpy as np
import torch
import torchvision
import time

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
import skeleton
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

LOGGER = logging.getLogger(__name__)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="fast converge cifar10")

    parser.add_argument('--dataset-base', type=str, default='./data')
    parser.add_argument('--batch', type=int, default=500)
    parser.add_argument('--epoch', type=int, default=25)

    parser.add_argument('--download', action='store_true')
    parser.add_argument('--seed', type=lambda x: int(x, 0), default=None)

    parser.add_argument('--log-filename', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def dataloaders(base, download, batch_size, device):
    train_dataset = torchvision.datasets.CIFAR10(
        root=base + '/cifar10',
        train=True,
        download=download
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=base + '/cifar10',
        train=False,
        download=download
    )

    post_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2471, 0.2435, 0.2616)
        ),
    ])

    train_dataloader = skeleton.data.FixedSizeDataLoader(
        skeleton.data.TransformDataset(
            skeleton.data.prefetch_dataset(
                skeleton.data.TransformDataset(
                    train_dataset,
                    transform=torchvision.transforms.Compose([
                        skeleton.data.transforms.Pad(2),
                        post_transform
                    ]),
                    index=0
                ),
                num_workers=16
            ),
            transform=torchvision.transforms.Compose([
                skeleton.data.transforms.TensorRandomCrop(30, 30),
                skeleton.data.transforms.TensorRandomHorizontalFlip(),
                skeleton.data.transforms.Cutout(8, 8)
            ]),
            index=0
        ),
        steps=None,  # for prefetch using infinit dataloader
        batch_size=batch_size,
        num_workers=32,
        pin_memory=False,
        drop_last=True,
        shuffle=True,
        # sampler=skeleton.data.StratifiedSampler(train_dataset.targets)
    )

    test_dataloader = torch.utils.data.DataLoader(
        skeleton.data.prefetch_dataset(
            skeleton.data.TransformDataset(
                test_dataset,
                transform=torchvision.transforms.Compose([
                    # skeleton.data.transforms.Pad(4),
                    # torchvision.transforms.ToPILImage(),
                    # torchvision.transforms.TenCrop(32),
                    # torchvision.transforms.Lambda(
                    #     lambda tensors: torch.stack([
                    #         post_transform(tensor) for tensor in tensors
                    #     ], dim=0)
                    # )

                    torchvision.transforms.CenterCrop((30, 30)),
                    post_transform,
                    torchvision.transforms.Lambda(
                        lambda tensor: torch.stack([
                            tensor, torch.flip(tensor, dims=[-1])
                        ], dim=0)
                    )
                ]),
                index=0
            ),
            num_workers=16
        ),
        batch_size=batch_size // 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    train_dataloader = skeleton.data.PrefetchDataLoader(train_dataloader, device=device, half=True)
    test_dataloader = skeleton.data.PrefetchDataLoader(test_dataloader, device=device, half=True)
    return int(len(train_dataset) // batch_size), train_dataloader, test_dataloader


def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1, bn=True, activation=True):
    op = [
            torch.nn.Conv2d(channels_in, channels_out,
                            kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
    ]
    if bn:
        op.append(torch.nn.BatchNorm2d(channels_out))
    if activation:
        op.append(torch.nn.ReLU(inplace=True))
    return torch.nn.Sequential(*op)


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


def build_network(num_class=10):
    return torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        # torch.nn.MaxPool2d(2),

        Residual(torch.nn.Sequential(
            conv_bn(128, 128),
            conv_bn(128, 128),
        )),

        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),

        Residual(torch.nn.Sequential(
            conv_bn(256, 256),
            conv_bn(256, 256),
        )),

        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),

        torch.nn.AdaptiveMaxPool2d((1, 1)),
        skeleton.nn.Flatten(),
        torch.nn.Linear(128, num_class, bias=False),
        skeleton.nn.Mul(0.2)
    )


def main():
    timer = skeleton.utils.Timer()

    args = parse_args()
    log_format = '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)03d] %(message)s'
    level = logging.DEBUG if args.debug else logging.INFO
    if not args.log_filename:
        logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=level, format=log_format, filename=args.log_filename)

    torch.backends.cudnn.benchmark = True
    if args.seed is not None:
        skeleton.utils.set_random_seed_all(args.seed, deterministic=False)

    epoch = args.epoch
    batch_size = args.batch
    device = torch.device('cuda', 0)

    steps_per_epoch, train_loader, test_loader = dataloaders(args.dataset_base, args.download, batch_size, device)
    train_iter = iter(train_loader)
    # steps_per_epoch = int(steps_per_epoch * 1.0)

    model = build_network().to(device=device)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data.fill_(1.0)
            module.eps = 0.00001
            module.momentum = 0.1
        else:
            module.half()
        if isinstance(module, torch.nn.Conv2d) and hasattr(module, 'weight'):
            # torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))  # original
            torch.nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='linear')
            # torch.nn.init.xavier_uniform_(module.weight, gain=torch.nn.init.calculate_gain('linear'))
        if isinstance(module, torch.nn.Linear) and hasattr(module, 'weight'):
            # torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))  # original
            torch.nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='linear')
            # torch.nn.init.xavier_uniform_(module.weight, gain=1.)

    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    # criterion = skeleton.nn.CrossEntropyLabelSmooth(num_classes=10, epsilon=1e-3, reduction='sum')
    metrics = skeleton.nn.Accuracy(1)

    lr_scheduler = skeleton.optim.get_change_scale(
        skeleton.optim.get_piecewise([0, 4, epoch], [0.025, 0.4, 0.001]),
        1.0 / batch_size
    )
    optimizer = skeleton.optim.ScheduledOptimizer(
        [p for p in model.parameters() if p.requires_grad],
        torch.optim.SGD,
        steps_per_epoch=steps_per_epoch,
        lr=lr_scheduler,
        momentum=0.9,
        weight_decay=5e-4 * batch_size,
        nesterov=True
    )

    class ModelLoss(torch.nn.Module):
        def __init__(self, model, criterion):
            super(ModelLoss, self).__init__()
            self.model = model
            self.criterion = criterion

        def forward(self, inputs, targets):
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            return logits, loss
    model = ModelLoss(model, criterion)

    # warmup
    torch.cuda.synchronize()
    model.train()
    for _ in range(2):
        inputs, targets = next(train_iter)
        logits, loss = model(inputs, targets)
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    timer('init')

    # train
    results = ['epoch\thours\ttop1Accuracy']
    for epoch in range(epoch):
        model.train()
        train_loss_list = []
        timer('init', reset_step=True)
        for step in range(steps_per_epoch):
            inputs, targets = next(train_iter)
            logits, loss = model(inputs, targets)

            loss.sum().backward()
            train_loss_list.append(loss.detach() / batch_size)

            optimizer.update()
            optimizer.step()
            optimizer.zero_grad()
        timer('train')

        model.eval()
        accuracy_list = []
        test_loss_list = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                origin_targets = targets
                use_tta = len(inputs.size()) == 5
                if use_tta:
                    bs, ncrops, c, h, w = inputs.size()
                    inputs = inputs.view(-1, c, h, w)

                    targets = targets.view(bs, 1)
                    targets = torch.cat([targets for _ in range(ncrops)], dim=1)
                    targets = targets.view(bs * ncrops)

                logits, loss = model(inputs, targets)
                if use_tta:
                    logits = logits.view(bs, ncrops, -1).mean(1)

                accuracy = metrics(logits, origin_targets)
                accuracy_list.append(accuracy.detach())
                test_loss_list.append(loss.detach() / batch_size)
        timer('test')
        LOGGER.info(
            '[%02d] train loss:%.3f test loss:%.3f accuracy:%.3f lr:%.3f %s',
            epoch,
            np.average([t.cpu().numpy() for t in train_loss_list]),
            np.average([t.cpu().numpy() for t in test_loss_list]),
            np.average([t.cpu().numpy() for t in accuracy_list]),
            optimizer.get_learning_rate() * batch_size,
            timer
        )
        results.append('{epoch}\t{hour:.8f}\t{accuracy:.2f}'.format(**{
            'epoch': epoch,
            'hour': timer.accumulation['train'] / (60 * 60),
            'accuracy': float(np.average([t.cpu().numpy() for t in accuracy_list])) * 100.0
        }))
    print('\n'.join(results))
    # torch.save(model.state_dict(), 'assets/kakaobrain_custom-resnet9_single_cifar10.pth')


if __name__ == '__main__':
    start_time = time.time()
    # > python bin/dawnbench/cifar10.py --seed 0xC0FFEE --download > log_dawnbench_cifar10.tsv
    main()
    print('total time: {:.2f} seconds'.format((time.time() - start_time)))