import sys
sys.path.append('..')

import os
import time
import random
import argparse
import numpy as np
import torch
from tqdm import tqdm

import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from dataset import MRNetDataset
from model import SSAMNet
import torchmetrics


def train_epoch(model, optimizer, data_loader, device):
    model.train()

    total_tloss = 0
    tauc = torchmetrics.AUROC(task="binary").to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)

    for i, (image, label, weight) in enumerate(data_loader):
        optimizer.zero_grad()

        image = image.to(device)
        label = label.to(device)
        weight = weight.to(device)
        prediction0 = model(image.float())

        loss = F.binary_cross_entropy_with_logits(prediction0, label, weight=weight)
        loss.backward()
        optimizer.step()

        total_tloss += loss.item()

        probas0 = torch.sigmoid(prediction0)

        tauc.update(probas0, label)

    train_loss_epoch = np.round(total_tloss / (i+1), 4)
    train_auc_epoch = np.round(tauc.compute().item(), 4)

    tauc.reset()

    return train_loss_epoch, train_auc_epoch


def main(args, i):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tb_writer = SummaryWriter()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_dataset = MRNetDataset(args.data_path, args.task, args.plane, train=True, transform=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=False)

    model = SSAMNet(cla_num=1, s_top=16, c_div=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.gamma)

    rindex = i + 1

    for epoch in range(args.epochs):

        # train
        train_loss, train_auc = train_epoch(model=model,
                                            optimizer=optimizer,
                                            data_loader=train_loader,
                                            device=device
                                            )

        scheduler.step()

        if (epoch + 1) == args.epochs:
            file_name = f'{args.model_name}_{args.task}_{args.plane}_{epoch + 1}.pth'
            model_params = model.state_dict()
            torch.save(model_params, result_dir + "/checkpoints/{0}/{1}/r{2}/".format(args.task, args.plane,
                                                                                     rindex) + f'{file_name}')

        tags = ["train_loss", "train_auc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_auc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)


        print(
            "train loss : {0} | train auc {1} ".format(
                train_loss, train_auc))
        print('-' * 30)



def parse():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.00001)

    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--dataset_name', type=str, default='Mrnetdata')
    parser.add_argument('-t', '--task', type=str, choices=['abnormal', 'acl', 'meniscus'], default="acl")
    parser.add_argument('-p', '--plane', type=str, choices=['sagittal', 'coronal', 'axial'], default="sagittal")
    parser.add_argument('--localtime', type=str, required=False,
                        default=time.strftime('%Y-%m-%d-%H-%M', time.localtime()))

    parser.add_argument('--data-path', type=str,
                        default="/home/ubuntu/datasets/MRNet-v1.0/")
    parser.add_argument('--model_name', default='ours_SSAMNet')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--gamma', type=float, default=0.95, help='gamma')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    result_dir = os.path.join('results', args.dataset_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        os.makedirs(os.path.join(result_dir, 'checkpoints'))

    for i in range(3):
        checkf_dir = result_dir + "/checkpoints/{0}/{1}/r{2}/".format(args.task, args.plane, i+1)
        if not os.path.exists(checkf_dir):
            os.makedirs(checkf_dir)
        main(args, i)
