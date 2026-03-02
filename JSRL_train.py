import argparse
import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
from model.JSRL.model_64 import DCDicl
from torch.utils.data import DataLoader
from data.dataprocess.dataprocess_JSRL import TrainDataset_JSRL
from utils import *


def main(args):
    sigma = torch.tensor(args.sigma)
    sigma = sigma.unsqueeze(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DCDicl().to(device)
    sigma = sigma.to(device)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    transform_train = transforms.Compose([transforms.ToTensor()])
    train_dataset = TrainDataset_JSRL(img_path_ir=os.path.join(args.train_path, 'ir'),
                                      img_path_vis=os.path.join(args.train_path, 'vis'),
                                      transform=transform_train)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=0)

    if args.check_point:
        checkpoint = torch.load(args.check_point)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    for epoch in range(args.start_epoch, args.epoches):
        model.train()
        total_loss = 0
        epoch_iterator = tqdm(trainloader, desc=f'Train Epoch {epoch+1}/{args.epoches}', dynamic_ncols=True)

        for i, (IR, VIS) in enumerate(epoch_iterator):
            IR, VIS = IR.to(device), VIS.to(device)
            ir, vis, d, _, _ = model(IR, VIS, sigma)
            loss1 = loss_fn(ir, IR)
            loss2 = loss_fn(vis, VIS)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            avg_loss = total_loss / len(trainloader)

            epoch_iterator.set_postfix(avg_loss=f'{avg_loss:.6f}')

        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, args.save_weight_path)
            save_weight_path_epoch = os.path.join(args.save_weight_path, f'epoch_{epoch}/')
            os.makedirs(save_weight_path_epoch, exist_ok=True)
            torch.save(d, os.path.join(save_weight_path_epoch, 'dictionary.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epoches', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--batch_size_train', type=int, default=24)
    parser.add_argument('--batch_size_val', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--milestones', type=int, nargs='+', default=[100, 200, 300, 400])
    parser.add_argument('--train_path', type=str, default='../data/MSRS/train')
    parser.add_argument('--save_weight_path', type=str, default='../weight')
    parser.add_argument('--check_point', type=str, default=None)
    args = parser.parse_args()

    main(args)