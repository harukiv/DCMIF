import argparse
import torch
import os
import torch.nn as nn
import torchvision
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from tqdm import tqdm
from model.VGII.model_64 import PIIG
from torch.utils.data import DataLoader
from data.dataprocess.dataprocess_VGII import TrainDataset_VGII
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import *

def main(args):
    device1 = torch.device('cuda:0')
    device2 = torch.device('cuda:1')
    sigma = torch.tensor(args.sigma)
    sigma = sigma.unsqueeze(0).repeat(2).to(device1)
    model = PIIG().to(device1)
    loss_fn = Total_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    LLM_tokenizer = AutoTokenizer.from_pretrained(args.LLM_path, trust_remote_code=True)
    LLM_model = AutoModelForCausalLM.from_pretrained(args.LLM_path, trust_remote_code=True).to(device2).eval()

    transform_train = transforms.Compose([transforms.ToTensor()])
    train_dataset = TrainDataset_VGII(img_path_ir=os.path.join(args.train_data_path, 'ir_day'),
                                      img_path_vis=os.path.join(args.train_data_path,'vis_day'),
                                      transform=transform_train)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers)

    if args.check_point:
        checkpoint = torch.load(args.check_point)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    D = torch.load(os.path.join(args.load_path, 'dictionary.pth'))
    for param in model.scg.parameters():
        param.requires_grad = False
    for param in model.tail.parameters():
        param.requires_grad = False
    for p in model.text_prompt.bert.parameters():
        p.requires_grad = False
    D.requires_grad = False

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        L1 = 0
        L2 = 0
        L3 = 0
        total_loss = 0
        epoch_iterator = tqdm(trainloader, desc=f'Train Epoch {epoch+1}/{args.epochs}', dynamic_ncols=True)

        for i, (IR, VIS) in enumerate(epoch_iterator):
            IR, VIS = IR.to(device1), VIS.to(device1)
            ir_p, ir_sc, ir_p_sc, _, ir_weight = model(IR, VIS, D, sigma, LLM_model, LLM_tokenizer)
            loss, l1, l2, l3 = loss_fn(ir_p, IR[:, :1, :, :], ir_p_sc, ir_sc, ir_weight, VIS)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            L1 += l1.item()
            L2 += l2.item()
            L3 += l3.item()
            total_loss += loss.item()
            avg_loss = total_loss / len(epoch_iterator)

            save_checkpoint(model, optimizer, scheduler, epoch, args.save_weight_path)
            save_weight_path_epoch = os.path.join(args.save_weight_path, f'epoch_{epoch}/')
            os.makedirs(save_weight_path_epoch, exist_ok=True)
            torch.save(model.scg.state_dict(), os.path.join(save_weight_path_epoch, 'scg.pth'))
            torch.save(model.pig.state_dict(), os.path.join(save_weight_path_epoch, 'pig.pth'))
            torch.save(model.text_prompt.state_dict(), os.path.join(save_weight_path_epoch, 'text_prompt.pth'))

            epoch_iterator.set_postfix(avg_loss=f'{avg_loss:.6f}')

        print(f'region_loss:{L1},grad_loss:{L2},l1_loss:{L3}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--batch_size_train', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--milestones', type=int, nargs='+', default=[10, 20, 30, 40])
    parser.add_argument('--LLM_path', type=str, default='../LLM/Qwen')
    parser.add_argument('--train_data_path', type=str, default='../data/MSRS/train')
    parser.add_argument('--load_path', type=str, default='../weight/epoch_500')
    parser.add_argument('--save_weight_path', type=str, default='../results/weight')
    parser.add_argument('--check_point', type=str, default=None)
    args = parser.parse_args()

    main(args)