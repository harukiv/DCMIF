import argparse
import os.path
import time
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from model.AFRI.model_64 import PIVIF
from data.dataprocess.dataprocess_AFRI import TrainDataset_ARFI
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import *

def main(args):
    device1 = torch.device('cuda:0')
    device2 = torch.device('cuda:1')
    sigma = torch.tensor(args.sigma)
    sigma = sigma.unsqueeze(0).repeat(2).to(device1)
    model = PIVIF().to(device1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    LLM_tokenizer = AutoTokenizer.from_pretrained(args.LLM_path, trust_remote_code=True)
    LLM_model = AutoModelForCausalLM.from_pretrained(args.LLM_path, trust_remote_code=True).to(device2).eval()

    transform_train = transforms.Compose([transforms.ToTensor()])
    train_dataset = TrainDataset_ARFI(img_path_ir=os.path.join(args.train_data_path, 'ir_day'),
                                      img_path_vis=os.path.join(args.train_data_path, 'vis_day'),
                                      transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers)

    model.scg.load_state_dict(torch.load(os.path.join(args.load_path_pi, 'epoch_1/scg.pth')))
    model.pig.load_state_dict(torch.load(os.path.join(args.load_path_pi, 'epoch_1/pig.pth')))
    model.text_prompt.load_state_dict(torch.load(os.path.join(args.load_path_pi, 'epoch_1/text_prompt.pth')))
    model.tail.load_state_dict(torch.load(os.path.join(args.load_path_d, 'tail.pth')))
    D = torch.load(os.path.join(args.load_path_d, 'dictionary.pth'))

    if args.check_point:
        checkpoint = torch.load(args.check_point)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    for param in model.scg.parameters():
        param.requires_grad = False
    for param in model.pig.parameters():
        param.requires_grad = False
    for param in model.tail.parameters():
        param.requires_grad = False
    for param in model.text_prompt.parameters():
        param.requires_grad = False
    D.requires_grad = False

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        L1 = 0
        L2 = 0
        Loss = 0
        epoch_iterator = tqdm(train_loader, desc=f'Train Epoch {epoch + 1}/{args.epochs}', dynamic_ncols=True)
        for i, (IR, VIS, cb, cr) in  enumerate(epoch_iterator):
            IR, VIS, cb, cr = IR.to(device1), VIS.to(device1), cb.to(device1), cr.to(device1)
            fuse, fuse_rgb = model(IR, VIS, D, sigma, cb, cr, LLM_model, LLM_tokenizer)
            loss1 = intensity_loss(fuse, IR, VIS)
            loss2 = fusion_gradient_loss(fuse, IR, VIS)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            L1 += loss1.item()
            L2 += loss2.item()
            Loss += loss.item()
            avg_loss = Loss / len(epoch_iterator)

            epoch_iterator.set_postfix(avg_loss=f'{avg_loss:.6f}')

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(args.save_path_weight, f'epoch_{epoch}.pth'))

        print(f'intensity_loss:{L1},gradient_loss:{L2}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--batch_size_train', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--milestones', type=int, nargs='+', default=[10, 20, 30, 40])
    parser.add_argument('--LLM_path', type=str, default='../LLM/Qwen')
    parser.add_argument('--train_data_path', type=str, default='../data/MSRS/train')
    parser.add_argument('--load_path_pi', type=str, default='../weight')
    parser.add_argument('--load_path_d', type=str, default='../weight/epoch_500')
    parser.add_argument('--save_path_weight', type=str, default='../weight')
    parser.add_argument('--check_point', type=str, default=None)

    args = parser.parse_args()

    main(args)