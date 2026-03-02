import argparse
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
from model.AFRI.model_64 import Test
from torch.utils.data import DataLoader
from data.dataprocess.dataprocess_AFRI import TestDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import *

def main(args):
    device1 = torch.device('cuda:0')
    device2 = torch.device('cuda:1')
    sigma = torch.tensor(args.sigma)
    sigma = sigma.unsqueeze(0).repeat(2).to(device1)
    model = Test().to(device1)
    LLM_tokenizer = AutoTokenizer.from_pretrained(args.LLM_path, trust_remote_code=True)
    LLM_model = AutoModelForCausalLM.from_pretrained(args.LLM_path, trust_remote_code=True).to(device2).eval()

    transform_test = transforms.Compose([transforms.ToTensor()])
    test_dataset = TestDataset(img_path_vis=args.vis_path, transform=transform_test)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    checkpoint = torch.load(args.check_point)
    model.load_state_dict(checkpoint['model_state_dict'])
    D = torch.load(os.path.join(args.d_path, 'dictionary.pth'))

    model.eval()
    name_list = test_dataset.names()
    val_iterator = tqdm(testloader, dynamic_ncols=True)
    with torch.no_grad():
        for i, (VIS, cb, cr) in enumerate(val_iterator):
            VIS, cb, cr = VIS.to(device1), cb.to(device1), cr.to(device1)
            fuse = model(VIS, D, sigma, cb, cr, LLM_model, LLM_tokenizer)
            save_test_image(fuse, args.batch_size, i, name_list, args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--LLM_path', type=str, default='../LLM/Qwen')
    parser.add_argument('--save_path', type=str, default='../result')
    parser.add_argument('--vis_path', type=str, default='../data/FLIR/val/vis_day')
    parser.add_argument('--d_path', type=str, default='../weight/epoch_490')
    parser.add_argument('--check_point', type=str, default='../weight/epoch_0.pth')
    args = parser.parse_args()

    main(args)