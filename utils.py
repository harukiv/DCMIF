import torch
import os
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


def fusion_gradient_loss(fusion_img, img1, img2):
    sobelconv = Sobelxy().requires_grad_(False)
    sobelconv.to(fusion_img.device)
    img1_grad = sobelconv(img1)
    img2_grad = sobelconv(img2)
    fusion_grad = sobelconv(fusion_img)
    loss_grad = F.l1_loss(fusion_grad, torch.maximum(img1_grad, img2_grad))
    return loss_grad


def intensity_loss(fusion_img, img1, img2):
    loss_intensity = F.l1_loss(fusion_img, torch.maximum(img1, img2))
    return loss_intensity

def gradient_loss(img1, img2):
    sobelconv = Sobelxy().requires_grad_(False)
    sobelconv.to(img1.device)
    img1_grad = sobelconv(img1)
    img2_grad = sobelconv(img2)
    loss_grad = F.l1_loss(img1_grad, img2_grad)
    return loss_grad

class Total_Loss(nn.Module):
    def __init__(self):
        super(Total_Loss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def forward(self, x, gt, x_sc, gt_sc, x_weight, vis):
        l1_loss = self.l1_loss(x, gt) + self.l1_loss(x_sc, gt_sc)
        region_loss = self.l1_loss(gt, x_weight)
        grad_loss = gradient_loss(x, vis)
        total_loss = grad_loss + l1_loss + region_loss

        return total_loss, region_loss, grad_loss, l1_loss


def save_checkpoint(model, optimizer, scheduler, epoch, path):
    os.makedirs(os.path.join(path, "all"), exist_ok=True)
    save_file = os.path.join(path, f"all/epoch_{epoch}.pth")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, save_file)


def save_weight(model, epoch, path):
    save_weight_path_epoch = os.path.join(path, f'epoch_{epoch}/')
    os.makedirs(save_weight_path_epoch, exist_ok=True)
    torch.save(model.scg.state_dict(), os.path.join(save_weight_path_epoch,'scg.pth'))
    torch.save(model.pig.state_dict(), os.path.join(save_weight_path_epoch,'pig.pth'))
    torch.save(model.text_prompt.state_dict(), os.path.join(save_weight_path_epoch,'text_prompt.pth'))

def save_stacked_image(tensor_stack, batch_size, i, name_list, epoch, path, prefix):
    save_folder = os.path.join(path, f'{prefix}_epoch{epoch}')
    os.makedirs(save_folder, exist_ok=True)

    for b in range(tensor_stack.shape[0]):
        name = name_list[i * batch_size + b]
        save_image(tensor_stack[b], os.path.join(save_folder, name))

def save_test_image(tensor_stack, batch_size, i, name_list, path):
    os.makedirs(path, exist_ok=True)

    for b in range(tensor_stack.shape[0]):
        name = name_list[i * batch_size + b]
        save_image(tensor_stack[b], os.path.join(path, name))
