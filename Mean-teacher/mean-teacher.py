from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from einops import reduce
import timm
import cv2
from thop import profile
import pandas as pd
import numpy as np

cnt = 0


def train(args, model, mean_teacher, device, train_loader, test_loader, optimizer, scheduler, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = reduce(data, 'b (c c1) h w -> b c h w', 'mean', c1=3)
        optimizer.zero_grad()

        output = model(data)
        output = F.log_softmax(output, dim = 1)

        ########################### CODE CHANGE HERE ######################################
        # forward pass with mean teacher
        # torch.no_grad() prevents gradients from being passed into mean teacher model
        with torch.no_grad():
            mean_t_output = mean_teacher(data)
            mean_t_output = F.log_softmax(mean_t_output, dim = 1)

        ########################### CODE CHANGE HERE ######################################
        # consistency loss (example with MSE, you can change)
        const_loss = F.mse_loss(output, mean_t_output)

        ########################### CODE CHANGE HERE ######################################
        # set the consistency weight (should schedule)
        weight = 0.2
        loss = F.nll_loss(output, target) + weight*const_loss
        loss.backward()
        optimizer.step()

        ########################### CODE CHANGE HERE ######################################
        # update mean teacher, (should choose alpha somehow)
        # Use the true average until the exponential average is more correct
        alpha = 0.95
        for mean_param, param in zip(mean_teacher.parameters(), model.parameters()):
            mean_param.data.mul_(alpha).add_(1 - alpha, param.data)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.4f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            min_loss = test(args, model, False, device, test_loader)
            test(args, mean_teacher,True, device, test_loader)
    scheduler.step(loss)


def test(args, model, teacher, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    global cnt
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = reduce(data, 'b (c c1) h w -> b c h w', 'mean', c1=3)
            output = model(data)
            output = F.log_softmax(output, dim = 1)

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    model.train()

    if not teacher:
        print(f"best score : {cnt}")

    if not teacher and cnt < correct:
        torch.save(model.module.state_dict(), "mean-teacher.pt")
        cnt = correct
        print("Save Model..")

class CustomDataset(Dataset):
    def __init__(self, img_paths, labels=None, transforms=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        # image = Image.open(img_path).convert('RGB')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        if self.labels is not None:
            label = self.labels[index]
            return image, label
        else:
            return image
    
    def __len__(self):
        return len(self.img_paths)

# 데이터 경로를 리턴해주는 함수를 통해 이미지 경로 추출
def get_data(df, test = False):
    if test is True:
        return "./test/"+df['Image_name'].values
    return "./train/"+df['Image_name'].values+'.png', df['Plane'].values
    
def flops(model, device, size, batchsize):
    inputs = torch.randn(batchsize, 1, size, size).cuda() # 자신의 model input size
    macs, params = profile(model.module.cuda(), inputs=(inputs, ))
    flops = macs*2
    gflops = round(flops/1000000000,7)
    print("내 모델의 FLOPs : ",gflops, "GFLOP")

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    df = pd.read_csv('./train.csv')
    train_df, val_df, _, _ = train_test_split(df, df['Plane'].values, test_size=0.2, shuffle=True, stratify=df['Plane'].values, random_state=34)


    train_img_paths, train_labels = get_data(train_df)
    val_img_paths, val_labels = get_data(val_df)

    transform = A.Compose([
        A.Resize(width = 224, height = 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomGridShuffle(grid = (7,7)),
        A.Equalize(always_apply=False, p=0.5, mode='cv', by_channels=False),
        A.GaussNoise(p = 0.5),
        A.GaussianBlur(p=0.5),
        A.Rotate(limit=15),
        A.Normalize(mean = [0.1776098, 0.1776098, 0.1776098], std= [0.037073515, 0.037073515, 0.037073515]),
        ToTensorV2(p=1.0)
    ])

    transform_val = A.Compose([
        A.Resize(width = 224, height = 224),
        A.Normalize(mean = [0.1776098, 0.1776098, 0.1776098], std = [0.037073515, 0.037073515, 0.037073515]),
        ToTensorV2(p=1.0)
    ])

    batchsize = 16
    train_dataset = CustomDataset(train_img_paths, train_labels, transforms = transform)
    train_loader = DataLoader(train_dataset, batch_size = batchsize, shuffle=True, num_workers=4)
    val_dataset = CustomDataset(val_img_paths, val_labels, transforms = transform_val)
    val_loader = DataLoader(val_dataset, batch_size = batchsize, shuffle=True, num_workers=4)
    # model = Net().to(device)
    model_name = "nf_regnet_b1"
    model = timm.create_model(model_name, num_classes=8, in_chans = 1, pretrained = True).to(device)
    model = nn.DataParallel(model).to(device)
    ########################### CODE CHANGE HERE ######################################
    # initialize mean teacher
    # mean_teacher = Net().to(device)
    mean_teacher = timm.create_model('nf_regnet_b1', num_classes=8, in_chans = 1, pretrained = True).to(device) 
    mean_teacher = nn.DataParallel(mean_teacher).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    # flops(model = model, device=device, size= 224, batchsize=16)

    for epoch in range(1, args.epochs + 1):
        train(args, model, mean_teacher, device, train_loader, val_loader, optimizer, scheduler, epoch)

    if (args.save_model):
        torch.save(model.module.state_dict(), "mean-teacher.pt")


if __name__ == '__main__':
    main()