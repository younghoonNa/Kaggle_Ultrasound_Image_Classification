import numpy as np
import pandas as pd 
import os

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader, WeightedRandomSampler
from PIL import Image 
import torch.nn as nn
from tqdm.auto import tqdm

import cv2
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


from trainer import Trainer

from thop import profile

def set_seed(seed=0):
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
#실험 재현을 위한 랜덤시드 고정
set_seed(42)

class CustomDataset(Dataset):
    def __init__(self, img_paths, labels=None, transforms=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = Image.open(img_path).convert('RGB')
        
        if self.transforms is not None:
            image = self.transforms(image)
        
        if self.labels is not None:
            label = self.labels[index]
            return image, label
        else:
            return image
    
    def __len__(self):
        return len(self.img_paths)

def main(image_size, model_name):
    os.chdir('/data/ssd1/nyh/Ultrasound')
    
    df = pd.read_csv('./train.csv')
    train_df, val_df, _, _ = train_test_split(df, df['Plane'].values, test_size=0.1, shuffle=True, stratify=df['Plane'].values, random_state=34)
    train_df.head()
    
    # 데이터 경로를 리턴해주는 함수를 통해 이미지 경로 추출
    def get_data(df, test = False):
        if test is True:
            return "./test/"+df['Image_name'].values
        return "./train/"+df['Image_name'].values+'.png', df['Plane'].values

    train_img_paths, train_labels = get_data(train_df)
    val_img_paths, val_labels = get_data(val_df)
    
    transform = transforms.Compose([
                                    transforms.Resize((image_size,image_size)),
                                    transforms.RandomRotation(degrees=15),
                                    transforms.ToTensor(),
                                    transforms.Grayscale(),
                                    transforms.Normalize([0.1776098, 0.1776098, 0.1776098], [0.037073515, 0.037073515, 0.037073515]),                                
                                ])

    train_dataset = CustomDataset(train_img_paths, train_labels, transform)
    train_loader = DataLoader(train_dataset, batch_size = 16, num_workers=4)

    val_dataset = CustomDataset(val_img_paths, val_labels, transform)
    val_loader = DataLoader(val_dataset, batch_size = 16, num_workers=4)
    
    model = timm.create_model(model_name, pretrained=True, num_classes = 8, in_chans = 1)
    model.to(device)
    inputs = torch.randn(1, 1, image_size, image_size).to(device) # 자신의 model input size
    macs, params = profile(model, inputs=(inputs, ))
    flops = macs*2
    gflops = round(flops/1000000000,3)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    

    # keep track of training and validation loss


    print("========================================================")
    print(f"Uisng Model Name : {model_name}  and Flops is {gflops} ++ Image size is ({image_size}, {image_size})")    
    print("========================================================")
    
    
    trainer = Trainer(
                model = model,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                model_name=model_name,
                flops = gflops)
    
    trainer.train(train_loader, val_loader, n_epochs=100)
        
        
            
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t = True
    for l, e in enumerate(timm.list_models()):
        # if e == "gluon_inception_v3":
        #     exit()
        print(e)
        if e == "visformer_small":
            t = False
        else:
            if t:
                continue

        
        image_size = 224
        model_name = e
        
        for word in e.split('_'):
            if word >= '000' and word <= "999" and word.__len__() == 3:
                image_size = int(word)
                break
        
        main(image_size, model_name) 
    