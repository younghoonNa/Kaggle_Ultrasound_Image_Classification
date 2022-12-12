from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import numpy as np

import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

class Trainer():
    
    def __init__(self, model, device, optimizer, scheduler, model_name, flops):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_name = model_name
        self.flops = flops

    
    def train(self, train_loader, val_loader, n_epochs):
        early_stopcount = 0 #early stop count
        alpha = 0.25 # focal loss alhpa
        gamma = 2    # focal loss gamma
        
        train_loss = torch.zeros(n_epochs)
        valid_loss = torch.zeros(n_epochs)

        train_acc = torch.zeros(n_epochs)
        valid_acc = torch.zeros(n_epochs)
        
        valid_loss_min = np.Inf # track change in validation loss
        valid_acc_max = 0
            
        for e in range(0, n_epochs):
            ###################
            # train the model #
            ###################
            self.model.train()
            for data, labels in tqdm(train_loader):
                
                data, labels = data.to(self.device), labels.to(self.device) # move tensors to GPU if CUDA is available
                self.optimizer.zero_grad() # clear the gradients of all optimized variables
                logits = self.model(data) # forward pass: compute predicted outputs by passing inputs to the model
                
                # loss = criterion(logits, labels) # calculate the batch loss
                ce_loss = F.cross_entropy(logits, labels, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = alpha * (1 - pt) ** gamma * ce_loss
                loss = focal_loss.mean()
                
                loss.backward() # backward pass: compute gradient of the loss with respect to model parameters
                self.optimizer.step() # perform a single optimization step (parameter update)
                train_loss[e] += loss.item() # update training loss
                
                ps = F.softmax(logits, dim=1)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.reshape(top_class.shape)
                train_acc[e] += torch.mean(equals.type(torch.float)).detach().cpu()
            
            train_loss[e] /= len(train_loader)
            train_acc[e] /= len(train_loader)
                
                
            ######################    
            # validate the model #
            ######################
            with torch.no_grad(): 
                self.model.eval()
                for data, labels in tqdm(val_loader):
                    
                    data, labels = data.to(self.device), labels.to(self.device) # move tensors to GPU if CUDA is available
                    logits = self.model(data) # forward pass: compute predicted outputs by passing inputs to the model
                    # loss = criterion(logits, labels) # calculate the batch loss=
                    ce_loss = F.cross_entropy(logits, labels, reduction='none')
                    pt = torch.exp(-ce_loss)
                    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
                    loss = focal_loss.mean()
                    valid_loss[e] += loss.item() # update average validation loss 
                    
                    ps = F.softmax(logits, dim=1)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.reshape(top_class.shape)
                    valid_acc[e] += torch.mean(equals.type(torch.float)).detach().cpu()
                    
            
            # calculate average losses
            valid_loss[e] /= len(val_loader)
            valid_acc[e] /= len(val_loader)
            
            self.scheduler.step(valid_loss[e])    
            # print training/validation statistics 
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                e, train_loss[e], valid_loss[e]))
            
            # print training/validation statistics 
            print('Epoch: {} \tTraining accuracy: {:.6f} \tValidation accuracy: {:.6f}'.format(
                e, train_acc[e], valid_acc[e]))
            
            # save model if validation loss has decreased
            if valid_loss[e] <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss[e]))
                torch.save(self.model.module.state_dict(), '/data/ssd1/nyh/Ultrasound/Save_model/' + self.model_name +'.pt') #모델파일이 저장되고자하는 경로 지정
                valid_loss_min = valid_loss[e]
                early_stopcount = 0
                valid_acc_max = valid_acc[e]

            if early_stopcount >= 5:
                break

            early_stopcount+=1
            
        print(self.model_name, valid_loss_min.item(), valid_acc_max.item())
        filename = self.model_name + "_" + str(valid_loss_min.item()) + "_" + \
            str(valid_acc_max.item()) + "_" + str(self.flops) + ".txt"
        f = open(filename, "w")
        f.close()