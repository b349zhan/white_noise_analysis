import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import torch.optim as optim

from tqdm import tqdm
from time import sleep
import pickle
from models.model1 import Model1


def train_model(model_name:str, train_loader, test_loader, save_path:str,cuda:bool, datasetName = "MNIST",EPOCHS = 15):
    if model_name == "model1":
        model = Model1()
    else:
        model = Model2()
    
    losses = []
    best_acc = 0
    optimizer = optim.Adam(model.parameters(), lr=1e-3) 
    model.train()
    
    for epoch in range(EPOCHS):
        
        # Training
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)

            if cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            
            # Forward
            y_pred = model(data)[0]
            loss = F.cross_entropy(y_pred, target)
            losses.append(loss.cpu().data)
            #losses.append(loss.cpu().data[0])     
    
            # Backward
            loss.backward()
            optimizer.step()

            # Display
            if batch_idx % 100 == 1:
                print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1,
                    EPOCHS,
                    batch_idx * len(data), 
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), 
                    loss.cpu().data), 
                    end='')
                
        # Validation
        evaluate_x = Variable(test_loader.dataset.data.type_as(torch.FloatTensor()))
        evaluate_y = Variable(test_loader.dataset.targets)
        if cuda:
            evaluate_x, evaluate_y = evaluate_x.cuda(), evaluate_y.cuda()

        model.eval()
        output = model(evaluate_x[:,None,...])[0]
        pred = output.data.max(1)[1]
        d = pred.eq(evaluate_y.data).cpu()
        accuracy = d.sum().type(dtype=torch.float64)/d.size()[0]

        # save best
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({'epoch': epoch,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()
                     }, f'{save_path}/{datasetName}_epoch_{epoch}.pth')
            print('\r Best model saved.\r')

        print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Test Accuracy: {:.4f}%'.format(
            epoch+1,
            EPOCHS,
            len(train_loader.dataset), 
            len(train_loader.dataset),
            100. * batch_idx / len(train_loader), 
            loss.cpu().data,
            accuracy*100,
            end=''))