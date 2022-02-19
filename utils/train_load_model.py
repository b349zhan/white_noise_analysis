import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import torch.optim as optim

from tqdm import tqdm
import pickle

from models.model1 import Model1
from models.model2 import Model2


def train_model(model_name:str, train_loader, test_loader,conf, dataset_name = "MNIST"):
    '''
    Trains and save the model based on the model name and the data loaders.
    Params:
        model_name: str  Currently only supports "model1" and "model2" Initialize the model based on this string
        train_loader, test_loader torch dataloader   Dataloader for training and testing
        conf: Dictionary   Configuration Dictionary
        dataset_name: str  Default to be "MNIST", used for saving the model and encode this parameter's value in 
                            output file's name
    Returns:
        None
    '''    
    image_size, image_channel, class_num = conf["IMAGE_SIZE"], conf["IMAGE_CHANNEL"], conf["CLASS_NUM"]
    if model_name == "model1":
        model = Model1(image_size, image_channel, class_num)
    else:
        model = Model2(image_size, image_channel, class_num)
    optimizer = optim.Adam(model.parameters(), lr=1e-3) 
    losses = []
    best_accuracy = 0

    model.train()
    for epoch in range(conf["TRAIN_EPOCHS"]):
        for batch_idx, (data, target) in enumerate(train_loader):
            
            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()

            y_pred = model(data) 
            loss = F.cross_entropy(y_pred, target)
            losses.append(loss.cpu().data)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 1:
                print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1,
                    conf["TRAIN_EPOCHS"],
                    batch_idx * len(data), 
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), 
                    loss.cpu().data), 
                    end='')
        evaluate_x = Variable(test_loader.dataset.data.type_as(torch.FloatTensor()))
        evaluate_y = Variable(test_loader.dataset.targets)
        if cuda: evaluate_x, evaluate_y = evaluate_x.cuda(), evaluate_y.cuda()

        model.eval()
        output = model(evaluate_x[:,None,...])
        pred = output.data.max(1)[1]
        d = pred.eq(evaluate_y.data).cpu()
        accuracy = d.sum().type(dtype=torch.float64)/d.size()[0]

        print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Test Accuracy: {:.4f}%'.format(
            epoch+1,
            conf["TRAIN_EPOCHS"],
            len(train_loader.dataset), 
            len(train_loader.dataset),
            100. * batch_idx / len(train_loader), 
            loss.cpu().data,
            accuracy*100,
            end=''))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({'epoch': epoch,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()
                           }, f'{conf["SAVE_PATH"]}/{dataset_name}_{model_name}_best.pth')
            print('\r Best model saved.\r')

        
def load_model(conf):
    '''
    Load all 4 models that are saved previously.
    Params:
        conf: Dictionary   Configuration Dictionary
    Returns:
        mnist_model1, mnist_model2, fashion_model1, fashion_model2: Returns 4 already trained models
    '''
    image_size, image_channel, class_num = conf["IMAGE_SIZE"],conf["IMAGE_CHANNEL"],conf["CLASS_NUM"]
    mnist_model1, mnist_model2, fashion_model1, fashion_model2 = Model1(image_size, image_channel, class_num),\
                                                                 Model2(image_size, image_channel, class_num),\
                                                                 Model1(image_size, image_channel, class_num),\
                                                                 Model2(image_size, image_channel, class_num)
    mnist_model1.load_state_dict(torch.load(conf["SAVE_PATH"]+"/MNIST_model1_best.pth")["model"])
    mnist_model2.load_state_dict(torch.load(conf["SAVE_PATH"]+"/MNIST_model2_best.pth")["model"])
    fashion_model1.load_state_dict(torch.load(conf["SAVE_PATH"]+"/FASHION_model1_best.pth")["model"])
    fashion_model2.load_state_dict(torch.load(conf["SAVE_PATH"]+"/FASHION_model2_best.pth")["model"])
    return mnist_model1, mnist_model2, fashion_model1, fashion_model2