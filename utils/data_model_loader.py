import torch.utils.data.dataloader as dataloader
from torchvision.datasets import FashionMNIST, MNIST
import torchvision.transforms as transforms
import os

def get_loader(dataset_name:str, image_transform, conf, download:bool):
    '''
    Returns the dataloader for the given dataset_name
    Params:
        dataset_name: str  Name of the dataset to load
        image_transform: torch transform   Transformation used for loading data
        conf: Dictionary  Configuration Dictionary
        download: bool  Indicator of download the dataset or not
    Returns:
        train_dataset, test_dataset, train_loader, test_loader: All the training, testing, dataset and dataloader 
                                                                corresponding to the dataset_name.
    '''
    if dataset_name == "MNIST":
        train_dataset = MNIST('./data', train=True, download=download, transform=image_transform)

        test_dataset = MNIST('./data', train=False, download=download, transform=image_transform)
    else:
        train_dataset = FashionMNIST('./data', train=True, download=download, transform=image_transform)

        test_dataset = FashionMNIST('./data', train=False, download=download, transform=image_transform)

    dataloader_args = dict(shuffle=True, batch_size=conf["TRAIN_BATCH_SIZE"],num_workers=4, pin_memory=True) if cuda\
                 else dict(shuffle=True, batch_size=conf["TEST_BATCH_SIZE"])
    train_loader = dataloader.DataLoader(train_dataset, **dataloader_args)
    test_loader = dataloader.DataLoader(test_dataset, **dataloader_args)
    return train_dataset, test_dataset, train_loader, test_loader

def get_all_loaders_and_models(conf):
    '''
    Returns all the dataset, dataloader, trained models for MNIST and FASHION MNIST
    Params: 
        conf: Dictionary  Configuration Dictionary
    Returns:
        train_mnist_dataset, test_mnist_dataset, train_mnist_loader, test_mnist_loader           Dataset, Dataloader for MNIST
        train_fashion_dataset, test_fashion_dataset, train_fashion_loader, test_fashion_loader   Dataset, Dataloader for Fashion MNIST
        mnist_model1, mnist_model2, Trained Models for MNIST
        fashion_model1, fashion_model2  Trained Models for Fashion MNIST
    '''
    dataset_name = "MNIST"
    train_mnist_dataset, test_mnist_dataset, train_mnist_loader, test_mnist_loader = \
                    get_loader(dataset_name, transforms.ToTensor(), conf, True)
    dataset_name = "FASHION"
    train_fashion_dataset, test_fashion_dataset, train_fashion_loader, test_fashion_loader = \
                    get_loader(dataset_name, transforms.ToTensor(), conf, True)
    path = conf["SAVE_PATH"] + "/"
    trained = len(os.listdir(path))>=4
    if trained == False: 
        dataset_name = "MNIST"
        print("Ohhh, we haven't fully trained the 4 models yet")
        dataset_name = "MNIST"
        if "MNIST_model1_best.pth" not in os.listdir(path):
            print(dataset_name,"Training Model 1") 
            train_model("model1",train_mnist_loader, test_mnist_loader,conf, "MNIST")
            print("*"*30)
        if "MNIST_model2_best.pth" not in os.listdir(path):
            print("Training Model 2")
            train_model("model2",train_mnist_loader, test_mnist_loader,conf, "MNIST")
            print("*"*30)
        dataset_name = "FASHION"
        print(dataset_name,"Model Training")
        if "FASHION_model1_best.pth" not in os.listdir(path):
            print("Training Model 1")
            train_model("model1",train_fashion_loader, test_fashion_loader,conf, "FASHION")
            print("*"*30)
        if "FASHION_model2_best.pth" not in os.listdir(path):
            print("Training Model 2")
            train_model("model2",train_fashion_loader, test_fashion_loader,conf, "FASHION") 
            print("*"*30)
    else:
        print("Nice! We have trained the models and found from:", path)
    mnist_model1, mnist_model2, fashion_model1, fashion_model2 = load_model(conf)
    return train_mnist_dataset, test_mnist_dataset, train_mnist_loader, test_mnist_loader, train_fashion_dataset, test_fashion_dataset, train_fashion_loader, test_fashion_loader, mnist_model1, mnist_model2, fashion_model1, fashion_model2
