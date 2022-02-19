from tqdm import tqdm
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

def generate_white_noise(model, conf):
    '''
    Given a model, randomly generate white noise based on the input image size, return the noise and 
    the summary of model's performance
    Params:
        model: Trained Model
        iteration: int  The number of times to generate noises
        batch_size: int The noise population size within each batch. 
                    (The number of noisy images generated in each batch)
        image_size: size of image that the model is trained on
        cuda: GPU enabled or not
    Returns:
        noise: Dictionary of tensors. Each key is the class label, each corresponding value is 
                                      generated noisy images
        noisy_image_summary: Dictionary of int. Each key is the class label, each corresponding
                                      value is the statistics of model's predictions across 10 classes.
    '''
    model.eval()
    noise = {}
    noisy_image_summary = {}
    for class_num in range(conf["CLASS_NUM"]):
        noise[class_num] = []
        noisy_image_summary[class_num] = 0
    for i in tqdm(range(conf["NOISE_ITERATION"])):
        z = torch.rand(conf["NOISE_BATCH_SIZE"],conf["IMAGE_SIZE"],conf["IMAGE_SIZE"])
        if conf["cuda"]:
            z = z.cuda()
        with torch.no_grad():
            prediction = model(z[:,None,...]).max(1)[1]
        for class_num in range(conf["CLASS_NUM"]):
            noise[class_num].append(z[prediction==class_num].cpu())
            noisy_image_summary[class_num] += (prediction == class_num).sum()
    for class_num in range(conf["CLASS_NUM"]):
        if noisy_image_summary[class_num]==0:
            noise[class_num] = torch.zeros(1, image_size, image_size)
        else:
            noise[class_num] = torch.cat(noise[class_num])
    return noise, noisy_image_summary

def visualize_noise_summary(noisy_image_summary,class_num):
    '''
    Plot histogram of noisy_image_summary across 10 classes
    Visualize noise for each class
    Params:
        noise: Dictionary of tensors. Each key is the class label, each corresponding value is 
                                      generated noisy images
        noisy_image_summary: Dictionary of int. Each key is the class label, each corresponding
                                      value is the statistics of model's predictions across 10 classes.
    Returns:
        None
    '''
    plt.xlabel('Class Labels')
    plt.ylabel('Model Prediction Count')
    plt.title(f'Model Prediction Count Across {class_num} Class Labels')
    plt.bar(list(range(class_num)), [noisy_image_summary[i] for i in range(class_num)])
    plt.xticks(list(range(class_num)), [str(i) for i in range(class_num)])
    plt.grid(True)
    plt.show()
    plt.ioff() 
    
def visualize_noise(noise, noise_summary,class_num,embedded=False):
    for class_label in range(class_num):
        if embedded==False:
            a = noise[class_label].mean(0)
            if noise_summary[class_label]>0:
                a = (a-a.min())/(a.max()-a.min())
        else:
            a = noise[class_label]
        plt.subplot(2, 5, class_label+1)
        plt.axis('off')
        plt.title("Class:"+str(class_label))
        plt.imshow(a)
    plt.suptitle("Average Noise Maps for Each Class")
    plt.ioff()

def generate_embedded_noise(model, dataset, conf):
    model.eval()
    
    signal = dataset.data.type(torch.FloatTensor)/256
    class_num = conf["CLASS_NUM"]
    class_stimulus = {i:[] for i in range(class_num)}
    class_noise = {i:[] for i in range(class_num)}
    cuda = conf["cuda"]
    image_size = conf["IMAGE_SIZE"]
    noisy_image_summary = {i:0 for i in range(class_num)}
    for i in tqdm(range(conf["NOISE_ITERATION"])):
        
        noise = torch.rand(conf["NOISE_BATCH_SIZE"],conf["IMAGE_SIZE"],conf["IMAGE_SIZE"])
        if conf["cuda"]: n = n.cuda()
        
        stimulus = conf["GAMMA"] * signal + (1 - conf["GAMMA"]) * noise
        if cuda: stimulus = stimulus.cuda()
            
        predictions = model(stimulus[:,None,...]).data.max(1)[1]
        for class_label in range(class_num):
            
            class_index = (predictions == class_label)
            if class_index.sum()==0:continue
            noisy_image_summary[class_label] += class_index.sum()
            class_stimulus_mean = stimulus[class_index].mean(dim=0)
            class_noise_mean = noise[class_index].mean(dim=0)
            class_stimulus[class_label].append(class_stimulus_mean)
            class_noise[class_label].append(class_noise_mean)
            
            
    for class_label in range(class_num):
        if len(class_noise[class_label])==0:
            class_noise[class_label] = torch.zeros(image_size, image_size)
            continue
        class_noise[class_label] = torch.mean(torch.stack(class_noise[class_label]),dim=0)
    return class_noise, noisy_image_summary

def feed_back_visualize(model, dataset, noise_map, conf):
    predictions = []
    class_num = conf["CLASS_NUM"]
    for class_label in range(class_num):
        batch_noise_image = noise_map[class_label].reshape(1,1,noise_map[0].shape[0],-1)
        batch_noise_image = batch_noise_image - batch_noise_image.min().item()
        batch_noise_image_normalized = batch_noise_image / batch_noise_image.max().item()
        output = model(batch_noise_image_normalized)
        prediction = torch.argmax(output)
        predictions.append(prediction.data)
    confusion = confusion_matrix(list(range(class_num)) ,predictions)
    plt.figure(figsize = (8,8))
    confusion = confusion/np.sum(confusion, axis = 1)
    confusionDf = pd.DataFrame(confusion, index=[i for i in list(range(class_num))], columns=[
                             i for i in list(range(class_num))])
    sns.heatmap(round(confusionDf, 2), annot=True, annot_kws={"size": 14})
    plt.title("Confusion Matrix for Noise Maps", fontsize=10)
    
def get_normalized_noise_map(noise_map):
    noise_map = noise_map.numpy()
    normalized_noise_map = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
    return torch.from_numpy(normalized_noise_map)

def noise_map_classify_and_report(data_loader, noise_maps, conf):
    noise_maps_normalized = [ noise_maps[class_label] for class_label in noise_maps]
    noise_maps_classifier = torch.stack(noise_maps_normalized,dim=0)
    truth = data_loader.dataset.targets
    predictions = []
    for (data, label) in data_loader:
        data = data.reshape(data.shape[0],1, data.shape[-1], data.shape[-1])
        probabilities = torch.sum(data * noise_maps_classifier, dim = [-2,-1])
        batch_prediction = torch.argmax(probabilities, dim = -1)
        predictions.extend(batch_prediction)
    confusion = confusion_matrix(truth, predictions, normalize = "true")
    plt.figure(figsize = (8,8))
    class_num = conf["CLASS_NUM"]
    confusionDf = pd.DataFrame(confusion, index=[i for i in list(range(class_num))], columns=[
                             i for i in list(range(class_num))])
    sns.heatmap(round(confusionDf, 2), annot=True, annot_kws={"size": 14})
    plt.title("Confusion Matrix for Noise Maps Classifier Test Dataset Performance", fontsize=10)
    
def embed_noise_visualize_classify_report(model, data_loader, conf):
    '''
    
    '''
    noise_maps, noisy_image_summary = generate_embedded_noise(model, data_loader.dataset, conf)
    visualize_noise(noise_maps, noisy_image_summary, conf["CLASS_NUM"],embedded=True)
    feed_back_visualize(model, data_loader.dataset, noise_maps, conf)
    noise_map_classify_and_report(data_loader, noise_maps,conf)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    