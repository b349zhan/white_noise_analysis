from tqdm import tqdm
import sys
import torch
import matplotlib.pyplot as plt

def generate_white_noise(model, iteration:int, batch_size:int, image_size:int, cuda:bool):
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
    for class_num in range(10):
        noise[class_num] = []
        noisy_image_summary[class_num] = 0
    for i in tqdm(range(iteration)):
        z = torch.rand(batch_size, image_size, image_size)
        if cuda:
            z = z.cuda()
        with torch.no_grad():
            prediction = model(z[:,None,...]).max(1)[1]
        for class_num in range(10):
            noise[class_num].append(z[prediction==class_num].cpu())
            noisy_image_summary[class_num] += (prediction == class_num).sum()
    for class_num in range(10):
        noise[class_num] = torch.cat(noise[class_num])
    return noise, noisy_image_summary

def visualize_noise_summary(noisy_image_summary):
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
    plt.title('Model Prediction Count Across 10 Class Labels')
    plt.bar(list(range(10)), [noisy_image_summary[i] for i in range(10)])
    plt.xticks(list(range(10)), [str(i) for i in range(10)])
    plt.grid(True)
    plt.show()
    plt.ioff() 
    
def visualize_noise(noise,embedded=False):
    for class_label in range(10):
        if embedded==False:
            a = noise[class_label].mean(0)
            a = (a-a.min())/(a.max()-a.min())
        else:
            a = noise[class_label]
        plt.subplot(2, 5, class_label+1)
        plt.axis('off')
        plt.title(str(class_label))
        plt.imshow(a)
    plt.suptitle("Predictions for Randomly Generated Noises")
    plt.ioff()

def generate_embedded_noise(model, dataset, gamma:float, iteration:int, batch_size:int, image_size:int, cuda:bool):
    model.eval()
    
    signal = dataset.data.type(torch.FloatTensor)/256
    class_stimulus = {i:[] for i in range(10)}
    class_noise = {i:[] for i in range(10)}
    for i in tqdm(range(iteration)):
        
        noise = torch.rand(batch_size, image_size, image_size)
        if cuda: n = n.cuda()
        
        stimulus = gamma * signal + (1 - gamma) * noise
        if cuda: stimulus = stimulus.cuda()
            
        predictions = model(stimulus[:,None,...]).data.max(1)[1]
        for class_label in range(10):
            class_index = (predictions == class_label)
            class_stimulus_mean = stimulus[class_index].mean(dim=0)
            class_noise_mean = noise[class_index].mean(dim=0)
            class_stimulus[class_label].append(class_stimulus_mean)
            class_noise[class_label].append(class_noise_mean)
            
    for class_label in range(10):
        class_noise[class_label] = torch.mean(torch.stack(class_noise[class_label]),dim=0)
    return class_noise


                