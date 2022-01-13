import torch
from torch.utils.data import Dataset
from skimage.io import imread
import os
import numpy as np
import random
from torchvision import transforms, datasets

from random_generator2 import make_shape

class cifar_part(Dataset):
    def __init__(self, num=500):
        super().__init__()
        
        self.len = num
        
        self.dataset = datasets.CIFAR10(
                        "./data/CIFAR10",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                #transforms.RandomCrop(32, padding=4),
                                transforms.RandomResizedCrop(32),#, scale=(0.5,1)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                ),
                        )
        
        
    def __getitem__(self, index):
        return self.dataset.__getitem__(index)
    
    def __len__(self):
        return self.len
        

class random_set(Dataset):
    def __init__(self):
        super().__init__()
        
        color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                #transforms.RandomCrop(32, padding=4),
                transforms.RandomResizedCrop(32),#, scale=(0.5,1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])  
                
        self.len = 48384
        
    def __getitem__(self, index):
        img = make_shape()

        img1 = self.transform(img)
        img2 = self.transform(img)
        
        return img1, img2
        
    def __len__(self):
        return self.len
    
if __name__ == "__main__":    
    import matplotlib.pyplot as plt
    
    a = random_set()
    
    b, c = a.__getitem__(0)
    
    b[0] = b[0]*0.5 + 0.5
    b[1] = b[1]*0.5 + 0.5
    b[2] = b[2]*0.5 + 0.5
       
    b = b.numpy().transpose(1,2,0)
    
    plt.imshow(b)
    plt.show()
    plt.close()
    
    c[0] = c[0]*0.5 + 0.5
    c[1] = c[1]*0.5 + 0.5
    c[2] = c[2]*0.5 + 0.5
       
    c = c.numpy().transpose(1,2,0)
    
    plt.imshow(c)
    plt.show()
    plt.close()
    '''
    d = cifar_part()
    
    b, c = d.__getitem__(np.random.choice(500,1)[0])
    
    b[0] = b[0]*0.5 + 0.5
    b[1] = b[1]*0.5 + 0.5
    b[2] = b[2]*0.5 + 0.5
       
    b = b.numpy().transpose(1,2,0)
    
    plt.imshow(b)
    plt.show()
    plt.close()
    '''