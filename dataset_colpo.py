import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from random import shuffle
import cv2
import time
from os.path import join
from gaussian_blur import GaussianBlur

def augmenter(image, train=False):
    if train:
        transform=transforms.Compose([transforms.RandomRotation(degrees=15),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.Resize((288, 288))])
    else:
        transform=transforms.Compose([transforms.Resize((288, 288))])

    return transform(image)

def process_image(image):    
    means = [0.485, 0.456, 0.406]
    inv_stds = [1/0.229, 1/0.224, 1/0.225]

    image = Image.fromarray(image)
    image = transforms.ToTensor()(image)
    for channel, mean, inv_std in zip(image, means, inv_stds):
        channel.sub_(mean).mul_(inv_std)
    return image

colpo_dict = {0:'CIN1', 1:'CIN23', 2:'cancer'}
categories = ['CIN1', 'CIN23', 'cancer']

class ColpoDataset(Dataset):
    def __init__(self, train, image_path, dataset_path):
        super(ColpoDataset, self).__init__()
        self.classes = categories
        self.image_path = image_path
        self.train = train

        self.listimgpaths = []
        self.listimglabels = []

        with open(dataset_path, "r") as reader:
            content = reader.readlines()
            for line in content:
                items = line.split()
                imagepath = items[0].strip()
                imagelabel = items[1:]
                imagelabel = [int(i) for i in imagelabel]
                self.listimgpaths.append(imagepath)
                self.listimglabels.append(imagelabel)

    def __len__(self):
        return len(self.listimgpaths)

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def __getitem__(self, idx):
        image_fn = join(self.image_path, self.listimgpaths[idx])
        image = Image.open(image_fn)

        #image = self.expand2square(image, (0))
        # if image.mode != 'RGB':
        #     image = image.convert('RGB')
        
        try:
            image = augmenter(image, self.train)
        except IOError:
            print("augmentation error")
            
        transform=transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                           ])
        try:
            image = transform(image)  
        except IOError:
            return None

        return_tuple = (idx, image, self.listimglabels[idx]) if self.train else  (idx, image, self.listimglabels[idx], image_fn)
        
        return return_tuple
