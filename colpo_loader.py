from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random
import json
from PIL import Image
import torchvision.transforms as transforms
from random import shuffle
import time
from os.path import join

def seed_worker(worker_id):
    # for reproducibility
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def setup_loader(image_path, dataset_path, class_count, batch_size=64, train=False, seed=None, pos_class=None):
    """
        Set up the dataloader
        :param image_path: folder with images
        :param dataset_path: text file with data labels 
        :param batch_size: batch size
        :param train: True for training, else False
        :param pos_class: positive classes based on label index  
        :param seed: seed init
        :return: returns a torch dataloader
    """

    dataset = ColpoDataset(train, image_path, dataset_path, class_count, pos_class=pos_class)
    if seed is None:
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                pin_memory=True,
                                shuffle=train,
                                drop_last=False)
    else:
        g = torch.Generator()
        g.manual_seed(seed)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            worker_init_fn=seed_worker,
            generator=g,
        )

    return dataloader

class ColpoDataset(Dataset):
    def __init__(self, train, image_path, dataset_path, class_count, pos_class=None):
        """
            :param train: True for training, else False
            :param image_path: folder with images
            :param dataset_path: text file with data labels 
            :param pos_class: positive classes based on label index  
            :return: nothing
        """
        super(ColpoDataset, self).__init__()
        
        self.classes = ['CIN1', 'CIN23', 'cancer']
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
                if class_count == 1:
                    imagelabel = 1 if 1 in [imagelabel[pos] for pos in pos_class] else 0

                self.listimgpaths.append(imagepath)
                self.listimglabels.append(imagelabel)

    def __len__(self):
        return len(self.listimgpaths)

    def get_labels(self):
        return self.listimglabels

    def get_pos_weight(self):
        labeled = sum(self.listimglabels)
        unlabeled = len(self.listimglabels) - labeled
        return torch.tensor(unlabeled/labeled)

    def expand2square(self, pil_img, background_color):
        # for square images while maitaining the ratio
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
        """
            :return: image index, transformed image, labels, image filename
        """
        image_fn = join(self.image_path, self.listimgpaths[idx])
        image = Image.open(image_fn)
        image = self.augmenter(image, self.train)    
        return idx, image, torch.FloatTensor(self.listimglabels[idx]), image_fn

    def augmenter(self, image, train=False):
        try:
            if train:
                transform = transforms.Compose([transforms.RandomRotation(degrees=5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        # transforms.ColorJitter(brightness=5, contrast=5, saturation=5, hue=5),
                                        transforms.Resize((288, 288))])
            else:
                transform = transforms.Compose([transforms.Resize((288, 288))])

            normalize = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])
            return normalize(transform(image))
        except:
            print("Augmentation Error")
        
        return None
