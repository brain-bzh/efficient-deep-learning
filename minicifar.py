#!/usr/bin/env python
# coding: utf-8

# This script generates the MINICIFAR dataset from CIFAR10
# The following parameters can be changed : 
# n_classes (between 2 and 10) 
# Reduction factor R (which will result in 10000 /  R examples per class for the train set, and 1000 / R per class for test)
# --


n_classes_minicifar = 4
R = 5


# Download the entire CIFAR10 dataset

from torchvision.datasets import CIFAR10
import numpy as np 
from torch.utils.data import Subset

import torchvision.transforms as transforms

## Normalization is different when training from scratch and when training using an imagenet pretrained backbone

normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

normalize_forimagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

# Data augmentation is needed in order to train from scratch
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_scratch,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])

## No data augmentation when using Transfer Learning
transform_train_imagenet = transforms.Compose([
    transforms.ToTensor(),
    normalize_forimagenet,
])

transform_test_imagenet = transforms.Compose([
    transforms.ToTensor(),
    normalize_forimagenet,
])


### The data from CIFAR10 will be downloaded in the following dataset
rootdir = './data/cifar10'

c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)

c10train_imagenet = CIFAR10(rootdir,train=True,download=True,transform=transform_train_imagenet)
c10test_imagenet = CIFAR10(rootdir,train=False,download=True,transform=transform_test_imagenet)

# Generating Mini-CIFAR
# 
# CIFAR10 is sufficiently large so that training a model up to the state of the art performance will take approximately 3 hours on the 1060 GPU available on your machine. 
# As a result, we will create a "MiniCifar" dataset, based on CIFAR10, with less classes and exemples. 

def generate_subset(dataset,n_classes,reducefactor,n_ex_class_init):

    nb_examples_per_class = int(np.floor(n_ex_class_init / reducefactor))
    # Generate the indices. They are the same for each class, could easily be modified to have different ones. But be careful to keep the random seed! 

    indices_split = np.random.RandomState(seed=42).choice(n_ex_class_init,nb_examples_per_class,replace=False)


    all_indices = []
    for curclas in range(n_classes):
        curtargets = np.where(np.array(dataset.targets) == curclas)
        indices_curclas = curtargets[0]
        indices_subset = indices_curclas[indices_split]
        #print(len(indices_subset))
        all_indices.append(indices_subset)
    all_indices = np.hstack(all_indices)
    
    return Subset(dataset,indices=all_indices)
    


### These dataloader are ready to be used to train for scratch 
minicifar_train= generate_subset(dataset=c10train,n_classes=n_classes_minicifar,reducefactor=R,n_ex_class_init=5000)
minicifar_val= generate_subset(dataset=c10test,n_classes=n_classes_minicifar,reducefactor=R,n_ex_class_init=1000) 
minicifar_test= generate_subset(dataset=c10test,n_classes=n_classes_minicifar,reducefactor=1,n_ex_class_init=1000) 


### These dataloader are ready to be used to train using Transfer Learning 
### from a backbone pretrained on ImageNet
minicifar_train_im= generate_subset(dataset=c10train_imagenet,n_classes=n_classes_minicifar,reducefactor=R,n_ex_class_init=5000)
minicifar_val_im= generate_subset(dataset=c10test_imagenet,n_classes=n_classes_minicifar,reducefactor=R,n_ex_class_init=1000)
minicifar_test_im= generate_subset(dataset=c10test_imagenet,n_classes=n_classes_minicifar,reducefactor=1,n_ex_class_init=1000)

