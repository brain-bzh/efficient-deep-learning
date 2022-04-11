# Lab Session 1 

The objectives of this first lab session are the following:
- Familiarize yourself with pytorch
- Train a model from scratch using a state of the art architecture
- Train a classifier using Transfer Learning from a pretrained model
- Explore hyperparameters of a given architecture

We will perform all experiments on the CIFAR10 dataset. 

---
## Part 1

Familiarize yourself with pytorch by doing the [Pytorch_tutorial.ipynb](Pytorch_tutorial.ipynb), a jupyter notebook that you will be able to run also in VScode, after installing the Jupyter extension (remember to "Trust" the notebook, as explained [here](https://code.visualstudio.com/docs/python/jupyter-support)). If you are familiar with pytorch, you can go quickly over the tutorial and try to train a classifier in section 4 where you are asked to complete some cells.

---
## Part 2 

The following code can be used to obtain a DataLoader for CIFAR10, ready for training in pytorch : 

```python
from torchvision.datasets import CIFAR10
import numpy as np 
import torchvision.transforms as transforms
import torch 
from torch.utils.data.dataloader import DataLoader

## Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network. 
# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
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

### The data from CIFAR10 will be downloaded in the following folder
rootdir = './data/cifar10'

c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)

trainloader = DataLoader(c10train,batch_size=32,shuffle=True)
testloader = DataLoader(c10test,batch_size=32) 
```

However, this will load the entire CIFAR10 dataset, which has 50000 examples per class for training ; this can result in a relatively long training. As a consequence, we encourage you to use the following code, with a [RandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.RandomSampler) in order to use a subset of training : 


```python
## number of target samples for the final dataset
num_train_examples = len(c10train)
num_samples_subset = 15000

## We set a seed manually so as to reproduce the results easily
seed  = 2147483647

## Generate a list of shuffled indices ; with the fixed seed, the permutation will always be the same, for reproducibility
indices = list(range(num_train_examples))
np.random.RandomState(seed=seed).shuffle(indices)## modifies the list in place

## We define the Subset using the generated indices 
c10train_subset = torch.utils.data.Subset(c10train,indices[:num_samples_subset])
print(f"Initial CIFAR10 dataset has {len(c10train)} samples")
print(f"Subset of CIFAR10 dataset has {len(c10train_subset)} samples")

# Finally we can define anoter dataloader for the training data
trainloader_subset = DataLoader(c10train_subset,batch_size=32,shuffle=True)
### You can now use either trainloader (full CIFAR10) or trainloader_subset (subset of CIFAR10) to train your networks.
```

We will now define a state of the art deep model and train it from scratch. Check out [here](https://github.com/kuangliu/pytorch-cifar/tree/master/models) for reference implementations of modern deep models for CIFAR10. 

### TASK 1. Train a model from scratch

Choose a model among the following ones : 
- ResNet
- PreActResNet
- DenseNet
- VGG
  
Next, train it on a subset of CIFAR10. Try to compare with the performances on the full CIFAR10 [reported here](https://github.com/kuangliu/pytorch-cifar). 

A few hints : 
- Learning rate is a very important (if not the most important) hyperparameter, and is routinely scheduled to change a few times during training. A typical strategy is to divide it by 10 when reaching a plateau in performance. 
- Be careful with overfitting, which happens when the gap between Train and Test accuracy keeps getting larger. 
- Think about plotting and saving your results, so as not to lose track of your experiments. 


### TASK 2. Figure Accuracy vs Number of Parameters
Consider the four models of TASK 1. and, taking in account the [accuracy obtained on CIFAR10](https://github.com/kuangliu/pytorch-cifar), generate a graph accuracy vs number of parameters such as this one

![Image](accuracy_vs_parameters_imagenet.png)

---
## Part 3: TRANSFER LEARNING

For Transfer Learning, we have pretrained the models 

- ResNet
- PreActResNet
- DenseNet
- VGG

of the [repository](https://github.com/kuangliu/pytorch-cifar) using a larger dataset [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) . CIFAR-100 is just like the CIFAR-10, except it has 100 classes containing 600 images each. To reduce the overlap between the lables from CIFAR-10 and CIFAR-100, we ignored super-classes of CIFAR-100 that are conceptually similar to CIFAR-10 classes. 

You can load the models from the folder models_cifar100, for example using

```python
import models_cifar100
backbone=models_cifar100.ResNet18()
```

then download the pretrained model weights  at this [link](https://partage.imt.fr/index.php/s/o4ZzekyBHjgx4iS) and load them 

```python
if torch.cuda.is_available():
    state_dict=torch.load(backbone_weights_path)
else:
    state_dict=torch.load(backbone_weights_path,map_location=torch.device('cpu'))

backbone.load_state_dict(state_dict['net'])

```

The backbone can be used to generate feature vectors, which will serve as input to a classifier (e.g. last fully connected layer) to perform classification on CIFAR10. One way to do this is presented in [this tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#convnet-as-fixed-feature-extractor), but you can also easily find other ones. For instance, after having trained only the classifier (transfer learning) you may want to **fine-tune for a few epochs the whole model** to improve performances.

---

## Part 4 - Project

Prepare a presentation (10 minutes + 5 minutes question) with the following content : 
- Description of the chosen architecture
- Hyperparameter exploration strategy 
- Results on CIFAR10 subset, focusing on illustrating the **compromises between model size, training time and performance**

If you are ahead in time, you can perform similar experiments on the full CIFAR-10 and CIFAR-100, but be aware that each run (e.g. 350 epochs) might take as long as 3 hours.

**N.B. It is very important that you consider the models in the kuangliu repository, as they have been dimensioned for the CIFAR-10 dataset. Respective models taken from other sources may be have been optimized for other datasets and therefore not adapted (over or underparametrized) to CIFAR-10.**