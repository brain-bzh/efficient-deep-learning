# Lab Session 1 

**Made by:**

- Zijie NING @[mm0806son](https://github.com/mm0806son)
- Guoxiong SUN @[GuoxiongSUN](https://github.com/GuoxiongSUN)

-----

### TASK 1. Train a model from scratch

Choose a model among the following ones : 

- ResNet
- PreActResNet
- DenseNet
- VGG

Next, adapt its hyperparameters to make the model suitable for MINICIFAR, and train it from scratch. 

**Hyperparameters to modify:**

- Network

  - num of hidden layers
  - num of hidden layer units
  - activation function

- Optimization

  - learning rate
  - n_epochs
  - batch-size
  - optimizer(SGD, Adam, ...)
  - transform

- Regularization(**deal with overfitting**)

  - weight_decay

  - dropout


### TASK 2. Figure Accuracy vs Number of Parameters
Consider the four models of TASK 1. and, taking in account the [accuracy obtained on CIFAR10](https://github.com/kuangliu/pytorch-cifar), generate a graph accuracy vs number of parameters.

[image]



## Part 3: TRANSFER LEARNING

For Transfer Learning, we have pretrained the models adapted to CIFAR 10

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

The backbone can be used to generate feature vectors, which will serve as input to a classifier (e.g. last fully connected layer) to perform classification on MINICIFAR. One way to do this is presented in [this tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#convnet-as-fixed-feature-extractor), but you can also easily find other ones. For instance, after having trained only the classifier (transfer learning) you may want to **fine-tune for a few epochs the whole model** to improve performances.

---

## Part 4 - Project

Prepare a presentation (10 minutes + 5 minutes question) with the following content : 
- Description of the chosen architecture
- Hyperparameter exploration strategy 
- Results on MINICIFAR, focusing on illustrating the **compromises between model size, training time and performance**

