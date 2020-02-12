Lab Session 2
--
The objectives of this second lab session are the following:
- Quantize a neural network post-training
- Quantize during training using Binary Connect
- Explore the influence of quantization on performance on a modern DL architecture

We will perform all experiments on the MINICIFAR dataset. 

Part 1
--
Choose one of the models you obtained in Lab Session 1, reload the weights and quantize after training. 

There are several ways to test this : 
- use the following syntax (this can be done either before or after training): 
    
        model.half()  # convert to half precision
- Directly access the weights and overwrite them with a quantized version. For this, you can checkout the [binaryconnect.py](binaryconnect.py) file that we provide.

- Here is a more advanced [pytorch tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#post-training-static-quantization) for quantization. 

What is the impact of post-training quantization on performance of your model(s) on MiniCIFAR ? 

Part 2 
--

In this part we will work with [Binary Connect](http://papers.nips.cc/paper/5647-binaryconnect-training-deep-neural-networks-with-b) in pytorch. 

Implementing the method from a blank page would probably be too long for this lab session, so will give you the a model with a few empty code blocks to be completed. 

Use the starting point in the file [binaryconnect.py](binaryconnect.py), as well as the paper, to implement binaryconnect. We ask you to implement the *deterministic* version of quantification. You have to complete the class BC(model), which uses a model definition, and can subsequently be used to train a model using binaryconnect. Be careful that you need to follow  the algorithm from the paper in the training loop. 

    
Part 3 - CIFAR10 and CIFAR100
--

Now is the time to start working properly on the MicroNet challenge ! A few starting points : 
- Take some time to check out the [state of the art](paperswithcode.com) results that can be obtained on CIFAR10 and CIFAR100. Which architecture did they use ? Did they use any additional data apart from the dataset ? 
- Experiment with various modern DL networks with quantization while training, using either binaryconnect, or another method such as BWN, XNorNet, while training on CIFAR10
- Have a look at the winners of the last MicroNet competition. We have not yet introduced all the concepts in the course, but it is still interesting that you begin to familiarize yourself with this litterature. 

Spend some time reading while you launch some large scale training on CIFAR10 and CIFAR100. You will be evaluated at Session 4 on the work you did so far on the challenge. 