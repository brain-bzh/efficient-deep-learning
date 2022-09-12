Lab Session 2
--
The objectives of this second lab session are the following:
- Quantize a neural network post-training
- Quantize during training using Binary Connect
- Explore the influence of quantization on performance on a modern DL architecture

We will perform experiments on CIFAR10  datasets. 


Prologue - How to reload your previous models
--
Pytorch has a [page](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference) explaining how to save and load models. But here are a few additional details. 

Most probably, in the last session you explored various architecture hyperparameters. In order to load a model, you need to define the model in the same way that it was defined when training it. 

Let's assume your model definition during training had a single hyperparameter that you explored with various values.

        model = MySuperModel(hyperparam = hparam_currentvalue)

The following code enables you to save the currently trained model (his parameters, it is called a *state dictionary* in pytorch) as well as the current value `hparam_currentvalue` of the hyperparameter.

        state = {
                'net': model.state_dict(),
                'hyperparam': hparam_currentvalue
        }

        torch.save(state, 'mybestmodel.pth')

In order to reload this model, first we need to define it. This means we need to fetch the value of the hyperparameter before defining the model and loading the trained parameters. 

        # We load the dictionnary
        loaded_cpt = torch.load('mybestmodel.pth')

        # Fetch the hyperparam value
        hparam_bestvalue = loaded_cpt['hyperparam']

        # Define the model 
        model = MySuperModel(hyperparam = hparam_bestvalue)

        # Finally we can load the state_dict in order to load the trained parameters 
        model.load_state_dict(loaded_cpt['net'])

        # If you use this model for inference (= no further training), you need to set it into eval mode
        model.eval()



Part 1 - Quantization to half and integer precision
--
The goal of this part is to work with one of the models you obtained in Lab Session 1, reload the weights and quantize after training. 

While converting a model post-training to floating point half-precision is straightforward, converting to integer is more complex because all operators need to be adapted, and dynamic ranges in computations differ (see the [course](course2.pdf))

PyTorch has recently introduced a set of tools for quantization to 8bit integer, using specific quantized tensor types, quantized version of operators, as well as utily functions to manage quantization. Please read the following post for a general explanation of [Pytorch quantization features](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/) for quantization. 

While 8-bit quantization in pytorch is still experimental (the features are not very well documented), quantization to 16 bits is already working. 
You can convert models and tensors to half, by simply doing `.half()`. 

Use the following syntax (this can be done either before or after training): 
    
        model.half()  # convert all the model parameters to 16 bits half precision
and in order to perform inference, you should also convert your inputs to half.

What is the impact of post-training quantization on performance of your model(s)  ? 

Part 2 - Quantization to binary
--

Now we would like to go even further in quantization, by considering only 1 bit for representing weights. 

In this part we will work with [Binary Connect](http://papers.nips.cc/paper/5647-binaryconnect-training-deep-neural-networks-with-b) in pytorch. We presented this method in [the course](cours2.pdf)

Implementing the method from a blank page would probably be too long for this lab session, so will give you the a model with a few empty code blocks to be completed. 

Use the starting point in the file [binaryconnect.py](binaryconnect.py), as well as the paper, to implement binaryconnect. We ask you to implement the *deterministic* version of quantification. 

More details:

We propose an implementation that will use a class `binaryconnect.BC()` that supersets the model that will be binarized. 

Before explaining how this class works internally, here are a few examples on how to use the class :  

        import binaryconnect
        ### define your model somewhere, let's say it is called "mymodel"

        mymodelbc = binaryconnect.BC(mymodel) ### use this to prepare your model for binarization 

        mymodelbc.model = mymodelbc.model.to(device) # it has to be set for GPU training 

        ### During training (check the algorithm in the course and in the paper to see the exact sequence of operations)

        mymodelbc.binarization() ## This binarizes all weights in the model

        mymodelbc.restore() ###  This reloads the full precision weights

        ### After backprop

        mymodelbc.clip() ## Clip the weights 
With this information it should be rather clear how to implement the training loop for the Binary Connect algorithm. 

Start by having a look at this [Notebook](Reading_copying_modifying_weights.ipynb), which will show you how to read, copy and write the weights in a pytorch model. 

Now, have a look at the [binaryconnect.py](binaryconnect.py) file to see how the `binaryconnect.BC()` class is implemented. 

- The class defines several internal structures to save the parameters in full precision in `self.save_params`, 
- The class uses the `self.target_modules` list to store the modules that contain parameters. The items in this list are the module weights, and the corresponding values can be written by used the `.copy_` method of the `.data` property. See the implementation of `self.save_params` function as well as the initialization of the class. 
- The `self.binarization()` is supposed to first save the full precision weigths by calling `self.save_params`, read the list of target modules, and write a binarized version of the weights in the model using the `self.target_modules` list. 
- `self.restore()` restores the full precision version by reading them in `self.saved_params` and writing back in the model, again using the `self.target_modules` list. 

Part 3 - CIFAR10 and CIFAR100
--

Now is the time to start working properly on some real challenging dataset ! The final goal of the project for this course is to explore how to reduce the number of computations and memory requirements for performing inference on CIFAR10 and CIFAR100. So far, we have seen how to explore hyperparameters (in session 1) and how to consider quantization (this session). Same question than for the same presentation : can you explore these concepts with a chosen deep learning architecture, and explore the accuracy / architecture size tradeoff ? 

A few starting points : 
- Take some time to check out the [state of the art](paperswithcode.com) results that can be obtained on CIFAR10 and CIFAR100. Which architecture did they use ? Did they use any additional data apart from the dataset ? 
- Experiment with various modern DL networks with quantization while training, using either binaryconnect, or another method such as BWN, XNorNet, while training on CIFAR10
- Have a look at the winners of the last MicroNet competition. We have not yet introduced all the concepts in the course, but it is still interesting that you begin to familiarize yourself with this litterature. 

Spend some time reading while you launch some large scale training on CIFAR10 and CIFAR100. You will be evaluated at Session 3 and 5 on the work you did so far. 
