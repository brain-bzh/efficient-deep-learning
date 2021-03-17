
# Long Project Scoring Rules 

### (from [micronet challenge 2019](https://micronet-challenge.github.io/))

-------------------------------------------------------------

 The aim of the Long project is to design a model that **minimizes** a global score while keeping the accuracy higher than a predefined threshold. 


The global score takes in account **parameter storage** and **math operations** with respect to a state-of-the-art baseline model.


**ResNet18** and **WideResNet** will be considered as baseline models for, respectively, the CIFAR10 and CIFAR100 classification tasks.


Threshold accuracies are of **90% for CIFAR10** and **80% for CIFAR100**: models whose performances are below the theshold accuracy won't be taken in account for the *challenges*.

The [profile.py](https://github.com/brain-bzh/ai-optim/blob/master/micronet-ressources/profile.py) we provided is a good example of how to calculate the score. It follows the micronet challenges rules for scoring that take in account sparsity and quantization. However, depending on your implementation, you may need to slightly modify profile.py to adapt it to your model, following the rules listed below.

## Parameter storage
### *Number of model parameters required to perform inference.*

- If you performed (unstructured) pruning, you should modify the parameters count in profile.py to disregard zero weights, as only non-zero parameters should be taken into account.

- If you performed quantization, remember that the final parameter storage is equal to 

```
Par_Storage = N_par * Par_Quant_Ratio 
```
32-bit parameter counts as Par_Quant_Ratio=1. Quantized parameters of less than 32-bits will be counted as a fraction of one parameter. For example, an 8-bit parameter counts as 1/4th a parameter. In the micronet challenge (and the profile.py) it is assumed that 16-bits quantization is the base one, as it comes for "free" (no accuracy loss).

*N.B. if different format are used in the same model, then you should consider the sum of parameters storage indexes for the different quantization formats.*

## Math operations

### *Mean number of arithmetic operations required to perform inference.*

- If you performed structured pruning, you should modify the operation count in profile.py to disregard entirely pruned filters from the operations count.

- As for the parameter storage index, if quantization is performed, an operation on data of less than 32-bits will be counted as a fraction of one operation, where the numerator is the maximum number of bits in the inputs of the operation and the denominator is 32. For example, a multiplication operation with one 3-bit and one 5-bit input, with a 7-bit output, will count as 5/32nd of an operation. The score will than be equal to 

```
Math_Ops = N_ops * Ops_Quant_Ratio = N_ops * M/32
```

where the numerator M is the maximum number of bits in the inputs of the operation and the denominator is 32.

## Global Score


```
Final_Score = Par_Storage/Baseline_Par + Math_Ops/Baseline_Ops
```
