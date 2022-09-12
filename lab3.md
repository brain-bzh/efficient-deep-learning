Lab Session 3
--
The objectives of this third lab session is to perform experiments using pruning methods.

Part 1
--
Pytorch provides a library designed to ease the pruning of neural networks : [Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html).
Pay attention to the difference between the pruning functions (like [`prune.random_unstructured`](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.random_unstructured.html#torch-nn-utils-prune-random-unstructured)) and the [`prune.remove`](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.remove.html#torch-nn-utils-prune-remove) function.

For example, when it is applied to weights, applying pruning functions on a module create a duplicate of the original weights (`weight_orig`) and a related mask (`weight_mask`). It changes the structure of the module.

`weight_orig` becomes a [torch.nn.parameter.Parameter](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#parameter), and therefore if you train the model, it's this tensor that will be modified. `weight` is a simple attribute that is computed during the forward pass, by applying `weight_mask` on `weight_orig`.

If you want to permanently apply the pruning and get back to the original structure of your model, you have to apply [`prune.remove`](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.remove.html#torch-nn-utils-prune-remove) on the module. It will recreate `weight` as a parameter, with the content of `weight_orig` for unpruned weights, and 0s on pruned weights.

The example from [Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) considers a very simple network. Yours will be more complex. A first step should be to extract the modules to be pruned in order to prun them. Iterate over (torch.nn.Module.modules)[https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.modules] to extract the conv and linear layers. Then apply pruning.

The goal of today's session is to apply this previous knowledge in order to implement a pruning method. You can choose any of the methods that we studied in [course3](cours3.pdf), but probably the following four are the most straightforward to implement :
1. Global Pruning, no retrain : simply remove weights with lowest l1 norm, measure accuracy for different pruning ratios
2. [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626) :  apply a retrain after the first global pruning
3. [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) : gradually prune and retrain accross layers
4. [ThiNet](https://arxiv.org/abs/1707.06342): same, but based on the norms of feature maps

There are several ways to prune, be innovative ! Different ratios, on different layers, different pruning criteria, diffrent ways of finetuning... Play !

Part 2 - Combining all techniques on CIFAR10 and CIFAR100
--
Now, it's your turn to combine everything we have seen so far to start performing some interesting comparisons using the three datasets CIFAR10 and CIFAR100.

Consider the different factors that can influence the total memory footprint needed to store the network parameters as well as feature maps / activations.

The key question we are interested in :

**What is the best achievable accuracy with the smallest memory footprint ?**

Prepare a presentation for session 5, detailing your methodology and explorations to adress this question. You will have 10 minutes to present, followed by 5 minutes of questions. Good luck !

