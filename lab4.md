# Lab Session 4

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

Part 2 - Combining all techniques on CIFAR10.
--
Now, it's your turn to combine everything we have seen so far to start performing some interesting comparisons using the datasets CIFAR10. The goal is to design and train a network that **achieves 90% accuracy on CIFAR10**, while having the **lowest possible score**.

$$\text{score} =\underset{param}{\underbrace{\dfrac{[1-(p_s+p_u)]\dfrac{q_w}{32}w}{5.6\cdot10^6}}} + \underset{ops}{\underbrace{\dfrac{(1-p_s)\dfrac{\max(q_w,q_a)}{32}f}{2.8\cdot10^8}}} $$

Where:
- $p_s$: structured pruning
- $p_u$: unstructured pruning
- $q_w$: quantization of weights
- $q_a$: quantization of activations
- $w$: number of weights
- $f$: number of mult-adds (MACs) operations
- $5.6\cdot10^6$ and $2.8\cdot10^8$ are the reference param and ops scores of the ResNet18 network in half precision.

Prepare a presentation for session 5, detailing your methodology and explorations. You will have 7 minutes to present, followed by 3 minutes of questions.

  - Detail your experiments
    - Hyperparameters (learning rate, scheduling, ...)
  - Detail your techniques
    - Architecture search and changes (depth, width, ...), data augmentation, pruning, quantization, ...
  - Calculate the scores of your architectures
  - Summarize your results on a plot with accuracy as a function of the score (with the 90% limit).
