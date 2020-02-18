Lab Session 3
--
The objectives of this third lab session is to perform experiments using pruning methods. 

Part 1
--
Using the work performed in [lab session 2](lab2.md), you should know enough about pytorch to know how to : 
- access the weights or feature maps in a pytorch model 
- perform operations on weights / feature maps
- modify weights accordingly

The goal of today's session is to apply this previous knowledge in order to implement a pruning method. You can choose any of the methods that we studied in [course3](cours3.pdf), but probably the following two are the most straightforward to implement : 
- [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)
- [ThiNet](https://arxiv.org/abs/1707.06342)


Part 2 - Combining all techniques on MiniCIFAR, CIFAR10 and CIFAR100
--
Now, it's your turn to combine everything we have seen so far to start performing some interesting comparisons using the three datasets MiniCIFAR, CIFAR10 and CIFAR100. 

Consider the different factors that can influence the total memory footprint needed to store the network parameters as well as feature maps / activations. 

The key question we are interested in : 

**What is the best achievable accuracy with the smallest memory footprint ?**

Prepare a presentation for session 4, detailing your methodology and explorations to adress this question. You will have 10 minutes to present, followed by 5 minutes of questions. Good luck ! 

This work is part of your work for the MicroNet challenge, as long as you apply it on CIFAR100. 