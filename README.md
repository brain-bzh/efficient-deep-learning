Repository for the course "Optimizing Artificial Intelligence" at IMT Atlantique
--
Getting started
--
Please see [here](getting_started.md) for instructions on how to setup your environment on your personal computer.

Course organisation / Syllabus
--

Here is a detailed schedule, session by session: 
1. Introduction / Refresher on Deep Learning 
   1. [General Intro](intro.pdf) - Why do we need to optimize deep learning ? Introduction of the MicroNet Challenge. 
   2. [Course](cours1.pdf) - Deep Learning and Transfer Learning.
   3. [Practical session](lab1.md) - introduction to PyTorch, transfer learning. 
   4. Short project - exploring hyper parameters on a fixed architecture
2. Quantification
   1. **Short evaluation** on Deep Learning Essentials
   2. **Student's presentation** of short project - exploring hyper parameters on a fixed architecture
   3. [Course](cours2.pdf) - Quantifying Deep neural networks
   4. [Practical session](lab2.md) - quantification on a small convolutional network 
   5. Long project
3. Pruning
   1. **Short evaluation** on Quantification
   2. [Course](cours3.pdf) - Pruning Deep neural networks
   3. [Practical session](lab3.md) - pruning on a small convolutional network.
   4. Long project
4. Factorization
   1. **Short evaluation** on Pruning
   2. **Student's presentation** on current work on Long Project
   3. [Course](cours4.pdf) - Factorizing Deep neural networks
   4. Practical session - factorizing a small convolutional network
   5. Long Project
5. Factorization - Part 2 - Operators and Architectures
   1. [Course](cours5.pdf) - Factorization Pt2, alternative operators and efficient architectures
   2. Long Project
6. Distillation
   1. **Short evaluation** on Factorization Pt1 and Pt2 and previous courses
   2. [Course](cours6.pdf) - Distillation of knowledge and features between neural networks
   3. Long Project

7. Embedded Software and Hardware for Deep Learning 
   1. **Short evaluation** on Distillation
   2. Course - Embedded Software and Hardware for Deep Learning
   3. Long Project
8. Final Session
   1. **Short evaluation** on embedded software and hardware for Deep Learning
   2. Long Project
   3. **Student's presentation** - Final presentation on Long Project

What is expected for the Long Project
--
Short version : **Exploration of the accuracy / complexity tradeoff**

Long version : this course is mostly based on the long project, and you have a lot of freedom, which we expect you to use. The overarching goal is to explore the tradeoff between the performances of neural networks (= Accuracy on the test set) and complexity. Complexity can be either computational complexity (number of arithmetic operations), or memory complexity (memory size necessary to use the network). 

We encourage students to get creative and test combinations of the various ideas that we present. Starting from the end of Session 1, you already have enough knowledge to explore the tradeoff between architecture, number of parameters, and accuracy. Then, we study new notions that open new avenues to explore this tradeoff : quantization, pruning, factorization, distillation. In session 7, you'll have a deeper insight on how to thing about specific software or hardware architecture in order to fully exploit all the optimizations that can be done. 

Evaluation in this course 
--

There are **short written evaluations** during the first 10 minutes of **each** session starting from session 2. Don't be late!  

**For the final session**, we ask you to prepare a **20 minutes presentation**, that will be followed by 10 Minutes of question. 

What we expect for the presentations : 
1. Explain your strategy to explore the complexity / accuracy tradeoff. We will judge whether you understood the goal, and whether the proposed strategy follows a rigourous approach.  
2. The clarity of your exposition and quality of your support (slides)

Importantly : We will NOT judge you based on accuracy scores, as long as you have baselines (non optimized) that are not too far from the usual accuracy for the considered dataset. 

General References
--

[List of references IMT Atlantique and AI](https://docs.google.com/document/d/1-IX-IO8DXYOZSiihOe0ttjvJvcEO9WLU2UtZgej86gQ/edit#heading=h.iueps2uhjocc)

Amazon Book - [Dive into Deep learning](https://d2l.ai/)

[Tutorial presentation on Efficient Deep Learning from NeurIPS'19](http://eyeriss.mit.edu/2019_neurips_tutorial.pdf)


Training Deep Networks
--

Here are some academic papers discussing learning rate strategies : 

- [Cyclic learning rates](https://arxiv.org/abs/1506.01186)
- [Demystifying Learning Rate Policies for High Accuracy Training of Deep Neural Networks](https://arxiv.org/abs/1908.06477)
- [A Closer Look at Deep Learning Heuristics: Learning rate restarts, Warmup and Distillation](https://arxiv.org/abs/1810.13243)

Main strategies are [readily available in pytorch.](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

Pytorch
--

Start page to access the full [python API](https://pytorch.org/docs/stable/torch.html) of pytorch, to check all existing functions.

[A useful tutorial on Saving and Loading models](https://pytorch.org/tutorials/beginner/saving_loading_models.html).

[Pytorch Cheat Sheet](https://pytorch.org/tutorials/beginner/ptcheat.html).

Data Augmentation
--
Popular methods : 

[Cut Out](https://github.com/uoguelph-mlrg/Cutout)

[Auto Augment](https://github.com/DeepVoltaire/AutoAugment)

Other ressources : 

[A list of papers and code for data augmentation](https://github.com/CrazyVertigo/awesome-data-augmentation)

[IMGAUG](https://imgaug.readthedocs.io/en/latest/index.html) and [Colab Notebook showing how to use IMGAUG with pytorch](https://colab.research.google.com/drive/109vu3F1LTzD1gdVV6cho9fKGx7lzbFll)

A popular python package in Kaggle competitions : [Albumentations](https://github.com/albumentations-team/albumentations)

Quantization
--
[Binary Connect](http://papers.nips.cc/paper/5647-binaryconnect-training-deep-neural-networks-with-b)

[XnorNet](https://link.springer.com/chapter/10.1007/978-3-319-46493-0_32)

[BNN+](https://openreview.net/forum?id=SJfHg2A5tQ)

[Whitepaper of quantization](https://arxiv.org/abs/1806.08342)


Pruning
--
[Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)

[ThiNet](https://arxiv.org/abs/1707.06342)


[AutoML for Model Compression (AMC)](https://arxiv.org/abs/1802.03494)

[Pruning Channel with Attention Statistics (PCAS)](https://arxiv.org/abs/1806.05382)

[BitPruning: Learning Bitlengths for Aggressive and Accurate Quantization](https://arxiv.org/abs/2002.03090)

Factorization and operators
-- 

[Deep Compression](https://arxiv.org/abs/1510.00149)

[Deep K-means](https://arxiv.org/abs/1806.09228)

[SqueezeNet](https://arxiv.org/abs/1602.07360)

[MobileNet](https://arxiv.org/abs/1704.04861)

[MobileNetV2](https://arxiv.org/abs/1801.04381)

[Shift Attention Layers](https://arxiv.org/abs/1905.12300)

Distillation
--
[Distilling the knowledge in a neural network](https://arxiv.org/abs/1503.02531)

[Fitnets: Hints for thin deep nets](https://arxiv.org/abs/1412.6550)

[LIT: Learned Intermediate Representation Training for Model Compression](http://proceedings.mlr.press/v97/koratana19a.html)

[A Comprehensive Overhaul of Feature Distillation](https://arxiv.org/abs/1904.01866)

[And the bit goes down: Revisiting the quantization of neural networks](https://arxiv.org/abs/1907.05686)


Embedded Software and Hardware
--

See references section of [Tutorial presentation on Efficient Deep Learning from NeurIPS'19](http://eyeriss.mit.edu/2019_neurips_tutorial.pdf).


Companies / private sector
-- 

[13 highest funded startups for hardware for DL](https://www.crunchbase.com/lists/relevant-ai-chip-startups/922b3cf5-b19d-4c28-9978-4e66ccb52337/organization.companies)

[More complete list of companies working on hardware DL](https://roboticsandautomationnews.com/2019/05/24/top-25-ai-chip-companies-a-macro-step-change-on-the-micro-scale/22704/)

