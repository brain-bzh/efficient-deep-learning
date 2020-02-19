Repository for the course "Optimizing Artificial Intelligence" at IMT Atlantique
--


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
   5. Long project 1 - MicroNet Challenge
3. Pruning
   1. **Short evaluation** on Quantification
   2. [Course](cours3.pdf) - Pruning Deep neural networks
   3. [Practical session](lab3.md) - pruning on a small convolutional network.
   4. Long project 2 - MicroNet Challenge
4. Factorization
   1. **Short evaluation** on Pruning
   2. **Student's presentation** on current work on MicroNet
   3. Course - Factorizing Deep neural networks
   4. Practical session - factorizing a small convolutional network
   5. Long Project 3 - MicroNet Challenge
5. Distillation
   1. **Short evaluation** on Factorization
   2. Course - Distillation of knowledge and features between neural networks
   3. Practical session - a simple distillation case 
   4. Long Project 4 - MicroNet Challenge 
6. Operators and Architectures
   1. **Short evaluation** on Distillation
   2. **Student's presentation** on current work on MicroNet
   3. Course - Alternative operators and efficient architectures
   4. Practical session - defining and training alternative operators
   5. Long Project 5 - MicroNet Challenge
7. Embedded Software and Hardware for Deep Learning 
   1. **Short evaluation** on operators and architectures
   2. Course - Embedded Software and Hardware for Deep Learning
   3. Long Project 6 - MicroNet Challenge
8. Final Session
   1. **Short evaluation** on embedded software and hardware for Deep Learning
   2. Long Project 7 - MicroNet Challenge
   3. **Student's presentation** - Final results on MicroNet



Evaluation in this course 
--

There are **short written evaluations** during the first 10 minutes of **each** session starting from session 2. Don't be late!  
BNN+For the final session, we ask you to prepare a 20 minutes presentation, that will be followed by 10 Minutes of question. 

You'll find in the micronet-ressources folder, presentations from the winners of the 2019, and rules for the 2020 challenge. 

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

Factorization
-- 


Operators
--


Distillation
--
Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).

Romero, Adriana, et al. "Fitnets: Hints for thin deep nets." arXiv preprint arXiv:1412.6550 (2014).

Koratana, Animesh, et al. "LIT: Learned Intermediate Representation Training for Model Compression." International Conference on Machine Learning. 2019.

Heo, Byeongho, et al. "A Comprehensive Overhaul of Feature Distillation." arXiv preprint arXiv:1904.01866 (2019).

Stock, Pierre, et al. "And the bit goes down: Revisiting the quantization of neural networks." arXiv preprint arXiv:1907.05686 (2019).

Embedded Software and Hardware
--

See references section of [Tutorial presentation on Efficient Deep Learning from NeurIPS'19](http://eyeriss.mit.edu/2019_neurips_tutorial.pdf).


Companies / private sector
-- 

[13 highest funded startups for hardware for DL](https://www.crunchbase.com/lists/relevant-ai-chip-startups/922b3cf5-b19d-4c28-9978-4e66ccb52337/organization.companies)

[More complete list of companies working on hardware DL](https://roboticsandautomationnews.com/2019/05/24/top-25-ai-chip-companies-a-macro-step-change-on-the-micro-scale/22704/)

