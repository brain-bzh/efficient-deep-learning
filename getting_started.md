## General instructions
In this course, we will use pytorch for Deep Learning. Deep Learning benefits highly from acceleration using a Graphical Processing Unit (GPU), so it is recommended to use the GPU if you have one. If you don't, you should still be able to run basic tests on your machine without GPU, and you will have access to computers with GPUs starting from course 2. 

!! Note that these steps are just recommended guidelines, but there may be a few specifics depending on your system. In particular, if you already have a working python installation, you don't need to install it again.

First, you need to check whether your computer has a NVIDIA GPU or not. If it does, you will need to install CUDA, and specify the CUDA version when installing pytorch. 

## Windows instructions
### CUDA
If you have an NVIDIA GPU, install CUDA 11.0:
https://developer.nvidia.com/cuda-11.0-download-archive


1. Download executable (Windows - x86_64 - 10 - exe)
2. Launch installer
3. Default installation path
4. Express install
5. **NO NEED TO INSTALL VISUAL STUDIO** (tick checkbox "I understand, and wish to continue installation regardless")

### Python
1. Download and execute installer https://www.python.org/downloads/.
2. **When asked, check `Add Python to PATH`.**
3. Install

### Pytorch
To install PyTorch with pip, on the interface here : https://pytorch.org/get-started/locally/#start-locally, select `Stable` build, `Windows` OS `Pip` Package, `Python` language, and `11.0` version of CUDA. It will give you a command line for the python installation (e.g. `pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`). Copy Paste & Execute in Windows Powershell (`Start - Windows PowerShell`). It will install Pytorch in your python environment.

### VS Code
Download and install https://code.visualstudio.com/download.
Now open a python script (`.py` file) and write the following :
```python
import torch
x = torch.rand(5, 3)
print(x)
```
VS Code will propose you to install recommended extensions. Install Python extension.
Then, your python environment should be recognized.
Try and run this code which should display a randomly initialized tensor.

### To install Python packages
For example, install matplotlib using Windows PowerShell : `pip install matplotlib`

## Linux (Ubuntu) instructions

### NVIDIA drivers

The latest nvidia drivers can be installed using the command line. For Ubuntu it should be :
```bash
sudo apt install nvidia-driver-460
```

### Python / MiniConda
We recommend installing Miniconda, as it significantly simplifies the installation setup. 
Installation steps are [here](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html).

We recommend using Python 3.8. At the end of conda installation, make sure to add the conda paths to the .bashrc (it is explicitely asked). 

### Pytorch
Install pytorch with Conda (this will also install cuda necessary interfaces), on the interface here : https://pytorch.org/get-started/locally/#start-locally, select `Stable` build, `Linux` OS `Conda` Package, `Python` language, and `11.0` version of CUDA. It will give you a command line for the python installation (e.g. `conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch`). Run this command line in the terminal.

### VS Code
Download and install https://code.visualstudio.com/download.
Now open a python script (`.py` file) and write the following :
```python
import torch
x = torch.rand(5, 3)
print(x)
```
VS Code will propose you to install recommended extensions. Install Python extension.
Then, your python environment should be recognized.
Try and run this code which should display a randomly initialized tensor.

### To install Python packages
For example, install matplotlib using a terminal : `pip install matplotlib`
