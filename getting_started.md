
## General instructions
In this course, we will use pytorch for Deep Learning. Deep Learning benefits highly from acceleration using a Graphical Processing Unit (GPU). You will have access to computers with GPUs either from Campux Machines ( = any machine from the school that you can connect to using your IMT credentials. That includes the machines in the classroom) or from your Personal Laptop. In the machine with GPUs you will be using, we already have downloaded the dataset you need and installed pytorch and some other useful libraries in the ``effdl-venv`` virtual environment.

**Every binome will be assigned a computer with a GPU for the whole course.**

## Launching a script on the machines
1. Follow one of the two step-by-step guides below to see how to connect to these computers remotely using VS code, either on [Campux Machines (recommended)](#remote-vs-code-campux-machines) or on your [Personal Machines (advanced)](#remote-vs-code-personal-machines). 
2. Once you are connected remotely, open a terminal window in VS Code.
3. Activate the python environment ``effdl-venv`` (**to be done at each connection!**):
```bash
source /opt/img/effdl-venv/bin/activate
```
4. Run your script: `python3 myscript.py`

## Data storage 
By default you have access to your `/home` directory from the school, but as it is on the network it will be slower. 

In order to have a faster data access, create a directory in `/users/local` 
Remember this is an internal hard drive with limited capacity, and should only be used for temporary results. Always save your valuable code / results in a secure place. 

When saving network weights (checkpoints, in ‘pt’ or ‘pth’ format) , also use `/users/local` as they can be very large files. 

### CIFAR10 Dataset
In this course you will be using the **CIFAR10** dataset to train and test your model. We have downloaded CIFAR10 in the following folder: `/opt/img/effdl-cifar10/`

Remember to specify this path when you need to access the dataset (instead of dowloading it from scratch!)


## Remote VS Code (Campux Machines)
VS Code used for this tutorial: 1.64.2

1. Install VS Code Extension named `Remote Development`.
2. Hit F1 key and go to `Remote-SSH: Connect To Host...`
3. Enter your pc id: `fl-tp-br-xxx.imta.fr`
4. Enter your password when prompted
5. If asked about fingerprints, hit `Continue`
6. You should have a green rectangle with `SSH: fl-tp-br-xxx` on the bottom left corner of your screen. If you don't or got an error along the way, call the teacher for help.

## Remote VS Code (Personal Machines, only if connected to eduroam or VPN!!)
VS Code used for this tutorial: 1.64.2

1. Install an [OpenSSH compatible SSH client](https://code.visualstudio.com/docs/remote/troubleshooting#_installing-a-supported-ssh-client)
2. Install VS Code Extension named `Remote Development`.
3. Hit F1 key and go to `Remote-SSH: Open Configuration File...`. Click on the first line proposed.
    - On Linux or Mac, add these lines to the file (**replace YOUR_LOGIN by your own username**): 
    ```
    Host              brain1.imt
        User              YOUR_LOGIN
        Compression       yes
        HostName          fl-tp-br-xxx.imta.fr
   
    ```

    - On Windows, add these lines to the file (**replace YOUR_LOGIN by your own username**):

    ```
    Host              brain1.imt
        User              YOUR_LOGIN
        Compression       yes
        HostName          fl-tp-br-xxx.imta.fr

    ```
4. Hit F1 key and go to `Remote-SSH: Connect To Host...`
5. Select `brain1.imt`
6. Enter passwords when prompted
7. If asked about fingerprints, hit `Continue`
8. You should have a green rectangle with `SSH: fl-tp-br-xxx.imta.fr` on the bottom left corner of your screen. If you don't or got an error along the way, call the teacher for help.


***

***

# If you want to use your own machine wih GPUs (not recommended)

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

