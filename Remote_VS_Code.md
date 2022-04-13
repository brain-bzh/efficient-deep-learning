# Computer assignation
Every group will be assigned a computer with a GPU for the whole course.
[Work groups organization](https://docs.google.com/document/d/1RhU58oDF8zuCu3yXcQzmQn7ZCacSuL8YbmbcFwi6sSg/edit?usp=sharing)

# Launching a script on the machines
1. Follow one of the two step-by-step guides below to see how to connect to these computers remotely using VS code, either on [Campux Machines (recommended)](#remote-vs-code-campux-machines) or on your [Personal Machines (advanced)](#remote-vs-code-personal-machines).
2. Once you are connected remotely, open a terminal window in VS Code.
3. Activate the deep learning environment: `source /opt/campux/virtualenv/deeplearning-u20/bin/activate`. Pytorch is installed in this environment.
4. Run your script: `python3 myscript.py`

# Data storage
By default you have access to your /home directory from the school, but as it is on the network it will be slower. 

In order to have a faster data access, create a directory in /users/local 
Remember this is an internal hard drive with limited capacity, and should only be used for temporary results. Always save your valuable code / results in a secure place. 

For training networks, the datasets SHOULD be stored in /users/local/xxxx , otherwise training will be massively slowed down. 

When saving network weights (checkpoints, in ‘pt’ or ‘pth’ format) , also use /users/local as they can be very large files. 


# Remote VS Code (Campux Machines)
VS Code used for this tutorial: 1.64.2

1. Install VS Code Extension named `Remote Development`.
2. Hit F1 key and go to `Remote-SSH: Connect To Host...`
3. Enter your pc id: `pc-elec-xxx`
4. Enter your password when prompted
5. If asked about fingerprints, hit `Continue`
6. You should have a green rectangle with `SSH: pc-elec-xxx` on the bottom left corner of your screen. If you don't or got an error along the way, call the teacher for help.

# Remote VS Code (Personal Machines)
VS Code used for this tutorial: 1.64.2

1. Install an [OpenSSH compatible SSH client](https://code.visualstudio.com/docs/remote/troubleshooting#_installing-a-supported-ssh-client)
2. Install VS Code Extension named `Remote Development`.
3. Hit F1 key and go to `Remote-SSH: Open Configuration File...`. Click on the first line proposed.
    - On Linux or Mac, add these lines to the file (**replace YOUR_LOGIN by your own username**): 
    ```
    Host              brain1.imt
        User              YOUR_LOGIN
        Compression       yes
        HostName          pc-elec-XXX.priv.enst-bretagne.fr

    Host *.imt
        ProxyCommand ssh YOUR_LOGIN@ssh.telecom-bretagne.eu "/bin/nc `basename %h .imt` %p"
    ```

    - On Windows, add these lines to the file (**replace YOUR_LOGIN by your own username**):

    ```
    Host              brain1.imt
        User              YOUR_LOGIN
        Compression       yes
        HostName          pc-elec-XXX.priv.enst-bretagne.fr

    Host *.imt
        ProxyCommand C:\Windows\System32\OpenSSH\ssh.exe YOUR_LOGIN@ssh.telecom-bretagne.eu "/bin/nc `basename %h .imt` %p"
    ```
4. Hit F1 key and go to `Remote-SSH: Connect To Host...`
5. Select `brain1.imt`
6. Enter passwords when prompted
7. If asked about fingerprints, hit `Continue`
8. You should have a green rectangle with `SSH: pc-elec-XXX` on the bottom left corner of your screen. If you don't or got an error along the way, call the teacher for help.

