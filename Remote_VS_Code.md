

# Configure Remote SSH for VS Code (Campux Machines)
VS Code used for this tutorial: 1.64.2

1. Install an [OpenSSH compatible SSH client](https://code.visualstudio.com/docs/remote/troubleshooting#_installing-a-supported-ssh-client)
2. Install VS Code Extension named `Remote Development`.
3. Hit F1 key and go to `Remote-SSH: Connect To Host...`
4. Enter your pc id: `pc-elec-xxx`
5. Enter your password when prompted
6. If asked about fingerprints, hit `Continue`
7. You should have a green rectangle with `SSH: pc-elec-xxx` on the bottom left corner of your screen. If you don't or got an error along the way, call the teacher for help.

# Configure Remote SSH for VS Code (Personal Machines)
VS Code used for this tutorial: 1.64.2

1. Install an [OpenSSH compatible SSH client](https://code.visualstudio.com/docs/remote/troubleshooting#_installing-a-supported-ssh-client)
2. Install VS Code Extension named `Remote Development`.
3. Hit F1 key and go to `Remote-SSH: Open Configuration File...`
    - On linux, add these lines to the file (**replace m20leona by your own username**): 
    ```
    Host              brain1.imt
        User              YOUR_LOGIN
        Compression       yes
        HostName          10.29.232.81

    Host *.imt
        ProxyCommand ssh YOUR_LOGIN@ssh.telecom-bretagne.eu "/bin/nc `basename %h .imt` %p"
    ```

    - On Windows, add these lines to the file (**replace m20leona by your own username**):

    ```
    Host              brain1.imt
        User              YOUR_LOGIN
        Compression       yes
        HostName          10.29.232.81

    Host *.imt
        ProxyCommand C:\Windows\System32\OpenSSH\ssh.exe YOUR_LOGIN@ssh.telecom-bretagne.eu "/bin/nc `basename %h .imt` %p"
    ```
4. Hit F1 key and go to `Remote-SSH: Connect To Host...`
5. Select `brain1.imt`
6. Enter passwords when prompted
7. If asked about fingerprints, hit `Continue`
8. You should have a green rectangle with `SSH: pc-elec-xxx` on the bottom left corner of your screen. If you don't or got an error along the way, call the teacher for help.

