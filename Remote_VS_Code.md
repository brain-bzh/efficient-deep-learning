# Configure Remote SSH for VS Code

VS Code used for this tutorial: 1.43.2

1. Install an [OpenSSH compatible SSH client](https://code.visualstudio.com/docs/remote/troubleshooting#_installing-a-supported-ssh-client)
2. Install VS Code Extension named "Remote Development".
3. Hit F1 key and go to "Remote-SSH: Open Configuration File..."
    - On linux, add these lines to the file (**replace m20leona by your own username**): 
    ```
    Host              brain1.imt
        User              m20leona
        Compression       yes
        HostName          10.29.232.81

    Host *.imt
        ProxyCommand ssh m20leona@ssh.telecom-bretagne.eu "/bin/nc `basename %h .imt` %p"
    ```

    - On Windows, add these lines to the file (**replace m20leona by your own username**):

    ```
    Host              brain1.imt
        User              m20leona
        Compression       yes
        HostName          10.29.232.81

    Host *.imt
        ProxyCommand C:\Windows\System32\OpenSSH\ssh.exe m20leona@ssh.telecom-bretagne.eu "/bin/nc `basename %h .imt` %p"
    ```
4. Hit F1 key and go to "Remote-SSH: Connect To Host..."
5. Select "brain1.imt"
6. If you get the following error message : 
> Failed to find a non-Windows SSH installed. Password prompts may not be displayed properly! Disable `remote.SSH.useLocalServer` if needed.

It's a known bug. Follow instructions [here](https://github.com/microsoft/vscode-remote-release/issues/2523#issuecomment-597551802) and downgrade Remote - SSH extension to 0.49.0.

7. Enter passwords when prompted
8. If asked about fingerprints, hit "Continue"
