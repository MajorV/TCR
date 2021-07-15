import socket
import sys
import os
import torch
import numpy as np
print('*'*50)
print('Using new method for structuring projects')
print('Using Pycharm locally on my laptop to have no latency')
print('Specifying an ssh interpreter on UCF computer Andromeda to actually run the code')
print('data is read in and written out on Andromeda')
print('also making strict use of git and github')
print('interpreter:   ssh://bruce@10.173.214.70:22/home/bruce/anaconda3/bin/python')
print('hostname:',socket.gethostname())
print('data location',socket.gethostname() + ':' + os.getcwd())
print('git repo: https://github.com/MajorV/TCR')
print('instructions on how to do this are contained in this files comments')
print('*'*50)

'''
Procedure to setup a project like this in this order:
1)create the local project
2)create on Andromeda the base directory for project. source code and results will appear here
3)specify the ssh interpreter using ip address and path (ssh://bruce@10.173.214.70:22/home/bruce/anaconda3/bin/python)
4)enable git for source control of the project
5)share to git
'''


