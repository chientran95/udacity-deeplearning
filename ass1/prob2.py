
# coding: utf-8

# Deep Learning
# =============
# 
# Assignment 1
# ------------
# 
# The objective of this assignment is to learn about simple data curation practices, and familiarize you with some of the data we'll be reusing later.
# 
# This notebook uses the [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) dataset to be used with python experiments. This dataset is designed to look like the classic [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, while looking a little more like real data: it's a harder task, and the data is a lot less 'clean' than MNIST.

# In[3]:


# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import pdb

# ---
# Problem 2
# ---------
# 
# Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray. Hint: you can use matplotlib.pyplot.
# 
# ---

dirs = ['notMNIST_large', 'notMNIST_small']
for directory in dirs:
    letter = sorted([char for char in os.listdir(directory) if os.path.isfile(os.path.join(directory, char))])
    for let in letter:
        with open(os.path.join(directory, let), 'rb') as pklfile:
            data = pickle.load(pklfile)
            idx = np.random.choice(data.shape[0])
            sample = data[idx]
            plt.imshow(sample, cmap='gray')
            plt.title(os.path.join(directory, let))
            plt.show()
