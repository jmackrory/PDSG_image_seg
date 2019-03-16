# Tensorflow network for loading data.
# Aim to use datasets to manage data, with streaming/batch generation.
# Will use transfer learning to im

# May experiment with softer metrics for success.
# For now, use the standard metrics.  

# Will try to define a bunch of tests and lint the code for discipline.


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import img_util

class BaseNetworkClass:
    
    def __init__(self,param_path):
        

    def load_network_param:

        raise NotImplementedError
        
    def make_network:
        raise NotImplementedError

    def train_network:
        raise NotImplementedError

    def infer:
        raise NotImplementedError
