# Image Segmentation with a U-Net

This is my work for the PDSG's Mar 2019 series on image segmentation.
We're working with the ADE20K dataset, and initially a subset of the images. 
I intend to try to build a U-Net neural network to carry out the 
segmentation and classification.

I'll be working in Keras (and using the Keras API).

Currently set up to stream batch generation from files on disk, 
using batching based on size, with k-fold validation built in.

Trained to predict top 20-50 classes only with one-hot labelling.

Ultimately, it would be great to use the TFRecordDataset to speed up training.

Initially, let's set this up for a shallow network (2 layers, with easy 
ways to scale up).


Files:
/code/src
     - network.py - contains network builder, compiler, fit, batching and metrics. 
    - blocks.py - contains blocks for convolutional network and up/down sampling. 
    - img_util.py - functions to load indexes, transform indices, plot images.
    - basic.param - parameters to load into network.
/code
    - read\_clean\_index.m   - function to transform ADE20K index.mat file into tsv with desired features.

