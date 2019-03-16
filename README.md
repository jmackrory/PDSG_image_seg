
# Image Segmentation with a U-Net

This is my work for the PDSG's Mar 2019 series on image segmentation.
We're working with the ADE20K dataset, and initially a subset of the images. 
I intend to try to build a U-Net neural network to carry out the 
segmentation and classification.

First up, let's do some transfer learning.
I'll try to use one of the classic ImageNet winners as the initial
layers.

I'll be working in Keras (and using the Keras API).

I plan to use the batch generator, and get some time on an Amazon GPU,
once I've debugged things on my laptop.

Initially, let's set this up for a shallow network (2 layers, with easy 
ways to scale up)

Files:
 - network.py
 --define Base
 
 - util.py 
 -- functions to dynamically load files of similar size, pad, and batch. 
 - 
