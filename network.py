# Tensorflow network for loading data.
# Aim to use datasets to manage data, with streaming/batch generation.
# Will use transfer learning to im

# May experiment with softer metrics for success.
# For now, use the standard metrics.  

# Will try to define a bunch of tests and lint the code for discipline.


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import PIL

import imageio

from tf.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Conv2DTranspose
import tf.keras.image as image
import tf.keras.backend as K

import img_util
import util

    
def DownBlock(inputs, filters=32, kernel_size=3, alpha=0.1,dropout=0.2,scope='Down'):
    """Make a convolutional block with DownSampling.

    Conv -> Batch Norm -> Leaky ReLU -> MaxPool -> Dropout

    Args: inputs - Input Batch of Images [Nbatch, W, H,Nchannel]
    Returns: d - Output Batch of Images [Nbatch, W/2, H/2, filters]
    """
    with K.name_scope(scope):
        c = Conv2D(filters=filters,kernel_size=kernel_size,padding='same',
                   data_format='channels_last',name=scope+'_Conv')(inputs)
        b = BatchNormalization(name=scope+'_BatchNorm')(c)
        r = LeakyReLU(alpha=alpha,name=scope+'_RELU')(b)
        m = MaxPooling2D((2,2),name=scope+'_MaxPool')(r)
        d = Dropout(dropout,name=scope+'_Dropout')(m)
        return d

def MidBlock(inputs, filters=32, kernel_size=3, alpha=0.1,dropout=0.2,scope='Mid'):
    """Make a Convolutional block in the Middle

    Conv -> Batch Norm -> Leaky ReLU -> Dropout

    Args: inputs - Input Batch of Images [Nbatch, W, H, Nchannel]
    Returns: d - Output Batch of Images [Nbatch, 2*W2, 2*H, filters]
    """
    with K.name_scope(scope):
        c = Conv2D(filters=filters,kernel_size=kernel_size,padding='same',
                   data_format='channels_last', name=scope+'_Conv')(inputs)
        b = BatchNormalization(name=scope+'_BatchNorm')(c)
        r = LeakyReLU(alpha=alpha,name=scope+'_RELU')(b)
        d = Dropout(dropout,name=scope+'_Dropout')(r)
        return d
    
def UpBlock(inputs, filters=32, kernel_size=3, alpha=0.1,dropout=0.2,scope='Up'):
    """Make a Convolutional block with Upscaling.
    (Can swap out Upscaling or Conv2DTranspose)
    Conv -> Batch Norm -> Leaky ReLU -> Upscale -> Dropout

    Args: inputs - Input Batch of Images [Nbatch, W, H, Nchannel]
    Returns: d - Output Batch of Images [Nbatch, 2*W2, 2*H, filters]
    """
    with K.name_scope(scope):
        c = Conv2D(filters=filters,kernel_size=kernel_size,padding='same',name=scope+'_Conv')(inputs)
        b = BatchNormalization(name=scope+'_BatchNorm')(c)
        r = LeakyReLU(alpha=alpha,name=scope+'_RELU')(b)
        m = Upscaling2D((2,2),name=scope+'_Upscale')(r)
        d = Dropout(dropout,name=scope+'_Dropout')(m)
        return d


class kerasUNet(object):
    """kerasUNet
    Skeleton to hold network parameters, network archictecture, 
    and functions.
    Contains:
    build_network
    batch_generator
    train_network
    rand_batch_generator
    det_batch_generator
    evaluate_network
    save_model
    load_model
    """
    def __init__(self,param_path):
        self.param=util.load_param(param_path)
        self.indx = indx.sort_values(['size_bucket','fold'],in_place=True)
        self.param.Nbatch = 10
        
        np.random.seed(param.seed)
        keras.backend.clear_session()
        self.build_network(self.param.network_arch)

    def load_network_param:
        raise NotImplementedError

    def build_network(self):

        #Input_shape=(W,H)
        #down1_shape = (W/2, H/2)
        down1 = DownBlock(inputs, filters=8, kernel_size=3, dropout=0.25,scope='Down1')
        #down2_shape = (W/4, H/4)
        down2 = DownBlock(down1, filters=16, kernel_size=3, dropout=0.25,scope='Down2')        

        #middle layers:  mid_shape = (W/4, H/4)
        mid = MidBlock(down2, filters=32, kernel_size=3,dropout=0.25)
        
        #1st step up.  up2_shape = (W/2, H/2)
        up2 = UpBlock(mid, filters=64, kernel_size=3,dropout=0.25)
        # residual connection for results from upsampling with stuff from step before
        up2b = merge([up2,down1],mode='sum')
        
        #2nd step up up1_shape = (W,H)
        up1 = UpBlock(up2b, filters=64, kernel_size=3,dropout=0.25)        
        up1b = merge([up1,inputs],mode='sum')

        #could have more convolutions here. 
        #final 1D convolution  (pixel-wise Dense layer)
        final=Conv2D(kernel_size=1, filters=Nclasses, activation='softmax')(up1b)

        self.model=model(inputs=inputs, outputs=final)
        self.model.compile(loss='pixelwise_crossentropy',
                           optimizer='adam',
                           metrics=['IOU'])


    def pixelwise_crossentropy(Ytrue,Ypred):
        """
        compute pixelwise cross-entropy across most popular classes example by example.

        Input: Ytrue Tensor (Nbatch, W, H, 3)  
               Ypred Tensor (Nbatch, W, H, Nclass)
        """
        #define a custom loss function using the pixel-wise cross entropy.
        W,H,Nc = Ypred.shape
        R = Ytrue[:,:,0]
        G = Ytrue[:,:,1]

        ytrue_class= (R//10)*255 + G
        #get classes_presenre
        classes_present = set(ytrue_class.reshape(-1))
        cost=0
        for i, class_label in enumerate(self.class_lookup):
            #get numerical value associated with class cl
            ypred_c = K.clip(ypred[:,:,i],self.param.eps,1-self.param.eps)
            if class_label in classes_present:
                #make logical mask
                msk = K.equal(ytrue_class, class_label)
                cost += K.mean(msk*K.log(ypred_c)
                              +(~msk)*K.log(1-ypred_c))/(Nclasses)
            else:
                #find all erroneous classes 
                cost += K.mean(K.log(1-ypred_c))/(Nclasses)
        return cost
        
    def _get_X_y_batch(self,size_bucket=0,fold=0):

        #pick a random bin:

        rand_bin = np.random.randint(low=0,high=Nbin)
        size = bin_size(rand_bin)

        count=0

        Wtarget=util.Wlist[size_bucket]
        Htarget=util.Hlist[size_bucket]
        
        X=K.zeros([Nbatch, Wtarget, Htarget,3])
        i=0
        while (i< len(indx_df)):
            row = self.indx.iloc[i]
            j = i % self.param.Nbatch
            X_name= '/'.join([row['dir'],row['fname']])
            y_name= X_name[:-4]+'_seg.png'
            X[j]=image.load_img(X_name, target_size=(Htarget,Wtarget))
            y[j]=image.load_img(y_name, target_size=(Htarget,Wtarget))
            if (j == (self.param.Nbatch-1)):
               return X,y
        #catch last iteration       
        return X,y

        
    #Why generator? Easier to implement early stopping.
    def rand_batch_generator(self,X,y,vec):
        """
        Generator to make random batches of data.
        Intended for use in training to endlessly
        generate samples of data.

        X - list of lists of indices
        y - true categories
        vec - np.array with vectors.
        
        Yields: Xv - subset of X expanded to desired form.
        yv - corresponding subset of labels.
        """
        while True:
            ind_sub=np.random.randint(low=0,high=len(y),
                                      size=self.param.batch_size)
            yield self._get_X_y_batch(X,y,ind_sub,vec)

        
    def det_batch_generator(self,X,y,vec):
        """det_batch_generator
        Load deterministic batch generator.
        Intended for use with inference/prediction
        to deterministically loop over data once. 
        """
        Nb=self.param.batch_size
        i0=0
        i1=Nb
        while i1<len(y):
            ind_sub=np.arange(i0,i1)
            yield self._get_X_y_batch(X,y,ind_sub,vec)
            i0=i1
            i1+=Nb
        #Last iter    
        ind_sub=np.arange(i0,len(y))        
        yield self._get_X_y_batch(X,y,ind_sub,vec)
            
    def train_network(self,X,y,vec):
        """train_network(self,X,y,vec)
        Actually train the keras model.
        Uses random batches from rand_batch_generator.
        Input: X - np.array with rows of indices
               y  - np.array with value of true class.
               vec - np.array with wordvectors
        Output: None
        Side-effect: Trains self.model.
        """
        
        self.model.fit_generator(
            self.rand_batch_generator(X,y,vec),
            epochs=self.param.Nepoch,
            steps_per_epoch=self.param.steps)

    def predict(self,X,y,vec):
        """predict(X,y,vec)
        Loop over data and predict for all samples.
        Input: X - np.array with rows of indices
               y  - np.array with value of true class.
               vec - np.array with wordvectors
        Output pred - tuple with losses/metrics over dataset
        """
        Nsteps=np.ceil(len(y)/self.param.batch_size)
        pred=self.model.predict_generator(
            self.det_batch_generator(X,y,vec),steps=Nsteps)
        return pred
    
    def evaluate(self,X,y,vec):
        """evaluate(self,X,y,vec)
        Evaluates metrics by iterating over all datapoints in X,y
        using loss and metrics specified in model.

        Input: X - np.array with rows of indices
               y  - np.array with value of true class.
               vec - np.array with wordvectors
        Output eval_scores - tuple with losses/metrics over dataset
        """
        Nsteps=np.ceil(len(y)/self.param.batch_size)
        eval_scores=self.model.evaluate_generator(
            self.det_batch_generator(X,y,vec),steps=Nsteps)
        return eval_scores

    def save_model(self,path_base):
        """saves whole model in hdf5 format."""
        
        self.model.save(path+'.model')
        util.save_param(self.param,path_base+'.param')                    

    def load_model(self,path_base):
        """loads saved model from hdf5"""
        
        self.param=util.load_param(param_path+'.param')
        self.model=keras.models.load_model(path+'.model')


"""
How to organize this?  

We have around 20k images.  We have split it into 5 buckets, and 5 size buckets.
So there are 25 divisions.

We need a flag to select out a particular fold for train/validation. e.g. Fold!=0 and Fold==0

For sizing, this is necessary for remotely efficient training.  
We can sort the index dataframe.  Build batches within allowed buckets.  

Alternatively: 

Training - loop over whole dataset, building batches for allowed size

"""

        

               
