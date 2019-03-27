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

import img_util
import util



class BaseNeuralNetwork(object):
    """kerasNetworkBase
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
        np.random.seed(param.seed)
        keras.backend.clear_session()
        self.build_network(self.param.network_arch)

    def load_network_param:
        raise NotImplementedError
        
    def build_network(self):
        raise NotImplementedError()
        
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

    def _get_X_y_batch(self,X,y,ind,vec):
        raise NotImplementedError()
        
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
    
        
def UNet(BaseNeuralNetwork):

    def build_network(self):

        # Input layer (variable size)

        # 2 sets down:
        # Conv2D
        # MaxPooling
        # Dropout
        #1st set down 
        c1 = Conv2D(filters=32,kernel_size=3,padding='same',activation='relu')(inputs)
        d1 = BatchNormalization()(c1)        
        m1 = MaxPooling2D((2,2))(d1)

        #2nd set down
        c2 = Conv2D(filters=64,kernel_size=3,padding='same',activation='relu')(m1)
        d2 = BatchNormalization()(c2)        
        m2 = MaxPooling2D((2,2))(d2)

        #middle layers
        cmid = Conv2D(filters=128, kernel_size=3,padding='same', activation=relu)(m2)
        dmid = BatchNormalization()(cmid)
        
        #1st step up
        c3 = Conv2D(filters=64,kernel_size=3,padding='same',activation='relu')(dmid)
        d3 = BatchNormalization()(c3)        
        m3 = UpSampling2D(2,2))(d3)
        # residual connection for results from upsampling with stuff from step before
        z3 = merge([m3,d2],mode='sum')
        
        #2nd step up
        c4 = Conv2D(filters=32,kernel_size=3,padding='same',activation='relu')(d3)
        d3 = BatchNormalization()(c3)                
        m4 = UpSampling2D(2,2))(c4)
        z4 = merge([d4,d1],mode='sum')

        #final 1D convolution
        final=Conv2D(kernel_size=1, activation='relu')(z4)

        self.model=model(inputs=inputs, outputs=final)
        self.model.compile(optimizer='adam',metrics=['IOU'])
        

    def _get_X_y_batch(self,indx_df,size_bucket=0,fold=0):

        #pick a random bin:

        rand_bin = np.random.randint(low=0,high=Nbin)
        size = bin_size(rand_bin)

        count=0

        Wtarget=util.Wlist[size_bucket]
        Htarget=util.Hlist[size_bucket]
        
        X=np.zeros([Nbatch, Wtarget, Htarget,3])
        i=0
        while (i< len(indx_df):
            row = indx_df.iloc[i]
            j = i % self.param.Nbatch
            X_name= '/'.join([row['dir'],row['fname']])
            y_name= X_name[:-4]+'_seg.png'
            X[j]=image.load_img(X_name, target_size=(Htarget,Wtarget))
            y[j]=image.load_img(y_name, target_size=(Htarget,Wtarget))
            if (j == (self.param.Nbatch-1)):
               return X,y
        #catch last iteration       
        return X,y

    def loss(Ytrue,Ypred):

        #define a custom loss function using the pixel-wise cross entropy.
        R = Ytrue[:,:,0]
        G = Ytrue[:,:,1]

        y_class= (R//10)*255 + G
        #get classes_present
        classes_present=set(y_class.reshape(-1))
        for i in Nclasses:
            num = class_dict[i]
            if num in classes_present:
               msk = (y_class==num)
            else:
               continue
            loss+= msk*np.log10(ypred[:,:,i]+1E-16)
        

        #only considers top-50 classes.
        #uses a dict to lookup classes.

        # use R/G channels to look up masks in those classes
        # compute cross-entropy on those.
        # average across 
               
        

               
