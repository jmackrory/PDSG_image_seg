# Tensorflow network for loading data.
# Aim to use datasets to manage data, with streaming/batch generation.
# Will use transfer learning to im

# May experiment with softer metrics for success.
# For now, use the standard metrics.  

# Will try to define a bunch of tests and lint the code for discipline.


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# import PIL
# import imageio

import keras
from keras.layers import Input, Dropout, Dense, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers import LeakyReLU, Add
from keras.models import Model
import keras.preprocessing.image as image
import keras.backend as K

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
        m = UpSampling2D((2,2),name=scope+'_Upscale')(r)
        d = Dropout(dropout,name=scope+'_Dropout')(m)
        return d

def SkipConnection(In1, In2,dim2,scope='Skip'):
    """Make a skip connection from one side of U to the other.
    Includes a 1x1 convolution to change dimension.
    Inputs:
    In1 - tensor from down leg
    In2 - tensor from up leg
    dim2 -number of channels on up leg.
    Return:
    skip - tensor from In1+In2
    """
    C1=Conv2D(kernel_size=1,filters=dim2,padding='same',name=scope+'_Conv')(In1)
    skip = Add()([C1,In2])
    return skip
    
class NetworkParam(object):

    """NetworkParam
    Object to hold parameters for use in constructing 
    Neural Networks.
    """
    def __init__(self,param_file,**kwargs):
        #Gotta be a less stupid way of doing this.
        #So set some sensible defaults.
        self.learning_rate=0.001
        self.activation='relu'
        self.wordvec_dim=50
        self.regtype='dropout'
        self.dropout=0.25
        self.metrics=['acc']
        self.optimizer="adam"
        self.loss='sparse_categorical_crossentropy'
        self.seed=1029
        self.batch_size=100
        self.Nepoch=5
        self.kernel_size=3
        self.pool=2

        param_dict=util.load_param(param_file)
        #Now overwrite any values with the 
        for key,value in param_dict.items():
            setattr(self,key,value)

        #Now overwrite any values with the 
        for key,value in kwargs.items():
            setattr(self,key,value)

    def save_param(self,paramdict,paramfile):
        """saves parameter dict to JSON"""
        with open(paramfile,'rb') as f:
            json.save(paramdict,f)

    def load_param(self, paramfile):
        """loads parameters from JSON to a Python Dict"""
        with open(paramfile,'rb') as f:
            param_dict=json.load(f)
        return param_dict
            

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
    def __init__(self,indx_df,object_df,param_file='basic.param'):
        self.indx = indx_df
        self.param = NetworkParam(param_file)
        self.train_dict, self.val_dict = img_util.get_training_dicts(indx_df,self.param.size_buckets,self.param.val_fold)
        self.class_lookup=img_util.get_common_class_indx(object_df,Nclasses=self.param.Nclasses)
        self.param.train_steps = len(self.train_dict)//self.param.batch_size
        self.param.val_steps = len(self.val_dict)//self.param.batch_size
        
        np.random.seed(self.param.seed)
        keras.backend.clear_session()
        self.build_network()

        
        
    def load_param(self,param_file):

        param_dict=util.load_param(param_file)
        #Now overwrite any values with the 
        for key,value in kwargs.items():
            setattr(self,key,value)
        
    def build_network(self):

        #Input_shape=(W,H)
        #down1_shape = (W/2, H/2)
        inputs=Input(shape=(None,None,3))
        down1 = DownBlock(inputs, filters=8, kernel_size=3, dropout=self.param.dropout,scope='Down1')
        #down2_shape = (W/4, H/4)
        down2 = DownBlock(down1, filters=16, kernel_size=3, dropout=self.param.dropout,scope='Down2')        

        #middle layers:  mid_shape = (W/4, H/4)
        mid = MidBlock(down2, filters=32, kernel_size=3,dropout=self.param.dropout,scope='Mid')
        
        #1st step up.  up2_shape = (W/2, H/2)
        up2 = UpBlock(mid, filters=64, kernel_size=3,dropout=self.param.dropout,scope='Up2')
        # residual connection for results from upsampling with stuff from step before
        
        up2b = SkipConnection(down1,up2,64,scope='Skip2')
        
        #2nd step up up1_shape = (W,H)
        up1 = UpBlock(up2b, filters=64, kernel_size=3,dropout=self.param.dropout,scope='Up1')        
        up1b = SkipConnection(inputs,up1,64,scope='Skip1')

        #could have more 2D convolutions here. 
        #final 1D convolution  (pixel-wise Dense layer)
        final=Conv2D(kernel_size=1, filters=self.param.Nclasses, activation='softmax',name='FinalConv')(up1b)

        self.model=Model(inputs=inputs, outputs=final)
        self.model.compile(loss=self.pixelwise_crossentropy,
                           optimizer="adam",
                           metrics=[self.IOU])

    def IOU(self,Ytrue,Ypred):
        """
        compute pixelwise cross-entropy across most popular classes example by example.

        Input: Ytrue Tensor (Nbatch, W, H, 3)  
               Ypred Tensor (Nbatch, W, H, Nclass)
        """
        #define a custom loss function using the pixel-wise cross entropy.
        #Nb,W,H,Nc = Ypred.shape
        R = Ytrue[:,:,0]
        G = Ytrue[:,:,1]

        ytrue_class= (R//10)*256 + G
        #get classes_present
        #classes_present = set(K.reshape(ytrue_class,-1))
        #classes_present = tf.unique(K.reshape(ytrue_class,-1))
        cost=0
        for i, class_label in enumerate(self.class_lookup):
            #get numerical value associated with class cl
            pred_msk = tf.cast(Ypred[:,:,i]>0.5,tf.int8)
            #if class_label in classes_present:
            #make logical mask
            true_msk = tf.cast(K.equal(ytrue_class, class_label),tf.int8)
            iou = K.sum(true_msk * pred_msk)/(K.sum(true_msk + pred_msk)+1)
            cost = cost+iou
        cost = cost/self.param.Nclasses
        return cost
    

    def pixelwise_crossentropy(self,Ytrue,Ypred):
        """
        compute pixelwise cross-entropy across most popular classes example by example.

        Input: Ytrue Tensor (Nbatch, W, H, 3)  
               Ypred Tensor (Nbatch, W, H, Nclass)
        """
        #define a custom loss function using the pixel-wise cross entropy.
        #W,H,Nc = tf.shape(Ypred)
        R = Ytrue[:,:,0]
        G = Ytrue[:,:,1]

        ytrue_class= (R//10)*256 + G
        #get classes_presenre
        #classes_present = set(K.reshape(ytrue_class,-1))
        #classes_present = tf.unique(K.reshape(ytrue_class,[-1,W*H]))        
        cost=0
        for i, class_label in enumerate(self.class_lookup):
            #get numerical value associated with class cl
            Ypred_c = K.clip(Ypred[:,:,i],self.param.eps,1-self.param.eps)
            #if class_label in classes_present:
            #make logical mask
            msk = K.equal(ytrue_class, class_label)
            cost = cost+K.sum(K.log(tf.boolean_mask(Ypred_c, msk  ))) \
                   +K.sum(K.log(tf.boolean_mask(1-Ypred_c, (~msk)  )))
            #else:
            #    #find all erroneous classes 
            #    cost += K.mean(K.log(1-ypred_c))/(Nclasses)
        shapes=tf.cast(tf.shape(Ypred),tf.float32)
        cost = cost/(self.param.Nclasses*shapes[-2]*shapes[-3])
        return cost
        
    def _get_X_y_batch(self,ind,size_bucket=0):
        """given list of indices and a corresponding bucket,
        returns a batch for training.  
        """
        Wtarget=img_util.Wlist[size_bucket+1]
        Htarget=img_util.Hlist[size_bucket+1]

        Nb=len(ind)
        X=np.zeros([Nb, Wtarget, Htarget,3])
        y=np.zeros([Nb, Wtarget, Htarget,3])
        for j,i in enumerate(ind):
            row = self.indx.iloc[i]
            X_name= '/'.join([row['dir'],row['fname']])
            y_name= X_name[:-4]+'_seg.png'
            X[j]=image.load_img(X_name, target_size=(Wtarget,Htarget))
            y[j]=image.load_img(y_name, target_size=(Wtarget,Htarget))
        return X,y

    def get_random_train_bucket(self,file_dict):
        """generator to pick a random allowed_bucket using relative sizes
        """
        #list of tuples with key and number in each bucket.
        N_in_bucket=[(key, len(val)) for key,val in file_dict.items()]
        vals=[x[1] for x in N_in_bucket]
        #maps those sizes to [0,1) interval to randomly select one
        rand_boundary=np.cumsum(vals)/np.sum(vals)
        while True:
            r=np.random.random()
            for i,v in enumerate(vals):
                if (r<v):
                    yield N_in_bucket[i][0] 
    
    def rand_batch_generator(self,file_dict):
        """
        Generator to make random batches of data.
        Intended for use in training to endlessly
        generate samples of data.

        dict - dict of list of files to lookup
        """
        while True:
            size_bucket=self.get_random_train_bucket(file_dict)
            ind_sub=np.random.choice(self.file_dict[size_bucket],
                                     size=self.param.batch_size)
            yield self._get_X_y_batch(ind_sub,size_bucket)

    def det_batch_generator(self,file_dict):
        """det_batch_generator
        Load deterministic batch generator.
        Intended for use with inference/prediction
        to deterministically loop over data once. 
        """
        Nb=self.param.batch_size
        for bucket,ind_list in file_dict.items():
            N_in_bucket=len(ind_list)
            i0=0
            i1=Nb
            while i1<len(ind_list):
                ind_sub=np.arange(i0,i1)
                yield self._get_X_y_batch(ind_sub,bucket)
                i0=i1
                i1+=Nb
            #Last iter    
            ind_sub=np.arange(i0,len(y))        
            yield self._get_X_y_batch(ind_sub,bucket)
            
    def train_network(self):
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
            self.rand_batch_generator(self.train_dict),
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
            self.det_batch_generator(self.val_dict),steps=Nsteps)
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
            self.det_batch_generator(),steps=Nsteps)
        return eval_scores

    def save_model(self):
        """saves whole model in hdf5 format."""

        save_name='/'.join([self.param.model_dir,self.param.model_name])
        self.model.save(save_name+'.model')
        self.param.save_param(self.param,save_name+'.param')                    

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

    
    
