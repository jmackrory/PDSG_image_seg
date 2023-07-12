# Tensorflow network for loading data.
# Aim to use datasets to manage data, with streaming/batch generation.
# Will use transfer learning to im

# May experiment with softer metrics for success.
# For now, use the standard metrics.

# Will try to define a bunch of tests and lint the code for discipline.

import json

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dropout, Dense, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, Add
from tensorflow.keras.models import Model
import tensorflow.keras.preprocessing.image as image
import tensorflow.keras.backend as K

from imageseg.img_util import get_training_dicts, get_common_class_index
import imageseg.util import load_param, save_param

from imageseg.blocks import DownBlock, MidBlock, UpBlock, SkipConnection


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

tf.keras.backend.set_image_data_format('channels_last')

class NetworkParam(object):

    """NetworkParam
    Object to hold parameters for use in constructing
    Neural Networks.
    """
    def __init__(self,param_file,**kwargs):
        #Gotta be a less stupid way of doing this.
        #So set some sensible defaults.
        self.learning_rate = 0.001
        self.activation = 'relu'
        self.wordvec_dim = 50
        self.regtype = 'dropout'
        self.dropout = 0.05
        self.metrics = ['acc']
        self.optimizer = "adam"
        self.loss = 'sparse_categorical_crossentropy'
        self.seed = 1029
        self.batch_size = 12
        self.Nepoch = 5
        self.kernel_size = 3
        self.pool = 2

        param_dict = load_param(param_file)
        #Now overwrite any values with the paramfile
        for key,value in param_dict.items():
            setattr(self,key,value)

        #Now overwrite any values with the kwargs.
        for key,value in kwargs.items():
            setattr(self,key,value)

    def save_param(self, paramdict, paramfile):
        """saves parameter dict to JSON"""
        with open(paramfile,'rb') as f:
            json.save(paramdict,f)

    def load_param(self, paramfile):
        """loads parameters from JSON to a Python Dict"""
        with open(paramfile,'rb') as f:
            param_dict = json.load(f)
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
    def __init__(self,index_df,object_df,param_file='basic.param'):
        self.index  =  index_df
        self.param = NetworkParam(param_file)
        self.train_dict, self.val_dict = get_training_dicts(index_df,self.param.size_buckets,self.param.val_fold)
        self.class_lookup = get_common_class_index(object_df,Nclasses=self.param.Nclasses)

        Ntrain = 0
        for key, file_list in self.train_dict.items():
            Ntrain += len(file_list)
        Nval = 0
        for key, file_list in self.val_dict.items():
            Nval+=len(file_list)

        self.param.train_steps = np.ceil(Ntrain/self.param.batch_size)
        self.param.val_steps = np.ceil(Nval/self.param.batch_size)

        np.random.seed(self.param.seed)
        keras.backend.clear_session()
        self.build_network()

    def load_param(self,param_file):

        param_dict = util.load_param(param_file)
        #Now overwrite any values with the
        for key,value in kwargs.items():
            setattr(self,key,value)

    def build_network(self):

        #Input_shape = (W,H)
        #down1_shape = (W/2, H/2)
        inputs = Input(shape=(None,None,3))
        down1 = DownBlock(inputs, filters=4, kernel_size=3, dropout=self.param.dropout,scope='Down1')
        #down2_shape = (W/4, H/4)
        down2 = DownBlock(down1, filters=8, kernel_size=3, dropout=self.param.dropout,scope='Down2')
        #down3_shapes:  (W/8, H/8)
        down3 = DownBlock(down2, filters=16, kernel_size=3, dropout=self.param.dropout,scope='Down3')

        #middle layers:  mid_shape = (W/8, H/8)
        mid = MidBlock(down3, filters=16, kernel_size=3, dropout=self.param.dropout,scope='Mid')

        #1st step up.  up2_shape = (W/4, H/4)
        up3 = UpBlock(mid, filters=32, kernel_size=3, dropout=self.param.dropout, scope='Up3')
        up3b = SkipConnection(down2 ,up3, 32,scope='Skip3')

        #1st step up.  up2_shape = (W/2, H/2)
        up2 = UpBlock(up3b, filters=32, kernel_size=3,dropout=self.param.dropout, scope='Up2')
        # residual connection for results from upsampling with stuff from step before
        up2b = SkipConnection(down1,up2,32,scope='Skip2')

        #2nd step up up1_shape = (W,H)
        up1 = UpBlock(up2b, filters=32, kernel_size=3, dropout=self.param.dropout, scope='Up1')
        up1b = SkipConnection(inputs,up1,32,scope='Skip1')

        #could have more 2D convolutions here.
        #final 1D convolution  (pixel-wise Dense layer)
        #softmax to make a probability distribution across classes.
        final = Conv2D(kernel_size=1, filters=self.param.Nclasses, activation='softmax',name='FinalConv')(up1b)

        self.model = Model(inputs=inputs, outputs=final)
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
        R = Ytrue[:,:,:,0]
        G = Ytrue[:,:,:,1]

        Ytrue_class= (R//10)*256 + G
        #get classes_present
        #classes_present = set(K.reshape(ytrue_class,-1))
        #classes_present = tf.unique(K.reshape(ytrue_class,-1))
        cost = 0
        for i, class_label in enumerate(self.class_lookup):
            #get numerical value associated with class cl
            pred_msk = Ypred[:,:,:,i]>0.5
            #if class_label in classes_present:
            #make logical mask
            label_msk = K.equal(Ytrue_class, class_label)
            label_AND_pred = tf.cast(tf.math.logical_and(label_msk, pred_msk),tf.float32)
            label_OR_pred = tf.cast(tf.math.logical_or(label_msk, pred_msk),tf.float32)
            iou = K.sum(label_AND_pred)/(K.sum(label_OR_pred)+0.01)
            cost = cost+iou
        cost = cost/self.param.Nclasses
        return cost


    def pixelwise_crossentropy(self, Ytrue, Ypred):
        """
        compute pixelwise cross-entropy across most popular classes example by example.

        Input: Ytrue Tensor (Nbatch, W, H, 3)
               Ypred Tensor (Nbatch, W, H, Nclass)
        """
        #define a custom loss function using the pixel-wise cross entropy.
        #W,H,Nc = tf.shape(Ypred)
        R = Ytrue[:,:,:,0]
        G = Ytrue[:,:,:,1]

        ytrue_class = (R//10)*256 + G
        #get classes_present
        shapes = tf.cast(tf.shape(ytrue_class),tf.float32)
        #classes_present = set(K.reshape(ytrue_class,-1))
        classes_present = tf.unique(K.reshape(ytrue_class,[K.prod(shapes)]))
        cost = 0
        for i, class_label in enumerate(self.class_lookup):
            #get numerical value associated with class cl
            Ypred_c = K.clip(Ypred[:,:,:,i],self.param.eps,1-self.param.eps)
            #if class_label in classes_present:
            #make logical mask
            msk = K.equal(ytrue_class, class_label)
            cost = cost - K.sum(tf.boolean_mask(K.log(1-Ypred_c), tf.math.logical_not(msk)  ))
            cost = cost - K.sum(tf.boolean_mask(K.log(Ypred_c), msk  ))

            #else:
            #    #find all erroneous classes
            #    cost += K.mean(K.log(1-ypred_c))/(Nclasses)
        shapes = tf.cast(tf.shape(Ypred),tf.float32)
        cost = cost/(self.param.Nclasses*shapes[1]*shapes[2])
        return cost

    def _get_X_y_batch(self,ind,size_bucket=0):
        """given list of indices and a corresponding bucket,
        returns a batch for training.
        """
        Wtarget = img_util.Wlist[size_bucket+1]
        Htarget = img_util.Hlist[size_bucket+1]

        Nb = len(ind)
        X = np.zeros([Nb, Wtarget, Htarget,3])
        y = np.zeros([Nb, Wtarget, Htarget,3])
        for j,i in enumerate(ind):
            row = self.index.iloc[i]
            X_name =  '/'.join([row['folder'],row['filename']])
            y_name =  X_name[:-4]+'_seg.png'
            X[j] = image.load_img(X_name, target_size = (Wtarget,Htarget))
            y[j] = image.load_img(y_name, target_size=(Wtarget,Htarget))
        return X,y

    def get_random_train_bucket(self,file_dict):
        """generator to pick a random allowed_bucket using relative sizes
        """
        #list of tuples with key and number in each bucket.
        N_in_bucket = [(key, len(val)) for key,val in file_dict.items()]
        vals = [x[1] for x in N_in_bucket]
        #maps those sizes to [0,1) interval to randomly select one
        rand_boundary = np.cumsum(vals)/np.sum(vals)
        r = np.random.random()
        for i,v in enumerate(rand_boundary):
            if (r<v):
                return N_in_bucket[i][0]

    def rand_batch_generator(self,file_dict):
        """  Generator to make random batches of data.
        Intended for use in training to endlessly
        generate samples of data.

        dict - dict of list of files to lookup
        """
        while True:
            size_bucket = self.get_random_train_bucket(file_dict)
            ind_sub = np.random.choice(file_dict[size_bucket],
                                     size=self.param.batch_size)
            yield self._get_X_y_batch(ind_sub,size_bucket)

    def det_batch_generator(self,file_dict):
        """det_batch_generator
        Load deterministic batch generator.
        Intended for use with inference/prediction
        to deterministically loop over data once.
        """
        Nb = self.param.batch_size
        for bucket,ind_list in file_dict.items():
            N_in_bucket = len(ind_list)
            i0 = 0
            i1 = Nb
            while i1<len(ind_list):
                ind_sub = np.arange(i0,i1)
                yield self._get_X_y_batch(ind_sub,bucket)
                i0 = i1
                i1 += Nb
            #Last iter
            ind_sub = np.arange(i0,len(y))
            yield self._get_X_y_batch(ind_sub,bucket)

    def train_network(self):
        """train_network()
        Actually train the keras model.
        Uses random batches from rand_batch_generator and train_dict
        Input: None
        Output: None
        Side-effect: Trains self.model.
        """
        self.model.fit_generator(
            self.rand_batch_generator(self.train_dict),
            epochs = self.param.Nepoch,
            steps_per_epoch = self.param.train_steps)

    def predict_all(self,file_dict):
        """predict(X,y,vec)
        Loop over whole file_dict and predict for all samples (HUGE TASK! DONT DO IT)
        Input: X - np.array with rows of indices
               y  - np.array with value of true class.
               vec - np.array with wordvectors
        Output pred - tuple with losses/metrics over dataset
        """
        Ntot = 0
        for key,val in file_dict.items():
            Ntot+=len(val)

        Nsteps = np.ceil(Ntot/self.param.batch_size)

        pred = self.model.predict_generator(
            self.det_batch_generator(file_dict),steps=Nsteps)
        return pred

    def predict_afew(self,ind_sub=np.arange(3),size_bucket=0):
        """
        Loop over a few instances and data and predict for all samples.
        Input: file_dict - dict with list of entries to sample from index_df
               ind_sub  - np.array with locations to use
               size_bucket - bucket from file_dict to try using
        Output pred - Output images for that batch  (pixelwise probabilities for that class
               batch - initial raw images
               y - ground truth

        """
        batch, y = self._get_X_y_batch(ind_sub,size_bucket)
        pred = self.model.predict(batch)
        return pred, batch, y

    def evaluate(self):
        """
        Evaluates metrics by iterating over all files in file_dict
        using loss and metrics specified in model.

        Input: file_dict - np.array with rows of indices
        Output eval_scores - tuple with losses/metrics over dataset
        """
        Nsteps = np.ceil(len(y)/self.param.batch_size)
        eval_scores = self.model.evaluate_generator(
            self.det_batch_generator(),steps=Nsteps)
        return eval_scores

    def save_model(self):
        """saves whole model in hdf5 format."""

        save_name = '/'.join([self.param.model_dir,self.param.model_name])
        self.model.save(save_name+'.model')
        self.param.save_param(self.param,save_name+'.param')

    def load_model(self,path_base):
        """loads saved model from hdf5"""

        self.param = util.load_param(param_path+'.param')
        self.model = keras.models.load_model(path+'.model')

    def get_most_likely(self,pred):
        """convert a (width, height, Nclass) array to (width,height,3) array R/G colors.
        """
        return img_util.get_color_from_pred(pred, self.class_lookup)

    def compare_pred_images(self, pred, y, num=0):

        plt.figure()
        plt.subplot(121)
        plt.imshow(pred2[num].astype(int))

        plt.subplot(122)
        tmp = np.zeros(y[num].shape)
        for i in range(2):
            tmp[:,:,i] = y[num,:,:,i]
        plt.imshow(tmp.astype(int))

if __name__ =="__main__":
    index_df, object_df = img_util.load_and_clean_indices()
    UNet = kerasUNet(index_df, object_df)
    UNet.train_network()
    pred, batch, y = UNet.predict_afew(range(5))
    pred2 = UNet.get_most_likely(pred)
    UNet.compare_pred_images(pred2, y, 0)
