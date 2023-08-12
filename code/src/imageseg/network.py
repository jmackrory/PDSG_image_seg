# Tensorflow network for loading data.
# Aim to use datasets to manage data, with streaming/batch generation.
# Will use transfer learning to im

# May experiment with softer metrics for success.
# For now, use the standard metrics.

# Will try to define a bunch of tests and lint the code for discipline.

import json
import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.preprocessing.image as tfimage
import tensorflow.keras.backend as K

from imageseg.img_util import (
    DATA_PATH,
    get_color_from_pred,
    get_training_dicts,
    get_common_class_index,
    Wlist,
    Hlist,
    load_and_clean_indices,
)
from imageseg.util import load_param

from imageseg.blocks import DownBlock, MidBlock, UpBlock, SkipConnection


# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
# config = tf.ConfigProto(gpu_options=gpu_options)
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

K.set_image_data_format("channels_last")

PARAM_PATH = "/tf/models"


class NetworkParam(object):

    """NetworkParam
    Object to hold parameters for use in constructing
    Neural Networks.
    """

    def __init__(self, param_file, **kwargs):
        # Gotta be a less stupid way of doing this.
        # So set some sensible defaults.
        self.learning_rate = 0.001
        self.activation = "relu"
        self.wordvec_dim = 50
        self.regtype = "dropout"
        self.dropout = 0.05
        self.metrics = ["acc"]
        self.optimizer = "adam"
        self.loss = "sparse_categorical_crossentropy"
        self.seed = 1029
        self.batch_size = 12
        self.Nepoch = 5
        self.kernel_size = 3
        self.pool = 2
        self.score_tree = False

        self.param_file = param_file
        param_dict = load_param(self.param_file)
        # Now overwrite any values with the paramfile
        for key, value in param_dict.items():
            setattr(self, key, value)

        # Now overwrite any values with the kwargs.
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.Nclasses_final = (
            np.ceil(np.log2(self.Nclasses))
            if self.score_tree is True
            else self.Nclasses
        )

    def save_param(self, paramdict, paramfile):
        """saves parameter dict to JSON"""
        with open(paramfile, "rb") as f:
            json.save(paramdict, f)

    def load_param(self, paramfile):
        """loads parameters from JSON to a Python Dict"""
        with open(paramfile, "rb") as f:
            param_dict = json.load(f)
        return param_dict


class kerasUNetBase(object):
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

    def __init__(self, index_df, object_df, param_file="basic.param"):
        self.index = index_df
        param_path = os.path.join(PARAM_PATH, param_file)
        self.param = NetworkParam(param_path)
        self.train_dict, self.val_dict = get_training_dicts(
            index_df, self.param.size_buckets, self.param.val_fold
        )
        self.class_lookup = get_common_class_index(
            object_df, Nclasses=self.param.Nclasses
        )

        self._get_bucket_boundaries(self.train_dict)

        Ntrain = 0
        for key, file_list in self.train_dict.items():
            Ntrain += len(file_list)
        Nval = 0
        for key, file_list in self.val_dict.items():
            Nval += len(file_list)

        self.param.train_steps = np.ceil(Ntrain / self.param.batch_size)
        self.param.val_steps = np.ceil(Nval / self.param.batch_size)

        np.random.seed(self.param.seed)
        K.clear_session()
        self.build_network()

    def load_param(self, param_file):
        param_dict = load_param(param_file)
        # Now overwrite any values with the
        for key, value in param_dict.items():
            setattr(self, key, value)

    def build_network(self):
        # Input_shape = (W,H)
        # down1_shape = (W/2, H/2)
        inputs = Input(shape=(None, None, 3))
        down1 = DownBlock(
            inputs,
            filters=self.param.fd1,
            kernel_size=self.param.kd1,
            poolsz=self.param.poolsz,
            dropout=self.param.dropout,
            scope="Down1",
        )
        # down2_shape = (W/4, H/4)
        down2 = DownBlock(
            down1,
            filters=self.param.fd2,
            kernel_size=self.param.kd2,
            poolsz=self.param.poolsz,
            dropout=self.param.dropout,
            scope="Down2",
        )
        # down3_shapes:  (W/8, H/8)
        # down3 = DownBlock(
        #    down2,
        #    filters=self.param.fd3,
        #    kernel_size=self.param.kd3,
        #    dropout=self.param.dropout,
        #    poolsz=self.param.poolsz,
        #    scope="Down3"
        # )

        # middle layers:  mid_shape = (W/8, H/8)
        mid = MidBlock(
            down2,
            filters=self.param.fmid,
            kernel_size=self.param.kmid,
            dropout=self.param.dropout,
            scope="Mid",
        )

        # 1st step up.  up2_shape = (W/4, H/4)
        # up3 = UpBlock(
        #    mid,
        #    filters=self.param.fu3,
        #    kernel_size=self.param.ku3,
        #    dropout=self.param.dropout,
        #    poolsz = self.param.poolsz,
        #    scope="Up3"
        # )
        # up3b = SkipConnection(down2, up3, self.param.fu3, scope="Skip3")

        # 1st step up.  up2_shape = (W/2, H/2)
        up2 = UpBlock(
            mid,
            filters=self.param.fu2,
            kernel_size=self.param.ku2,
            poolsz=self.param.poolsz,
            dropout=self.param.dropout,
            scope="Up2",
        )
        # residual connection for results from upsampling with stuff from step before
        up2b = SkipConnection(down1, up2, self.param.fu2, scope="Skip2")

        # 2nd step up up1_shape = (W,H)
        up1 = UpBlock(
            up2b,
            filters=self.param.fu1,
            kernel_size=self.param.ku1,
            poolsz=self.param.poolsz,
            dropout=self.param.dropout,
            scope="Up1",
        )
        up1b = SkipConnection(inputs, up1, self.param.fu1, scope="Skip1")

        # could have more 2D convolutions here.
        # final 1D convolution  (pixel-wise Dense layer)
        # softmax to make a probability distribution across classes.
        final = Conv2D(
            kernel_size=1,
            filters=self.param.Nclasses_final,
            activation="softmax",
            name="FinalConv",
        )(up1b)

        self.model = Model(inputs=inputs, outputs=final)
        self.model.compile(
            loss=self.pixelwise_crossentropy, optimizer="adam", metrics=[self.IOU]
        )

    def IOU(self, Ytrue, Ypred):
        """
        compute pixelwise cross-entropy across most popular classes example by example.

        Input: Ytrue Tensor (Nbatch, W, H, 3)
               Ypred Tensor (Nbatch, W, H, Nclass)
        """
        # define a custom loss function using the pixel-wise cross entropy.
        raise NotImplementedError()

    def pixelwise_crossentropy(self, Ytrue, Ypred):
        """
        compute pixelwise cross-entropy across most popular classes example by example.

        Input: Ytrue Tensor (Nbatch, W, H, 3)
               Ypred Tensor (Nbatch, W, H, Nclass)
        """
        raise NotImplementedError()

    def _get_X_y_batch(self, ind, size_bucket=0):
        """given list of indices and a corresponding bucket,
        returns a batch for training.
        """
        raise NotImplementedError()

    def _get_bucket_boundaries(self, file_dict):
        self._N_in_bucket = [(key, len(val)) for key, val in file_dict.items()]
        vals = [x[1] for x in self._N_in_bucket]
        # maps those sizes to [0,1) interval to randomly select one
        self._rand_boundary = np.cumsum(vals) / np.sum(vals)

    def get_random_train_bucket(self, file_dict):
        """generator to pick a random allowed_bucket using relative sizes"""
        # list of tuples with key and number in each bucket.
        # Note: opportunity to calculate the rand_boundary once, rather than every iteration.

        r = np.random.random()
        for i, v in enumerate(self._rand_boundary):
            if r < v:
                return self._N_in_bucket[i][0]

    def rand_batch_generator(self, file_dict):
        """Generator to make random batches of data.
        Intended for use in training to endlessly
        generate samples of data.

        dict - dict of list of files to lookup
        """
        # JM: need to allow a validation-fold number to be picked here.
        while True:
            size_bucket = self.get_random_train_bucket(file_dict)
            ind_sub = np.random.choice(
                file_dict[size_bucket], size=self.param.batch_size
            )
            yield self._get_X_y_batch(ind_sub, size_bucket)

    def det_batch_generator(self, file_dict):
        """det_batch_generator
        Load deterministic batch generator.
        Intended for use with inference/prediction
        to deterministically loop over data once.
        """
        Nb = self.param.batch_size
        for bucket, ind_list in file_dict.items():
            i0 = 0
            i1 = Nb
            while i1 < len(ind_list):
                ind_sub = np.arange(i0, i1)
                yield self._get_X_y_batch(ind_sub, bucket)
                i0 = i1
                i1 += Nb
            # Last iter
            ind_sub = np.arange(i0, len(y))
            yield self._get_X_y_batch(ind_sub, bucket)

    def train_network(self):
        """train_network()
        Actually train the keras model.
        Uses random batches from rand_batch_generator and train_dict
        Input: None
        Output: None
        Side-effect: Trains self.model.
        """
        self.model.fit(
            self.rand_batch_generator(self.train_dict),
            epochs=self.param.Nepoch,
            steps_per_epoch=self.param.train_steps,
        )

    def predict_all(self, file_dict):
        """predict(X,y,vec)
        Loop over whole file_dict and predict for all samples (HUGE TASK! DONT DO IT)
        Input: X - np.array with rows of indices
               y  - np.array with value of true class.
               vec - np.array with wordvectors
        Output pred - tuple with losses/metrics over dataset
        """
        Ntot = 0
        for key, val in file_dict.items():
            Ntot += len(val)

        Nsteps = np.ceil(Ntot / self.param.batch_size)

        pred = self.model.predict_generator(
            self.det_batch_generator(file_dict), steps=Nsteps
        )
        return pred

    def predict_afew(self, ind_sub=np.arange(3), size_bucket=0):
        """
        Loop over a few instances and data and predict for all samples.
        Input: file_dict - dict with list of entries to sample from index_df
               ind_sub  - np.array with locations to use
               size_bucket - bucket from file_dict to try using
        Output pred - Output images for that batch  (pixelwise probabilities for that class
               batch - initial raw images
               y - ground truth

        """
        batch, y = self._get_X_y_batch(ind_sub, size_bucket)
        pred = self.model.predict(batch)
        return pred, batch, y

    def evaluate(self):
        """
        Evaluates metrics by iterating over all files in file_dict
        using loss and metrics specified in model.

        Input: file_dict - np.array with rows of indices
        Output eval_scores - tuple with losses/metrics over dataset
        """
        Nsteps = np.ceil(len(y) / self.param.batch_size)
        eval_scores = self.model.evaluate_generator(
            self.det_batch_generator(), steps=Nsteps
        )
        return eval_scores

    def save_model(self):
        """saves whole model in hdf5 format."""
        save_name = self._get_save_name()
        self.model.save(save_name + ".model")
        self.param.save_param(self.param, save_name + ".param")

    def _get_save_name(self):
        save_name = "/".join([self.param.model_dir, self.param.model_name])
        return save_name

    def load_model(self):
        """loads saved model from hdf5"""
        self.param = load_param(self.param_file)
        save_name = self._get_save_name()
        self.model = load_model(save_name)

    def get_most_likely(self, pred: np.ndarray):
        """convert a (width, height, Nclass) array to (width,height,3) array R/G colors."""
        return get_color_from_pred(pred, self.class_lookup)

    def compare_pred_images(self, pred: np.ndarray, y: np.ndarray, num=0):
        plt.figure()
        ax1 = plt.subplot(121)
        plt.imshow(pred[num].astype(int))
        ax1.set_title("Pred")

        ax2 = plt.subplot(122)
        tmp = np.zeros(y[num].shape)
        for i in range(2):
            tmp[:, :, i] = y[num, :, :, i]
        plt.imshow(tmp.astype(int))
        ax2.set_title("True")


class kerasUNetLinear(kerasUNetBase):
    """kerasUNetLinear

    Uses one output layer per class.

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

    def __init__(self, index_df, object_df, param_file="basic.param"):
        super().__init__(index_df, object_df, param_file)

    def get_class_label(self, Y):
        R = Y[:, :, :, 0]
        G = Y[:, :, :, 1]

        ytrue = (R // 10) * 256 + G
        return ytrue

    def _get_X_y_batch(self, ind, size_bucket=0):
        """given list of indices and a corresponding bucket,
        returns a batch for training.
        """
        Wtarget = Wlist[size_bucket + 1]
        Htarget = Hlist[size_bucket + 1]

        Nb = len(ind)
        X = np.zeros([Nb, Wtarget, Htarget, 3])
        y = np.zeros([Nb, Wtarget, Htarget, 3])
        for j, i in enumerate(ind):
            row = self.index.iloc[i]
            X_name = os.path.join(DATA_PATH, row["folder"], row["filename"])
            y_name = X_name[:-4] + "_seg.png"
            X[j] = tfimage.load_img(X_name, target_size=(Wtarget, Htarget))
            y[j] = tfimage.load_img(y_name, target_size=(Wtarget, Htarget))
        return X, y

    def IOU(self, Ytrue, Ypred):
        """
        compute pixelwise cross-entropy across most popular classes example by example.

        Input: Ytrue Tensor (Nbatch, W, H, 3)
               Ypred Tensor (Nbatch, W, H, Nclass)
        """
        # define a custom loss function using the pixel-wise cross entropy.
        # JM: currently bugged and returned zero for all results.  should be similar to pixelwise cross entropy.
        # Nb,W,H,Nc = Ypred.shape
        R = Ytrue[:, :, :, 0]
        G = Ytrue[:, :, :, 1]

        Ytrue_class = (R // 10) * 256 + G
        # get classes_present
        # classes_present = set(K.reshape(ytrue_class,-1))
        # classes_present = tf.unique(K.reshape(ytrue_class,-1))
        cost = 0
        for i, class_label in enumerate(self.class_lookup):
            # get numerical value associated with class cl
            pred_msk = Ypred[:, :, :, i] > 0.5
            # if class_label in classes_present:
            # make logical mask
            label_msk = K.equal(Ytrue_class, class_label)
            label_AND_pred = tf.cast(
                tf.math.logical_and(label_msk, pred_msk), tf.float32
            )
            label_OR_pred = tf.cast(tf.math.logical_or(label_msk, pred_msk), tf.float32)
            iou = K.sum(label_AND_pred) / (K.sum(label_OR_pred) + self.param.eps)
            cost = cost + iou
        cost = cost / self.param.Nclasses
        return cost

    def pixelwise_crossentropy(self, Ytrue, Ypred):
        """
        compute pixelwise cross-entropy across most popular classes example by example.

        Input: Ytrue Tensor (Nbatch, W, H, 3)
               Ypred Tensor (Nbatch, W, H, Nclass)
        """
        # define a custom loss function using the pixel-wise cross entropy.
        # W,H,Nc = tf.shape(Ypred)
        R = Ytrue[:, :, :, 0]
        G = Ytrue[:, :, :, 1]

        ytrue_class = (R // 10) * 256 + G
        # get classes_present
        # JM note: could use float16 to reduce mem?
        shapes = tf.cast(tf.shape(ytrue_class), tf.float32)
        # classes_present = set(K.reshape(ytrue_class,-1))
        cost = 0
        for i, class_label in enumerate(self.class_lookup):
            # get numerical value associated with class cl
            # clip to avoid over/underflow issues from log.
            Ypred_c = K.clip(Ypred[:, :, :, i], self.param.eps, 1 - self.param.eps)
            # if class_label in classes_present:
            # make logical mask
            msk = K.equal(ytrue_class, class_label)
            cost = cost - K.sum(
                tf.boolean_mask(K.log(1 - Ypred_c), tf.math.logical_not(msk))
            )
            cost = cost - K.sum(tf.boolean_mask(K.log(Ypred_c), msk))

            # else:
            #    #find all erroneous classes
            #    cost += K.mean(K.log(1-ypred_c))/(Nclasses)
        shapes = tf.cast(tf.shape(Ypred), tf.float32)
        cost = cost / (self.param.Nclasses * shapes[1] * shapes[2])
        return cost


# should
class kerasUNetTree(kerasUNetBase):
    """kerasUNetLinear

    Uses binary encoding from class label to layer number.
    Cuts down on memory needed.

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

    def __init__(self, index_df, object_df, param_file="basic.param"):
        super().__init__(index_df, object_df, param_file)

    def get_class_label(self, Y):
        R = Y[:, :, :, 0]
        G = Y[:, :, :, 1]

        ytrue = (R // 10) * 256 + G
        return ytrue

    def _get_X_y_batch(self, ind, size_bucket=0):
        """given list of indices and a corresponding bucket,
        returns a batch for training.
        """
        Wtarget = Wlist[size_bucket + 1]
        Htarget = Hlist[size_bucket + 1]

        Nb = len(ind)
        X = np.zeros([Nb, Wtarget, Htarget, 3])
        y = np.zeros([Nb, Wtarget, Htarget, 3])
        for j, i in enumerate(ind):
            row = self.index.iloc[i]
            X_name = os.path.join(DATA_PATH, row["folder"], row["filename"])
            y_name = X_name[:-4] + "_seg.png"
            X[j] = tfimage.load_img(X_name, target_size=(Wtarget, Htarget))
            y[j] = tfimage.load_img(y_name, target_size=(Wtarget, Htarget))
        return X, y

    def IOU(self, Ytrue, Ypred):
        """
        compute pixelwise cross-entropy across most popular classes example by example.

        Input: Ytrue Tensor (Nbatch, W, H, 3)
               Ypred Tensor (Nbatch, W, H, Nclass)
        """
        # define a custom loss function using the pixel-wise cross entropy.
        # JM: currently bugged and returned zero for all results.  should be similar to pixelwise cross entropy.
        # Nb,W,H,Nc = Ypred.shape
        Ytrue_class = self.get_class_label(Ytrue)
        # get classes_present
        # classes_present = set(K.reshape(ytrue_class,-1))
        # classes_present = tf.unique(K.reshape(ytrue_class,-1))

        # np.unpackbits(np.array([[0,1,4,7]],np.uint8)).reshape(-1,8)[:,4:]
        cost = 0
        for i, class_label in enumerate(self.class_lookup):
            # get numerical value associated with class cl
            pred_msk = Ypred[:, :, :, i] > 0.1
            # if class_label in classes_present:
            # make logical mask
            label_msk = K.equal(Ytrue_class, class_label)
            label_AND_pred = tf.cast(
                tf.math.logical_and(label_msk, pred_msk), tf.float32
            )
            label_OR_pred = tf.cast(tf.math.logical_or(label_msk, pred_msk), tf.float32)
            iou = K.sum(label_AND_pred) / (K.sum(label_OR_pred) + self.param.eps)
            cost = cost + iou
        cost = cost / self.param.Nclasses
        return cost

    def pixelwise_crossentropy(self, Ytrue, Ypred):
        """
        compute pixelwise cross-entropy across most popular classes example by example.

        Input: Ytrue Tensor (Nbatch, W, H, 3)
               Ypred Tensor (Nbatch, W, H, Nclass)
        """
        # define a custom loss function using the pixel-wise cross entropy.
        # W,H,Nc = tf.shape(Ypred)
        ytrue_class = self.get_class_label(Ytrue)
        # get classes_present
        # JM note: could use float16 to reduce mem?
        shapes = tf.cast(tf.shape(ytrue_class), tf.float32)
        # classes_present = set(K.reshape(ytrue_class,-1))
        cost = 0
        for i, class_label in enumerate(self.class_lookup):
            # get numerical value associated with class cl
            # clip to avoid over/underflow issues from log.
            Ypred_c = K.clip(Ypred[:, :, :, i], self.param.eps, 1 - self.param.eps)
            # if class_label in classes_present:
            # make logical mask
            msk = K.equal(ytrue_class, class_label)
            cost = cost - K.sum(
                tf.boolean_mask(K.log(1 - Ypred_c), tf.math.logical_not(msk))
            )
            cost = cost - K.sum(tf.boolean_mask(K.log(Ypred_c), msk))

        shapes = tf.cast(tf.shape(Ypred), tf.float32)
        cost = cost / (self.param.Nclasses * shapes[1] * shapes[2])
        return cost


class BinNetwork(object):
    def __init__(self, Nmax, Ntrain):
        pass

    def build_network(self):
        pass

    def make_data(self, N):
        pass

    def train_network(self, data):
        pass


# make a log-tree approach to handle many more classes.  just assigned buckets based on label.
# could use a semantic similarity notion to put in related classes.

if __name__ == "__main__":
    index_df, object_df = load_and_clean_indices()
    UNet = kerasUNetLinear(index_df, object_df)
    UNet.train_network()
    pred, batch, y = UNet.predict_afew(range(5))
    pred2 = UNet.get_most_likely(pred)
    UNet.compare_pred_images(pred2, y, 0)
