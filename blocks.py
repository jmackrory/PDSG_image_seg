import keras.backend as K 
from keras.layers import Input, Dropout, Dense, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers import LeakyReLU, Add




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
        m = MaxPooling2D(name=scope+'_MaxPool')(r)
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
        m = UpSampling2D(name=scope+'_Upscale')(r)
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
