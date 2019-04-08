import subprocess
from os import walk
import re

import imageio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Hlist=[0,240,480,1200,2400,10000]
Wlist=[0,320,640,1600,3200,20000]
Nbucket=len(Hlist)-1
    
def get_img_size(path_str):
    """get tuple for width, height of image"""
    cmd_list=['identify', '-format', '(%w,%h)', "{}".format(path_str)]
    size_str=subprocess.run(cmd_list,stdout=subprocess.PIPE)
    size_tuple=eval(size_str.stdout.decode('utf-8'))
    return size_tuple

def make_img_index(dir_str):
    #make a glob of files.
    count=0
    indx_list=[]
    for root, dir, files in walk(dir_str):
        for file in files:
            if 'jpg' in file:
                #strip tag and get seg_png.
                png_file=file[:-4]+'_seg.png'
                name='{}/{}'.format(root,png_file)
                #get image_number
                img_num=int(re.search('[0-9]+',file).group(0))
                #grab last sub-directory to treat as image class.                
                img_class=root.split('/')[-1]
                img_size=get_img_size(name)
                #get the number as 
                indx_list.append([img_num,img_class,img_size,root,file])
                if len(indx_list)%100 == 0:
                    print(len(indx_list))
    return indx_list

def make_indx_df(indx_list):
    indx_df=pd.DataFrame(indx_list,columns=['num','class','size','dir','fname'])
    return indx_df

def augment_indx_df(indx_df):
    """augments the dataframe with bucketing based on sizes, and a random k-fold.
    Buckets according to smallest frame sie that will contain the image.
    Buckets scale according to factor of 2, with approximately equal contents.  
    """
    #add sizing batches
    Nexample = len(indx_df)
    msk_arr = np.zeros((Nexample,Nbucket))

    width = indx_df['size'].apply(lambda x: eval(x)[0])
    height = indx_df['size'].apply(lambda x: eval(x)[1])

    indx_df['width'] = width
    indx_df['height'] = height

    if (np.max(width)> Wlist[-1]) | (np.max(height)> Hlist[-1]):
        raise Exception('Image size larger than largest maximum bucket!')
    
    for i in range(Nbucket):
        print("Iter {}, Width bounds: {}, {}, Height bounds: {}, {}".format(
            i,Wlist[i],Wlist[i+1],Hlist[i],Hlist[i+1]))
        #Width upper/lower bounds
        box0 = (width <= Wlist[i]) & (height <= Hlist[i])
        box1 = (width <= Wlist[i+1]) & (height <= Hlist[i+1])
        msk_arr[:,i] = np.logical_and(~box0, box1)
        
    msk_arr=msk_arr.astype(bool)
    indx_df['size_bucket']=np.zeros(Nexample)
    for i in range(Nbucket):
        indx_df.loc[msk_arr[:,i],'size_bucket']=i
    #add random label for subsetting.
    indx_df['fold']=np.random.randint(low=0,high=5,size=Nexample)
    return msk_arr

def plot_sizes(indx_df,bucket=0):
    msk=indx_df['size_bucket']==bucket
    w=indx_df.loc[msk]['width']
    h=indx_df.loc[msk]['height']
    plt.plot(w,h,'.')
    plt.show()

def plot_counts(indx_df,bucket=None):
    if bucket==None:
        msk = [True]*len(indx_df)
    else:
        msk=indx_df['size_bucket']==bucket
    sub_df=indx_df.loc[msk]
    class_counts=sub_df.groupby('class').apply(len)
    class_counts=class_counts.sort_values(ascending=False)
    class_counts.iloc[:20].plot.barh()
    plt.show()
    return class_counts

def load_img_indx_df(index_str='index/train_indx.csv'):
    """Load Python DataFrame with images including path, size.
    Returns pandas DataFrame.
    """
    df=pd.read_csv(index_str)
    return df

def load_object_index_df(file_name='index/ADE20K_index_object.csv'):
    df=pd.read_csv(file_name)
    return df.iloc[:,:5]


def save_img_indx_df(df,index_str='index/train_indx.csv'):
    df.to_csv(index_str,index=False)
    return None


def get_image(indx_df,num=0):
    img_dir=indx_df.loc[num,'dir']
    img_name=indx_df.loc[num,'fname']
    jpeg_name='/'.join([img_dir,img_name])
    png_name= jpeg_name[:-4]+'_seg.png'
    img_jpg=imageio.imread(jpeg_name)
    img_png=imageio.imread(png_name)    
    return img_jpg, img_png

def get_classes(img):
    """compute mapping from R/G to class number used in object_df"""
    classes=256.0*(img[:,:,0])//10  + img[:,:,1]
    return classes


def get_training_dicts(indx_df,size_buckets=[0,1],val_fold=0):
    """
    given desired buckets (0,1,2,3,4) returns a dict of lists
    of filenames for each bucket using indx_df
    Separates out val_fold to use for validation.

    Input: indx_df - pandas dataframe with file info, and size_buckets and fold numbers

    Return train_files: dict of lists with indices in indx_df to train on 
           val_files: dict of lists with indices in indx_df for validation
 
    """
    train_ind=dict()
    val_ind=dict()
    val_msk = indx_df['fold']==val_fold
    irange=np.arange(len(indx_df))
    #irange = indx_df.index
    for size in set(size_buckets):
        size_msk = (indx_df['size_bucket']==size)
        train_ind[size] = irange[size_msk & (~val_msk)]
        val_ind[size] = irange[size_msk & (val_msk)]
    return train_ind, val_ind
    
def get_common_class_indx(object_df,ispart_frac=0.5,Nclass=50):
    #try to consider big features first.  
    msk= object_df['proportionClassIsPart'] < ispart_frac
    ind=object_df[msk]['Index'].values
    return ind[:Nclass]

def get_counts_match(object_df,string,n=10):
    """gets rows where an exact match occurs in the name
    e.g. string=sea will not match seat or sea boat.
    """
    msk=object_df['objectnames'].str.match('^{}$'.format(string))
    return object_df[msk].iloc[:n]

#get counts of objects in images.

#look at most popular classes.

if __name__=="__main__":
    indx_df=load_img_indx_df()
    augment_indx_df(indx_df)
    indx_df.sort_values(['fold','size_bucket'],inplace=True)
    #why? why is this a special separate function? who thought this was a good idea?
    indx_df.reset_index(inplace=True)
    object_df=load_object_index_df()
