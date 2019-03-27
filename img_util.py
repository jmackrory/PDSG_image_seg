import subprocess
from os import walk
import re

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
        
        # width_cond0 =  (Wlist[i] < width )
        # width_cond1 = (width  <= Wlist[i+1])
        # #Height upper/lower bounds
        # height_cond0 = (Hlist[i] < height)
        # height_cond1 = (height <= Hlist[i+1])
        # #find width_strip 
        # width_cond = (width_cond0 & width_cond1 & height_cond1)
        # #find height_strip 
        # height_cond = (height_cond0 & height_cond1 & width_cond1)
        # msk_arr[:,i] = (height_cond | width_cond)
        # print('wtf bro?')
    msk_arr=msk_arr.astype(bool)
    indx_df['size_bucket']=np.zeros(Nexample)
    for i in range(Nbucket):
        indx_df.loc[msk_arr[:,i],'size_bucket']=i
    #add random label for subsetting.
    print('why?')
    indx_df['Fold']=np.random.randint(low=0,high=5,size=Nexample)
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

def save_img_indx_df(df,index_str='index/train_indx.csv'):
    df.to_csv(indx_df,index=False)


#get counts of objects in images.

#look at most popular classes.


    
