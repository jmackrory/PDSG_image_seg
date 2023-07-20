import os
import subprocess
from os import walk
import re

import imageio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Hlist = [0, 240, 480, 1200, 2400, 10000]
Wlist = [0, 320, 640, 1600, 3200, 20000]
Nbucket = len(Hlist) - 1

INDEX_PATH = "/tf/data/index"
IMG_INDEX_FILE = "ADE20K_img_index_mk2.tsv"
OBJ_INDEX_FILE = "ADE20K_obj_index_mk2.tsv"
DATA_PATH = "/tf/data/ADE20K_2016_07_26"

# def get_img_size(path_str):
#     """get tuple for width, height of image"""
#     cmd_list=['identify', '-format', '(%w,%h)', "{}".format(path_str)]
#     size_str=subprocess.run(cmd_list,stdout=subprocess.PIPE)
#     size_tuple=eval(size_str.stdout.decode('utf-8'))
#     return size_tuple

# def make_img_index(dir_str):
#     #make a glob of files.
#     count=0
#     index_list=[]
#     for root, dir, files in walk(dir_str):
#         for file in files:
#             if 'jpg' in file:
#                 #strip tag and get seg_png.
#                 png_file=file[:-4]+'_seg.png'
#                 name='{}/{}'.format(root,png_file)
#                 #get image_number
#                 img_num=int(re.search('[0-9]+',file).group(0))
#                 #grab last sub-directory to treat as image class.
#                 img_class=root.split('/')[-1]
#                 img_size=get_img_size(name)
#                 #get the number as
#                 index_list.append([img_num,img_class,img_size,root,file])
#                 if len(index_list)%100 == 0:
#                     print(len(index_list))
#     return index_list

# def make_index_df(index_list):
#     index_df=pd.DataFrame(index_list,columns=['num','class','size','dir','fname'])
#     return index_df


# load new img index file.
# need to adjust augment file to new format.
def load_img_index_df(index_str=IMG_INDEX_FILE):
    index_path = os.path.join(INDEX_PATH, index_str)
    index_df = pd.read_csv(index_path, sep="\t")
    return index_df


def load_object_index_df(index_str=OBJ_INDEX_FILE):
    index_path = os.path.join(INDEX_PATH, index_str)
    object_df = pd.read_csv(index_path, sep="\t")
    object_df.sort_values("objectCount", inplace=True)
    return object_df


def test_load(path="index/ADE20K_obj_index_mk2.tsv"):
    with open(path, "r") as f:
        lines = f.readlines()
        split_line = [x.split("\t") for x in lines]
    return split_line[:20]


# load new object file.
# augment as required and considered changed names.


def augment_index_df(index_df):
    """augments the dataframe with bucketing based on sizes, and a random k-fold.
    Buckets according to smallest frame sie that will contain the image.
    Buckets scale according to factor of 2, with approximately equal contents.
    """
    # add sizing batches
    Nexample = len(index_df)
    msk_arr = np.zeros((Nexample, Nbucket))

    # width = index_df['size'].apply(lambda x: eval(x)[0])
    # height = index_df['size'].apply(lambda x: eval(x)[1])
    width = index_df["width"]
    height = index_df["height"]

    if (np.max(width) > Wlist[-1]) | (np.max(height) > Hlist[-1]):
        raise Exception("Image size larger than largest maximum bucket!")

    for i in range(Nbucket):
        print(
            "Iter {}, Width bounds: {}, {}, Height bounds: {}, {}".format(
                i, Wlist[i], Wlist[i + 1], Hlist[i], Hlist[i + 1]
            )
        )
        # Width upper/lower bounds
        box0 = (width <= Wlist[i]) & (height <= Hlist[i])
        box1 = (width <= Wlist[i + 1]) & (height <= Hlist[i + 1])
        msk_arr[:, i] = np.logical_and(~box0, box1)

    msk_arr = msk_arr.astype(bool)
    index_df["size_bucket"] = np.zeros(Nexample)
    for i in range(Nbucket):
        index_df.loc[msk_arr[:, i], "size_bucket"] = i
    # add random label for subsetting.
    index_df["fold"] = np.random.randint(low=0, high=5, size=Nexample)
    return msk_arr


def plot_sizes(index_df, bucket=0):
    msk = index_df["size_bucket"] == bucket
    w = index_df.loc[msk]["width"]
    h = index_df.loc[msk]["height"]
    plt.plot(w, h, ".")
    plt.show()


def plot_counts(index_df, bucket=None):
    if bucket == None:
        msk = [True] * len(index_df)
    else:
        msk = index_df["size_bucket"] == bucket
    sub_df = index_df.loc[msk]
    class_counts = sub_df.groupby("class").apply(len)
    class_counts = class_counts.sort_values(ascending=False)
    class_counts.iloc[:20].plot.barh()
    plt.show()
    return class_counts


def save_img_index_df(df, index_str="index/train_index.csv"):
    df.to_csv(index_str, index=False)
    return None


def get_image(index_df, num=0):
    img_dir = index_df.loc[num, "folder"]
    img_name = index_df.loc[num, "filename"]
    jpeg_name = "/".join([img_dir, img_name])
    png_name = jpeg_name[:-4] + "_seg.png"
    img_jpg = imageio.imread(jpeg_name)
    img_png = imageio.imread(png_name)
    return img_jpg, img_png


def get_classes(img):
    """compute mapping from R/G to class number used in object_df
    class = 256*R/10 + G
    """
    classes = 256.0 * (img[:, :, 0]) // 10 + img[:, :, 1]
    return classes


def get_color_from_pred(img, lookup_dict):
    """
    untested function to convert array of outputs
    to most likely classes.

    Should test via converting image to labels, then
    using this to convert back.
    """
    # need layer indexes
    indx_arg = np.argmax(img, axis=-1)
    vals = np.unique(indx_arg)

    Nbatch, width, height, Nclass = img.shape
    out_img = np.zeros((Nbatch, width, height, 3))

    for val in vals:
        msk = indx_arg == val
        # find actual class corresponding to local index
        cls = lookup_dict[val]

        G = cls % 256
        R = (cls // 256) * 10
        out_img[msk, 0] = R
        out_img[msk, 1] = G
    # class = 256*R/10 + G
    # to avoid collisions, G must be mod 256
    return out_img


def get_training_dicts(index_df, size_buckets=[0, 1], val_fold=0):
    """
    given desired buckets (0,1,2,3,4) returns a dict of lists
    of filenames for each bucket using index_df
    Separates out val_fold to use for validation.

    Input: index_df - pandas dataframe with file info, and size_buckets and fold numbers

    Return train_files: dict of lists with indices in index_df to train on
           val_files: dict of lists with indices in index_df for validation

    """
    train_ind = dict()
    val_ind = dict()
    val_msk = index_df["fold"] == val_fold
    irange = np.arange(len(index_df))
    # irange = index_df.index
    for size in set(size_buckets):
        size_msk = index_df["size_bucket"] == size
        train_ind[size] = irange[size_msk & (~val_msk)]
        val_ind[size] = irange[size_msk & (val_msk)]
    return train_ind, val_ind


def get_common_class_index(object_df, ispart_frac=0.5, Nclasses=50):
    # try to consider big features first.
    msk = object_df["proportionClassIsPart"] < ispart_frac
    ind = object_df[msk]["index"].values
    return ind[:Nclasses]


def get_counts_match(object_df, string, n=10):
    """gets rows where an exact match occurs in the name
    e.g. string=sea will not match seat or sea boat.
    """
    msk = object_df["objectnames"].str.match("^{}$".format(string))
    return object_df[msk].iloc[:n]


def load_and_clean_indices():
    index_df = load_img_index_df()
    augment_index_df(index_df)
    index_df.sort_values(["fold", "size_bucket"], inplace=True)
    # why? why is this a special separate function? who thought this was a good idea?
    index_df.reset_index(inplace=True)
    object_df = load_object_index_df()
    return index_df, object_df


if __name__ == "__main__":
    index_df, object_df = load_and_clean_indices()
