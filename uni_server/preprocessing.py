import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from utils import apply_windowing
from random import randint
#import matplotlib.pyplot as plt


#reshaping the image to 224
#input an image where dimension 1 and 2 are height and width. output the image reshaped to 224x224xN
def reshape(img):

    new_size = 224
    height = img.shape[0]
    width = img.shape[1]
    padding_value = img.min()

    #cropping anything larger than 224
    if height > new_size:
        img = img[(height//2)-(new_size//2):(height//2)+(new_size//2) ,: ,:]

    if width > new_size:
        img = img[:, (width//2)-(new_size//2):(width//2)+(new_size//2) ,:]

    #padding with minimum value if smaller than 224
    if height < new_size:

        pad_top = (new_size//2)-(height//2)
        pad_bottom = (new_size)-(height) - pad_top
        img = np.pad(img, ( (pad_top,pad_bottom), (0,0), (0,0) ), mode='constant', constant_values=padding_value)

    if width < new_size:

        pad_left = (new_size//2)-(width//2)
        pad_right = (new_size)-(width) - pad_left
        img = np.pad(img, ( (0,0), (pad_left,pad_right), (0,0) ), mode='constant', constant_values=padding_value)

    return img

#normalizing
#input an image and output normalized image from 0 to 1
def normalize(img):
    img = (img - img.min()) / (img.max() - img.min())
    return img

#generating the slices and saving them as numpy arrays
#input the file list, the path of original images and the save path
def dataset_generator(img_files, label_files, img_path, label_path, img_save_path, label_save_path):
    for nscan, (img_file, label_file) in enumerate(zip(img_files, label_files)):

        img = nib.load(os.path.join(img_path, img_file))
        label = nib.load(os.path.join(label_path, label_file))

        np_img = np.asanyarray(img.dataobj, dtype=np.float32)
        np_img = reshape(np_img)

        np_label = np.asanyarray(label.dataobj, dtype=np.float32)
        np_label = reshape(np_label)
        
        np_img = normalize(np_img)
        
        np_label = np_label>0
        np_label = np_label.astype(np.float32)

        #print(np_img.shape)

        if randint(0, 1) == 0:
            np_img = np.flip(np_img, axis=0)
            np_label = np.flip(np_label, axis=0)
        if randint(0, 1) == 0:
            np_img = np.flip(np_img, axis=1)
            np_label = np.flip(np_label, axis=1)

        for nslice in range(np_img.shape[2]):
            np.save(f"{img_save_path}image_{nscan+1}_{nslice+1}.npy", np_img[:,:,nslice])
            break
        for nslice in range(np_label.shape[2]):
            np.save(f"{label_save_path}image_{nscan+1}_{nslice+1}.npy", np_label[:,:,nslice])
            break


# #SBATCH -w ultramarine
img_path = "/home/mleeuwen/DATA/TSv3_Selection/Images/"
label_path = "/home/mleeuwen/DATA/TSv3_Selection/Labels/"

img_files = os.listdir(img_path)
label_files = os.listdir(label_path)
img_files.sort()
label_files.sort()

#splitting
img_train, img_val_test, label_train, label_val_test = train_test_split(img_files, label_files, test_size=0.3, random_state=42)
img_val, img_test, label_val, label_test = train_test_split(img_val_test, label_val_test, test_size=0.5, random_state=42)
print(len(img_train), len(img_val), len(img_test))
print(len(label_train), len(label_val), len(label_test))

#making datasets
dataset_generator(img_train, label_train, img_path, label_path, "/home/u372291/CODE/data/TotalSegmentator/images/train/", "/home/u372291/CODE/data/TotalSegmentator/labels/train/")
dataset_generator(img_val, label_val, img_path, label_path, "/home/u372291/CODE/data/TotalSegmentator/images/validation/", "/home/u372291/CODE/data/TotalSegmentator/labels/validation/")
dataset_generator(img_test, label_test, img_path, label_path, "/home/u372291/CODE/data/TotalSegmentator/images/test/", "/home/u372291/CODE/data/TotalSegmentator/labels/test/")

print('Done')
