import data_totalsegmentator as tseg
import time
import torch
from torch.utils.data import DataLoader
import data_totalsegmentator as tseg
import numpy as np
import nibabel as nib
import os


# img_path = "/home/mleeuwen/DATA/TSv3_Selection/Images/"

# original_1 = nib.load(os.path.join(img_path, "Scan_0065.nii"))
# original_2 = nib.load(os.path.join(img_path, "Scan_0076.nii"))
# original_3 = nib.load(os.path.join(img_path, "Scan_0091.nii"))

# np_original_1 = np.asanyarray(original_1.dataobj, dtype=np.float32)
# np_original_2 = np.asanyarray(original_2.dataobj, dtype=np.float32)
# np_original_3 = np.asanyarray(original_3.dataobj, dtype=np.float32)

# print("Original image shapes:", np_original_1.shape, np_original_2.shape, np_original_3.shape)



start = time.time()

#dataset and data loader

construct_start = time.time()

#constructing the datasets
train_img_dir = "/home/u372291/CODE/data/TotalSegmentator/images/train/"
train_set = tseg.TSegDataset(train_img_dir)

val_img_dir = "/home/u372291/CODE/data/TotalSegmentator/images/validation/"
val_set = tseg.TSegDataset(val_img_dir)

test_img_dir = "/home/u372291/CODE/data/TotalSegmentator/images/test/"
test_set = tseg.TSegDataset(test_img_dir)

#data loaders
batch_size = 1

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

from itertools import islice

image_list = []
slice_list = [752, 1670, 2500, 2900, 5250, 7840]

for slice in slice_list:
    images = next(islice(iter(test_loader), slice, None))
    images = images.squeeze()
    image_list.append(images)


construct_end = time.time()
print(f"Construct runtime {construct_end - construct_start}")


print("Squeezed image shape:", images.shape)


from matplotlib import pyplot as plt


# fig, axes = plt.subplots(1, 3, figsize=(14, 5))
# #fig.suptitle("Total Segmentator dataset slices")

# axes[0].axis('off')
# axes[1].axis('off')
# axes[2].axis('off')

# axes[0].imshow(np.flipud(np_original_1[:,:,24].T), cmap='gray')
# axes[1].imshow(np.flipud(np_original_2[:,:,389].T), cmap='gray')
# axes[2].imshow(np.flipud(np_original_3[:,:,200].T), cmap='gray')

# plt.tight_layout()
# plt.savefig("/home/u372291/CODE/thesis/images/total_segmentator_slices_original.png")


fig, axes = plt.subplots(2, 3, figsize=(14, 10))
#fig.suptitle("Total Segmentator dataset slices")

axes[0, 0].axis('off')
axes[0, 1].axis('off')
axes[0, 2].axis('off')
axes[1, 0].axis('off')
axes[1, 1].axis('off')
axes[1, 2].axis('off')

axes[0, 0].imshow(np.flipud(image_list[1].detach().numpy().T), cmap='gray')
axes[0, 1].imshow(np.flipud(image_list[0].detach().numpy().T), cmap='gray')
axes[0, 2].imshow(np.flipud(image_list[2].detach().numpy().T), cmap='gray')
axes[1, 0].imshow(np.flipud(image_list[3].detach().numpy().T), cmap='gray')
axes[1, 1].imshow(np.flipud(image_list[4].detach().numpy().T), cmap='gray')
axes[1, 2].imshow(np.flipud(image_list[5].detach().numpy().T), cmap='gray')

fig.tight_layout()

plt.savefig("/home/u372291/CODE/thesis/images/total_segmentator_slices3.png")



# images = next(islice(iter(test_loader), 3460, None))
# images = images.squeeze()

# plt.figure()
# plt.axis('off')
# plt.imshow(np.flipud(images.detach().numpy().T), cmap='gray')
# plt.savefig("/home/u372291/CODE/thesis/images/total_segmentator_slice_diagram.png")
