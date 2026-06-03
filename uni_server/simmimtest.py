import data_totalsegmentator as tseg
import time
import torch
from torch.utils.data import DataLoader
import data_totalsegmentator as tseg
import models
import numpy as np




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
#test
images = next(islice(iter(test_loader), 305, None))

images = torch.nn.functional.interpolate(images, size=(448, 448), mode='bilinear')

print("Image batch shape:", images.shape)

construct_end = time.time()
print(f"Construct runtime {construct_end - construct_start}")

device = torch.device("cuda")
#model = models.SwinUnetSimMIM()
model = models.SwinUnetSimMIMNewSize(img_size=448)
#model = models.SwinUnetSimMIMImagenet()
model.to(device)
model.load_state_dict(torch.load("/home/u372291/CODE/thesis/models/pretrain_simmim_448.pth", weights_only=True, map_location=device), strict=False)



output, inv_mask = model(images.to(device))
mask = 1 - inv_mask

print("Output shape:", output.shape)
print("Mask shape:", mask.shape)

images = images.squeeze().to(device)
output = output.squeeze().to(device)
mask = mask.squeeze().to(device)
inv_mask = inv_mask.squeeze().to(device)

print("Squeezed output shape:", output.shape)
print("Squeezed mask shape:", mask.shape)
print("Squeezed image shape:", images.shape)
print("Squeezed inv_mask shape:", inv_mask.shape)

masked_image = images * mask
masked_output = output * inv_mask
final_image = masked_image + masked_output

#test = torch.zeros(224, 224)
#test[0:112, 0:112] = 1

def normalize(img):
    img = (img - img.min()) / (img.max() - img.min())
    return img

from matplotlib import pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(5, 14))
fig.suptitle("SimMIM 448x448")

axes[0].axis('off')
axes[1].axis('off')
axes[2].axis('off')

#masked_image = normalize(masked_image)
#final_image = normalize(final_image)

axes[0].imshow(np.flipud(images.cpu().detach().numpy().T), cmap='gray')
axes[1].imshow(np.flipud(masked_image.cpu().detach().numpy().T), cmap='gray')
axes[2].imshow(np.flipud(final_image.cpu().detach().numpy().T), cmap='gray')

plt.savefig("/home/u372291/CODE/thesis/images/simmim_448_output.png")
