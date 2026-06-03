import data_totalsegmentator as tseg
import time
import torch
from torch.utils.data import DataLoader


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
batch_size = 24

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

#test
#images = next(iter(train_loader))
#print("Image batch shape:", images.shape)

construct_end = time.time()
print(f"Construct runtime {construct_end - construct_start}")




#training and validation loop

import torch
import torch.nn as nn
import torch.optim as optim
import models
import utils
from matplotlib import pyplot as plt
#import numpy as np
from timm.scheduler.cosine_lr import CosineLRScheduler

custom_image_size = 448

device = torch.device("cuda")
#model = models.SwinUnetSimMIM()
#model = models.SwinUnetSimMIMImagenet()
model = models.SwinUnetSimMIMNewSize(img_size=custom_image_size)
model.to(device)
#model.load_state_dict(torch.load("/home/u372291/CODE/thesis/models/swin_tiny_patch4_window7_224.pth", weights_only=True), strict=False)

criterion = nn.L1Loss(reduction='sum')
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)
num_epochs = 100
early_stopping = utils.EarlyStopping(patience=10, delta=0.001)

###

"""
Z. Xie et al., “SimMIM: A simple framework for masked image modeling,” 
in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 9653–9663.
https://github.com/microsoft/SimMIM/tree/main
"""

iterations_per_epoch = len(train_loader)
num_steps = num_epochs * iterations_per_epoch
warmup_epochs = 10
warmup_steps = warmup_epochs * iterations_per_epoch

scheduler = CosineLRScheduler(
                optimizer,
                t_initial=num_steps, 
                lr_min=5e-6,
                warmup_lr_init=5e-7,
                warmup_t=warmup_steps,
                cycle_limit=1,
                t_in_epochs=False,
            )

###

loss_save_path = "/home/u372291/CODE/thesis/loss/pretrain_simmim_448_loss.png"
model_save_path = "/home/u372291/CODE/thesis/models/pretrain_simmim_448.pth"
plot_title = "SimMIM 448 Pretraining Loss"




start_train = time.time()

model.to(device)

train_loss_list = []
val_loss_list = []

#training loop
for epoch in range(num_epochs):

    epoch_start = time.time()

    #training
    train_loss = 0.0
    model.train()

    for idx, (images) in enumerate(train_loader):

        images = images.to(device)

        #only remove comment for custom size
        images = torch.nn.functional.interpolate(images, size=(custom_image_size, custom_image_size), mode='bilinear')

        optimizer.zero_grad()
        outputs, inv_mask = model(images)
        #total number of pixels in the masks
        loss_num_pixels = round(model.mask_ratio * ((images.size(2) / model.mask_size)) ** 2) * (model.mask_size ** 2)
        #loss only on masked pixels and masked pixel predictions
        loss = criterion((outputs * inv_mask), (images * inv_mask)) / loss_num_pixels
        loss.backward()
        optimizer.step()
        scheduler.step_update(epoch * iterations_per_epoch + idx)
        
        #balance the loss per batch size
        train_loss += loss.item() * images.size(0)
    

    train_loss /= len(train_loader.dataset)

    train_loss_list.append(train_loss)

    #validation
    val_loss = 0.0
    model.eval()
    
    with torch.no_grad():

        for images in val_loader:

            images = images.to(device)

            #only remove comment for custom size
            images = torch.nn.functional.interpolate(images, size=(custom_image_size, custom_image_size), mode='bilinear')

            outputs, inv_mask = model(images)
            #total number of pixels in the masks
            loss_num_pixels = round(model.mask_ratio * ((images.size(2) / model.mask_size)) ** 2) * (model.mask_size ** 2)
            #loss only on masked pixels and masked pixel predictions
            loss = criterion((outputs * inv_mask), (images * inv_mask)) / loss_num_pixels
            
            #balance the loss per batch size
            val_loss += loss.item() * images.size(0)
        
    val_loss /= len(val_loader.dataset)

    val_loss_list.append(val_loss)

    #early_stopping.check(val_loss)
    #if early_stopping.stop_training:
        #break
    
    epoch_end = time.time()

    print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Runtime: {epoch_end - epoch_start}")


end_train = time.time()

print(f"Train runtime: {end_train - start_train}")

#plot training and validation loss
plt.figure()
plt.plot(train_loss_list, label='training loss')
plt.plot(val_loss_list,label='validation loss')
plt.title(plot_title)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(loss_save_path)
print(loss_save_path)

#save the model
torch.save(model.state_dict(), model_save_path)

end = time.time()
print(f"Runtime: {end - start}")