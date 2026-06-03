import data_bonesegmentation as tseg
import time
import torch
from torch.utils.data import DataLoader


start = time.time()

#dataset and data loader

construct_start = time.time()

#constructing the datasets
train_img_dir = "/home/u372291/CODE/data/TotalSegmentator/images/train/"
train_mask_dir = "/home/u372291/CODE/data/TotalSegmentator/labels/train/"
train_set = tseg.TSegDataset(train_img_dir, train_mask_dir)

val_img_dir = "/home/u372291/CODE/data/TotalSegmentator/images/validation/"
val_mask_dir = "/home/u372291/CODE/data/TotalSegmentator/labels/validation/"
val_set = tseg.TSegDataset(val_img_dir, val_mask_dir)

test_img_dir = "/home/u372291/CODE/data/TotalSegmentator/images/test/"
test_mask_dir = "/home/u372291/CODE/data/TotalSegmentator/labels/test/"
test_set = tseg.TSegDataset(test_img_dir, test_mask_dir)

#data loaders
batch_size = 24

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# #test
# images, masks = next(iter(train_loader))
# print("Image batch shape:", images.shape)
# print("Mask batch shape:", masks.shape)

construct_end = time.time()
print(f"Construct runtime {construct_end - construct_start}")

#training and validation loop

import torch
import torch.nn as nn
import torch.optim as optim
import utils
import models
from matplotlib import pyplot as plt

device = torch.device("cuda")
model = models.SwinUnetFinetune()
#model = models.SwinUnetFinetuneImagenet()
model.to(device)
model.load_state_dict(torch.load("/home/u372291/CODE/thesis/models/pretrain_simmim_2.pth", weights_only=True, map_location=device), strict=False)
#model.load_state_dict(torch.load("/home/u372291/CODE/thesis/models/swin_tiny_patch4_window7_224.pth", weights_only=True, map_location=device), strict=False)

base_lr = 0.01
criterion = utils.ComboLoss(alpha=0.6)
optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
num_epochs = 150
early_stopping = utils.EarlyStopping(patience=10, delta=0.001)

loss_save_path = "/home/u372291/CODE/thesis/loss/pretraining_simmim_finetune_bonesegmentation_loss.png"
best_model_save_path = "/home/u372291/CODE/thesis/models/pretraining_simmim_finetune_bonesegmentation.pth"
final_model_save_path = "/home/u372291/CODE/thesis/models/pretraining_simmim_finetune_bonesegmentation_(final).pth"
plot_title = "SimMIM pretraining bone segmentation finetuning loss"

iterations_per_epoch = len(train_loader)
num_steps = num_epochs * iterations_per_epoch
step_num = 0

start_train = time.time()

model.to(device)

train_loss_list = []
val_loss_list = []

best_val_dice = 0

#training loop
for epoch in range(num_epochs):

    epoch_start = time.time()

    #training
    train_loss = 0.0
    model.train()

    for idx, (images, masks) in enumerate(train_loader):

        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        #scheduler.step_update(epoch * iterations_per_epoch + idx)

        ###

        '''Cao, H., Wang, Y., Chen, J., Jiang, D., Zhang, X., Tian, Q., & Wang, M. (2021). 
        Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation (arXiv:2105.05537). arXiv.
        https://doi.org/10.48550/arXiv.2105.05537
        '''

        lr_ = base_lr * (1.0 - (step_num / num_steps)) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
        step_num = step_num + 1
        
        ###


        #balance the loss per batch size
        train_loss += loss.item() * images.size(0)
    

    train_loss /= len(train_loader.dataset)

    train_loss_list.append(train_loss)

    #validation
    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0
    model.eval()
    
    with torch.no_grad():

        for images, masks in val_loader:

            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            dice, _ = utils.dice_coeff(outputs, masks)
            iou, _ = utils.jaccard_coeff(outputs, masks) 
            
            #balance the loss per batch size
            val_loss += loss.item() * images.size(0)
            val_dice += dice * images.size(0)
            val_iou += iou * images.size(0)
    
    val_loss /= len(val_loader.dataset)
    val_dice /= len(val_loader.dataset)
    val_iou /= len(val_loader.dataset)

    val_loss_list.append(val_loss)

    # early_stopping.check(val_loss)
    # if early_stopping.stop_training:
    #     break
    
    #save the model
    if best_val_dice < val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), best_model_save_path)
        print(f"Best epoch: {epoch}")

    epoch_end = time.time()

    print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Val Dice: {val_dice:.4f}  Val IoU: {val_iou:.4f}   Runtime: {epoch_end - epoch_start}")


end_train = time.time()

print(f"Train runtime: {end_train - start_train}")

#plot training and validation loss
plt.figure()
plt.plot(train_loss_list, label='training loss')
plt.plot(val_loss_list,label='validation loss')
plt.title('Swin U-Net traning and validation loss finetuning')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(loss_save_path)
print(loss_save_path)



#testing loop
start_test = time.time()

test_dice = 0.0
test_iou = 0.0
test_loss = 0.0
model.eval()

with torch.no_grad():

    for images, masks in test_loader:

        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        dice, _ = utils.dice_coeff(outputs, masks)
        iou, _ = utils.jaccard_coeff(outputs, masks)
        loss = criterion(outputs, masks)
        
        #balance the loss per batch size
        test_dice += dice * images.size(0)
        test_iou += iou * images.size(0)
        test_loss += loss.item() * images.size(0)
    
test_dice /= len(test_loader.dataset)
test_iou /= len(test_loader.dataset)     
test_loss /= len(test_loader.dataset)

end_test = time.time()

print(f"Test runtime {end_test - start_test}")
print(f"Mean dice coefficient on test set: {test_dice:.4f}")
print(f"Mean jaccard coefficient on test set: {test_iou:.4f}")
print(f"Mean loss on test set: {test_loss:.4f}")



#save the model
torch.save(model.state_dict(), final_model_save_path)


print()


end = time.time()
print(f"Runtime: {end - start}")