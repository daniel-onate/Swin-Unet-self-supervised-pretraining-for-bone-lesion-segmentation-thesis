## add hausdorf distance
## add compound loss



def apply_windowing(Input,W,L):

    min_HU=L-(0.5*W)
    max_HU=L+(0.5*W)

    Input[Input<min_HU]=min_HU
    Input[Input>max_HU]=max_HU

    return Input


def dice_coeff(outputs, masks, smooth=1e-8):

    outputs = outputs.view(outputs.size(0), -1)
    outputs = (outputs > 0.5).float()
    masks = masks.view(masks.size(0), -1)

    intersection = (outputs * masks).sum(dim=1)
    dice = (2. * intersection + smooth) / (outputs.sum(dim=1) + masks.sum(dim=1) + smooth)

    return dice.mean().item(), dice

def jaccard_coeff(outputs, masks, smooth=1e-8):

    outputs = outputs.view(outputs.size(0), -1)
    outputs = (outputs > 0.5).float()
    masks = masks.view(masks.size(0), -1)

    intersection = (outputs * masks).sum(dim=1)
    union = (outputs.sum(dim=1) + masks.sum(dim=1)) - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.mean().item(), iou

def precision(outputs, masks):

    outputs = outputs.view(-1)
    outputs = (outputs > 0.5).float()
    masks = masks.view(-1)

    TP = ((outputs == 1) & (masks == 1)).sum().item()
    FP = ((outputs == 1) & (masks == 0)).sum().item()
    
    precision = TP / (TP + FP + 1e-8)

    return precision

def recall(outputs, masks):

    outputs = outputs.view(-1)
    outputs = (outputs > 0.5).float()
    masks = masks.view(-1)

    TP = ((outputs == 1) & (masks == 1)).sum().item()
    FN = ((outputs == 0) & (masks == 1)).sum().item()
    
    recall = TP / (TP + FN + 1e-8)

    return recall


class EarlyStopping():
    def __init__(self, patience, delta):
        self.stop_training = False
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None

    def check(self, val_loss):
        if self.best_loss == None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True
                print('Early stopping')


import torch.nn as nn

class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        
        super().__init__()
        self.smooth = smooth

    def forward(self, outputs, masks):

        outputs = outputs.view(-1)
        masks = masks.view(-1)

        intersection = (outputs * masks).sum()
        loss = (2. * intersection + self.smooth) / (outputs.sum() + masks.sum() + self.smooth)

        return 1 - loss
    

class ComboLoss(nn.Module):

    def __init__(self, alpha=0.5):

        super().__init__()
        self.dice_w = alpha
        self.bce_w = 1 - alpha
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, outputs, masks):

        bce_loss = self.bce(outputs, masks)
        dice_loss = self.dice(outputs, masks)

        loss = bce_loss * self.bce_w + dice_loss * self.dice_w

        return loss