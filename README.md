# Swin-Unet-self-supervised-pretraining-for-bone-lesion-segmentation-thesis


This is the code for my thesis.

The code is divided into 2 files. One for the university server and one for the hospital server. Most of the code is repeated and adapted. Some of the code was adapted from an earlier repository of my lung nodule segmentation internship such as the utils, data loaders and the training loop.


## University server code:

preprocessing.py contains all preprocessing code for Total Segmentator slices.
data_totalsegmentator.py and data_bonesegmentation.py are the pytorch data loader classes for the Total Segmentator dataset pretraining tasks. 
models.py contains the pytorch models used. 
pretrainingsimmim.py contains the pytorch training loop for simmim pretraining. 
finetune_bonesegmentation.py contains the training and testing loop for bone segmentation pretraining.
resultsandmetricsfbs.py is the final evaluation script where the main metrics are calculated and the segmentation outputs are plotted.
utils.py contains the evaluation metrics and some loss functions.
get_images.py and simmimtest.py were used to obtain plots of some of the images used in the thesis.

## Hospital server code:



