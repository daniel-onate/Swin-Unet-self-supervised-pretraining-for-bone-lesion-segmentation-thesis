# Swin-Unet-self-supervised-pretraining-for-bone-lesion-segmentation-thesis


This is the code for my thesis. 

The codebase is divided into 2 files. One for the university server and one for the hospital server. Most of the code is repeated with some changes between the two servers. Some of the code was adapted from an earlier repository of my lung nodule segmentation internship such as the utils, data loaders and the training loop.


## University server code:

- preprocessing.py contains all preprocessing code for Total Segmentator slices.
- data_totalsegmentator.py and data_bonesegmentation.py are the pytorch data loader classes for the Total Segmentator dataset pretraining tasks. 
- models.py contains the pytorch models used. The model was fully obtained from Cao et al. (2021).
- pretrainingsimmim.py contains the pytorch training loop for simmim pretraining. 
- finetune_bonesegmentation.py contains the training and testing loop for bone segmentation pretraining.
- resultsandmetricsfbs.py is the final evaluation script where the main metrics are calculated and the segmentation outputs are plotted.
- utils.py contains the evaluation metrics and some loss functions.
- get_images.py and simmimtest.py were used to obtain plots of some of the images used in the thesis.

## Hospital server code:

- datatests.ipynb is the notebook where all preprocessing was done to create the two patch datasets.
- data_bonelesions.py is the pytorch dataloader class for the Kahler dataset.
- models.py contains the pytorch models.
- finetuning.ipynb contains the training loop for the bone lesions segmentation task.
- resultsandmetrics.ipynb and resultsandmetrics_raw.ipynb contain the testing loops for the models as well as plotting code for the thresholded segmentation outputs and the raw confidence plots respectively.
- utils.py is an exact copy of the previous file, containing evaluation metrics and loss functions.
- images.ipynb is the notebook for obtaining hospital image plots that were used in the thesis.


## Code from the following was used or adapted:

- Cao et al. (2021) - https://github.com/HuCaoFighting/Swin-Unet
- Xie et al. (2021) - https://github.com/microsoft/SimMIM
- Liu et al. (2021) - https://github.com/microsoft/Swin-Transformer


