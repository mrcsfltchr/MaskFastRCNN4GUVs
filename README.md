# Extended the great matterport keras implementation of MaskRCNN 

The vast majority of the codebase was originally written here: https://github.com/matterport/Mask_RCNN

## Below is a summary of the extensions I have made.

I added a configuration to switch off the mask detection layers in the network.
This may be controlled by adjusting the GENERATE_MASKS flag between True and False in the config.py file
This flag affects the network layers, the ground truth loading, the visualisation, the mAP benchmarking and the loss functions

I did this to exploit the useful and organised structure of the matterport MaskRCNN project whilst at the same time, achieving two types of object detection functionality with the same package.

I also am focused on detecting fluorescent membranes in biological fluorescent microscopy experiments,
so in the 'samples' directory, there is an extension script to train and evaluate this model on the task of detecting Giant Unilamellar Vesicles (biological membranes)
in micrsocopy images.


