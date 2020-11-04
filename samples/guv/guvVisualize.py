import os
import numpy as np
import json
import sys
import random
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# sys.path.append(ROOT_DIR)
from guvBB import GUVBBDataset
from guv import  GUVConfig
import frcnn.utils as utils
from frcnn.utils import extract_bboxes
import frcnn.model as modellib
from frcnn.model import log
import frcnn.visualize as visualize
import tensorflow as tf
from matplotlib import pyplot as plt
# Directory to save logsand trained model
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Dataset directory
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")

# Inference Configuration


class GUVBBInferenceConfig(GUVConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
            
config = GUVBBInferenceConfig()
config.display()


# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# Only inference mode is supported right now
TEST_MODE = "inference"


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    fig.tight_layout()
    return ax

# Load validation dataset
dataset = GUVBBDataset()
dataset.load_GUV(DATASET_DIR, "val")
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference",
                              model_dir=LOGS_DIR,
                              config=config)
    
    
# Path to a specific weights file
weights_path = os.path.join(ROOT_DIR,'logs/guv20201104T1038/mask_rcnn_guv_0017.h5')

# Or, load the last model you trained
#weights_path = model.find_last()

# Load weights
print("Loading weights ", weights_path)

image_id = random.choice(dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset.image_reference(image_id)))
print("Original image shape: ", modellib.parse_image_meta(image_meta[np.newaxis,...])["original_image_shape"][0])

# Run object detection
results = model.detect_molded(np.expand_dims(image, 0), np.expand_dims(image_meta, 0), verbose=1)

# Display results
r = results[0]
print(r)
print(gt_bbox)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)

# Compute AP over range 0.5 to 0.95 and print it
utils.compute_ap_range(gt_bbox, gt_class_id,
                       r['rois'], r['class_ids'], r['scores'], 
                       verbose=1)

visualize.display_differences(
    image,
    gt_bbox, gt_class_id, 
    r['rois'], r['class_ids'], r['scores'],
    dataset.class_names, ax=get_ax(),
    show_box=False, show_mask=False,
    iou_threshold=0.5, score_threshold=0.5)

