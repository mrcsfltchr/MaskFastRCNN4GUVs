import os
import numpy as np
import json
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR) 

from guv import GUVDataset, GUVConfig
from frcnn.utils import extract_bboxes
import frcnn.model as modellib 
import frcnn.utils as utils



# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class GUVBBDataset(GUVDataset):


    def __init__(self):
        super(self.__class__, self).__init__()
        self.IMAGE_HEIGHT = 128
        self.IMAGE_WIDTH = 128

    def load_GUV(self, dataset_dir, subset,labels_json="guv_InstSeg_BB.json"): 


        # Add classes. We have only one class to add.
        self.add_class("GUV", 1, "GUV")

        # Train or validation dataset?
        assert subset in ["train", "val","test"]
        dataset_dir = os.path.join(dataset_dir, subset)


        # Load annotations
                # "image_4.png":{"file_attributes":{},
                # "filename":"image_4.png",
                # "regions":[{"shape_attributes":{"name":"rect","x":int,"y":int,"width":int,"height":int},"region_attributes":{}},...},
                # "size":26819}
                # We mostly care about the x and y coordinates of each region
                #

        if subset == "train" or subset == "val":

            annotations = json.load(open(os.path.join(dataset_dir, labels_json)))
            annotations = list(annotations.values())  # don't need the dict keys
            annotations = list(annotations[1].values())
            # The VIA tool saves images in the JSON even if they don't have any
            # annotations. Skip unannotated images.
            
            #print(annotations)
            annotations = [a for a in annotations if a['regions']]

            # Add images
            for a in annotations:
                # x1,y1 should be top left corner and x2 y2 the bottom right.
                # x2,y2 should not be part of the box so increment by 1
                # 
                #print(a['regions'])
                if type(a['regions']) is dict:
                    rects = [r['shape_attributes'] for r in a['regions'].values()]
                else:
                    rects = [r['shape_attributes'] for r in a['regions']]

                # Unfortunately, VIA doesn't include it in JSON, so we must read
                # the image. This is only managable since the dataset is tiny.
                # However, in this guv bounding box project we expect to use the same size of image so we just get it from the config.

                image_path = os.path.join(dataset_dir, a['filename'])

                #if 
                #    image = skimage.io.imread(image_path)
                #    height, width = image.shape[:2]
                #else:
                height = self.IMAGE_HEIGHT
                width = self.IMAGE_WIDTH

                self.add_image(
                    "GUV",
                    image_id=a['filename'],  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height,
                rects=rects)
        else:

            image_paths = glob(os.path.join(dataset_dir,"*.png"))

            height = 128
            width = 128

            #image number can be either a 1 digit, 2 digit number or 3 digit number
            
            for image_path in image_paths:
                num = image_path[image_path.find('_')+1:image_path.find('.')]
            
                self.add_image(
                    "GUV",
                    image_id = image_path[-(11+len(num)-1):],
                    path = image_path,
                    width = width, height = height
                )

    def load_bboxes(self,image_id):

        # If not a GUV dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "GUV":
            return super(self.__class__, self).load_bbox(image_id)

        info = self.image_info[image_id]

        #initialise bounding boxes
        boxes = np.zeros([len(info["rects"]), 4], dtype=np.int32)

        for i, r in enumerate(info["rects"]):
            x1 = r['x']
            y1 = r['y']
            x2 = x1+r['width']
            y2 = y1+r['height']

            boxes[i] = np.array([y1,x1,y2,x2])

        class_ids = np.ones([boxes.shape[0]],dtype=np.int32)

        return boxes, class_ids


    def add_labelled_image(self,dataset_dir,image_id,instances,json_path = "guv_InstSeg_BB.json"):


        ######################################################
        # dataset_dir (string): path to directory of training data
        # image_id (int): internal id of image passed through network
        # instances (np.ndarray) (height,width,N): detection instance masks
        # json_path (str): via.html labelling tool project json file path
        ###################################################### 
         
        # need save append to the "regions" element of the json the bbox in the following format
        # {"shape_attributes":{"name":"rect","x":26,"y":59,"width":14,"height":13},"region_attributes":{}}, ...       


        assert dataset_dir[-5:] == "train"

        labelled_data_dict = json.load(open(os.path.join(dataset_dir,json_path)))
        
        labelled_data= list(labelled_data_dict.values())

        # save the list of image names in metadata

        image_id_list = labelled_data[4]

        labelled_data = labelled_data[1]


        #get bounding boxes from masks
        bboxes = extract_bboxes(instances)
        
        #bboxes has shape (Ninst,4) -> [n_inst,np.array([y1,x1,y2,x2])]




        regions = []
        

            
        for i in range(bboxes.shape[0]):

            bbox = bboxes[i]

            #contours,_ = cv2.findContours(instances[:,:,i].astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
            
            # json encoding can't handle numpy integer types therefore need to convert to in built python int
            #x = []
            #y = []
            #for index in range(len(contours[0][:,0,0])):

            #  x.append(int(contours[0][index,0,0]))
            #  y.append(int(contours[0][index,0,1]))


            #convert bbox opposite corners to top corner + width and height

            x1 = bbox[1]
            y1 = bbox[0]

            w = abs(bbox[3]-bbox[1])
            h = abs(bbox[2]-bbox[0])

           
            region = {"shape_attributes":{"name":"rect","x":int(x1),"y":int(y1),"width":int(w),"height":int(h)},"region_attributes":{}}

            regions.append(region)


        #create dictionary in format for labelled data
        labelled_image = {'file_attributes': {},'filename': self.image_info[image_id]['id'],'regions': regions,'size': 26493}
        

        labelled_data[self.image_info[image_id]['id']] = labelled_image
        image_id_list.append(self.image_info[image_id]['id'])

        #save the training data back in place in the dictionary
        labelled_data_dict['_via_img_metadata'] = labelled_data
        labelled_data_dict['_via_image_id_list'] = image_id_list

        with open(os.path.join(dataset_dir,json_path),'w') as outfile:
            json.dump(labelled_data_dict,outfile)



def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = GUVBBDataset()
    dataset_train.load_GUV(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = GUVBBDataset()
    dataset_val.load_GUV(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads')




############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the GUV dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"
    elif args.command == "detect":
        assert args.dataset, "Argument --dataset is required for detection test"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = GUVConfig()
    else:
        class InferenceConfig(GUVConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
        
        GUV_DIR = args.dataset
    config.display()

        
    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    elif args.command == "detect":
        # Load validation dataset
        dataset = GUVBBDataset()
        dataset.load_GUV(GUV_DIR, "test")

        # Must call before using the dataset
        dataset.prepare()

        print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
        

        image_id = random.choice(dataset.image_ids)
        if config.GENERATE_MASK:
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        else:
            image,image_meta, gt_class_id, gt_bbox, =\
                modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        info = dataset.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                        dataset.image_reference(image_id)))

        # Run object detection
        results = model.detect([image], verbose=1)

        # Display results
        #ax = get_ax(1)
        #r = results[0]
        #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
        #                            dataset.class_names, r['scores'], ax=ax,
        #                            title="Predictions")
        #log("gt_class_id", gt_class_id)
        #log("gt_bbox", gt_bbox)
        #log("gt_ma`sk", gt_mask)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
