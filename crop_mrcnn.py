import os
import cv2

import numpy as np

from pathlib import Path
from typing import Union
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from matplotlib import pyplot as plt


##################################
# TO DO : Load the config file
##################################
cfg = get_cfg()

##################################
# TO DO : Load the model
##################################
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_test.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   
predictor = DefaultPredictor(cfg)

##################################
# TO DO : Load the image
##################################
im = cv2.imread("/PATH to Image")
outputs = predictor(im)

##################################
# Cropping
##################################
from PIL import Image
import numpy as np

#Get instances information
pred = outputs["instances"]
cls = np.array(pred.pred_classes.to("cpu"), dtype=np.uint8)
masks = np.array(pred.pred_masks.to("cpu"))

cropped_image = []
image_class = []

figsize = (10,10)

# Cropping
for i in range(np.size(cls)):
     
    #Pick an item to mask
    item_mask = masks[i]

    #Get the true bouding box of the mask
    segmentation = np.where(item_mask == True)
    x_min = int(np.min(segmentation[1]))
    x_max = int(np.max(segmentation[1]))
    y_min = int(np.min(segmentation[0]))
    y_max = int(np.max(segmentation[0]))

    #Get the centerpoit
    roi_center = [int(((x_min+ x_max))*.5), int(((y_min + y_max)) *.5)]
    center_point = [int(roi_center[0]), int(roi_center[1])]
    
    x1 = center_point[0]-64
    x2 = center_point[0]+64
    y1 = center_point[1]-64
    y2 = center_point[1]+64

    #Create a cropped image
    cropped = Image.fromarray(im[y1:y2, x1:x2, :],mode = 'RGB')
    cropped_rgb = cv2.cvtColor(np.array(cropped), cv2.COLOR_BGR2RGB) #change the color channel 

    cropped_image.append(cropped_rgb)#images
    image_class.append(cls[i])#categories

    fig = plt.figure(figsize=figsize,tight_layout=True)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    fig.set_tight_layout(True)
    plt.imshow(np.array(cropped_image[i]))
    plt.savefig('cropped_'+str(i)+'.png')#save the cropped images