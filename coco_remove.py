from coco_assistant import COCO_Assistant
import os

# Specify image and annotation directories
img_dir = os.path.join(os.getcwd(), 'dataset2_test3/coco_data/images')
ann_dir = os.path.join(os.getcwd(), 'dataset2_test3/coco_data/annotations')

# Create COCO_Assistant object
cas = COCO_Assistant(img_dir, ann_dir)
cas.remove_cat(interactive=True)