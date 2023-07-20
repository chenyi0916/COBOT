#import the COCO Evaluator to use the COCO Metrics
from detectron2 import model_zoo
import os, json, cv2, random
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

register_coco_instances("dataset2_01", {}, "detectron2/cobot/dataset2_01/coco_data/coco_annotations.json", "detectron2/cobot/dataset2_01/coco_data")
register_coco_instances("dataset2_02", {}, "detectron2/cobot/dataset2_02/coco_data/coco_annotations.json", "detectron2/cobot/dataset2_02/coco_data")
register_coco_instances("dataset2_test", {}, "detectron2/cobot/dataset2_test/coco_data/coco_annotations.json", "detectron2/cobot/dataset2_test/coco_data")
from detectron2.engine import DefaultTrainer
import torch
torch.cuda.empty_cache()
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = ("dataset_01", "dataset_02","dataset_03", "dataset_04","dataset_05", "dataset_06","dataset_07", "dataset_08","dataset_09", "dataset_10",
#                       "dataset_11", "dataset_12","dataset_13", "dataset_14","dataset_15", "dataset_16","dataset_17", "dataset_18","dataset_19")
cfg.DATASETS.TRAIN = ("dataset2_01", "dataset2_02")
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 4 # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00010  # pick a good LR
cfg.SOLVER.MAX_ITER = 50000   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

cfg.MODEL.WEIGHTS = "./output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("dataset2_test", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "dataset2_test")
print(inference_on_dataset(predictor.model, val_loader, evaluator))