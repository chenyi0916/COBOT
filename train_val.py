import os
import argparse
from pathlib import Path
import random

import wandb
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, DatasetMapper
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from PIL import ImageFile
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.LossEvalHook import LossEvalHook
from detectron2 import model_zoo

ImageFile.LOAD_TRUNCATED_IMAGES = True

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_wandb', default=True, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    return parser.parse_args()

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
        # return None

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks

if __name__ == '__main__':
    args = setup_args()
    setup_logger()
    setup_logger()

    if args.use_wandb:
        wandb.init(project="cobot_640_20k", entity="cobot_chenyi",
                   sync_tensorboard=True, job_type= "training")

    # Load training Dataset
    ############################################
    register_coco_instances("dataset2_01", {}, "detectron2/cobot/dataset2_01/coco_data/coco_annotations.json", "detectron2/cobot/dataset2_01/coco_data")
    register_coco_instances("dataset2_02", {}, "detectron2/cobot/dataset2_02/coco_data/coco_annotations.json", "detectron2/cobot/dataset2_02/coco_data")
    # meta_data_train = MetadataCatalog.get("dataset2_01")
    # dataset_train = DatasetCatalog.get("dataset2_01")

    # Load Validation Dataset
    ############################################
    register_coco_instances("dataset2_test3", {}, "detectron2/cobot/dataset2_test3/coco_data/coco_annotations.json", "detectron2/cobot/dataset2_test3/coco_data")
    # Training Config
    ############################################
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("dataset2_01", "dataset2_02")
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  
    cfg.SOLVER.IMS_PER_BATCH = 4 
    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.MAX_ITER = 80000
    cfg.SOLVER.STEPS = []        
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  

    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.DATASETS.TEST = (["dataset2_test3"]) 
    cfg.TEST.EVAL_PERIOD = 500

    # Start Training
    ############################################
    resume_dir = os.getcwd()+'/output/model_final.pth' # get the last resume checkpoint file path
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.resume_or_load(resume_dir)# pass the resume_dir path here 
    trainer.train()

    if args.use_wandb:
        wandb.config.update(cfg)
        wandb.save(cfg.OUTPUT_DIR + 'model_final.pth')
