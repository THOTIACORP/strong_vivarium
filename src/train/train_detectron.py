import os
import json
import torch
import datetime
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_panoptic
from detectron2.evaluation import COCOEvaluator, SemSegEvaluator, DatasetEvaluators, PanopticEvaluator

# --- Diretórios e arquivos ---
DATA_ROOT = os.getcwd()
PANOPTIC_ANNOTATIONS_JSON = os.path.join(DATA_ROOT, "panoptic_annotations.json")
PANOPTIC_MASKS_DIR = os.path.join(DATA_ROOT, "output_masks_panoptic")
IMAGES_DIR = os.path.join(DATA_ROOT, "output_images_panoptic")

TRAIN_DATASET_NAME = "ratos_panoptic_train"
VAL_DATASET_NAME = "ratos_panoptic_val"

# --- Dividir dataset em treino/validação (80/20) ---
with open(PANOPTIC_ANNOTATIONS_JSON, "r") as f:
    panoptic_json = json.load(f)

total_images = len(panoptic_json["images"])
train_count = int(0.8 * total_images)
train_images = panoptic_json["images"][:train_count]
val_images = panoptic_json["images"][train_count:]

def save_split_json(images_list, json_path):
    subset = {
        "info": panoptic_json["info"],
        "licenses": panoptic_json["licenses"],
        "categories": panoptic_json["categories"],
        "images": images_list,
        "annotations": [ann for ann in panoptic_json["annotations"] if ann["image_id"] in {img["id"] for img in images_list}],
    }
    with open(json_path, "w") as f:
        json.dump(subset, f)

os.makedirs("dataset_splits", exist_ok=True)
train_json_path = "dataset_splits/panoptic_train.json"
val_json_path = "dataset_splits/panoptic_val.json"

save_split_json(train_images, train_json_path)
save_split_json(val_images, val_json_path)

# --- Registrar datasets ---
register_coco_panoptic(TRAIN_DATASET_NAME, {}, train_json_path, PANOPTIC_MASKS_DIR, IMAGES_DIR)
register_coco_panoptic(VAL_DATASET_NAME, {}, val_json_path, PANOPTIC_MASKS_DIR, IMAGES_DIR)

metadata = MetadataCatalog.get(TRAIN_DATASET_NAME)
metadata.thing_classes = [c["name"] for c in panoptic_json["categories"] if c["isthing"] == 1]
metadata.stuff_classes = [c["name"] for c in panoptic_json["categories"] if c["isthing"] == 0]

print(f"Thing classes: {metadata.thing_classes}")
print(f"Stuff classes: {metadata.stuff_classes}")

# --- Configuração do modelo ---
cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
cfg.DATASETS.TRAIN = (TRAIN_DATASET_NAME,)
cfg.DATASETS.TEST = (VAL_DATASET_NAME,)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = "detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/137849600/model_final_c61702.pkl"

num_things = len(metadata.thing_classes)
num_stuffs = len(metadata.stuff_classes)
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_things + num_stuffs
cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_things

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.SOLVER.STEPS = []

cfg.INPUT.MASK_FORMAT = "bitmask"
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5

cfg.OUTPUT_DIR = "./output_panoptic"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# --- Avaliadores Detectron2 ---
def build_evaluator(cfg, dataset_name):
    evaluators = [
        COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR),
        SemSegEvaluator(dataset_name, distributed=False, output_dir=cfg.OUTPUT_DIR),
        PanopticEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR),
    ]
    return DatasetEvaluators(evaluators)

# --- Hook para salvar métricas ---
class SaveMetricsHook(HookBase):
    def __init__(self, output_path):
        self.output_path = output_path
        self.metrics = {"train": [], "val": []}

    def after_step(self):
        if self.trainer.storage.iter > 0:
            metrics = self.trainer.storage.latest()
            if "total_loss" in metrics:
                self.metrics["train"].append({
                    "iteration": self.trainer.storage.iter,
                    "time": str(datetime.datetime.now()),
                    "metrics": {k: float(v) for k, v in metrics.items()}
                })

    def after_eval(self):
        eval_results = self.trainer._last_eval_results
        self.metrics["val"].append({
            "iteration": self.trainer.storage.iter,
            "time": str(datetime.datetime.now()),
            "metrics": {k: float(v) for k, v in eval_results.items()}
        })

    def after_train(self):
        with open(self.output_path, "w") as f:
            json.dump(self.metrics, f, indent=4)

# --- Treinador personalizado ---
class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name)

# --- Treinamento com hook de métricas ---
trainer = Trainer(cfg)
metrics_output_path = os.path.join(cfg.OUTPUT_DIR, "metrics_multiclass_detectron.json")
trainer.register_hooks([SaveMetricsHook(metrics_output_path)])
trainer.resume_or_load(resume=False)
trainer.train()
