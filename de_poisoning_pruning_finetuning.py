#!/usr/bin/env python
# coding: utf-8

# # Neural Debris Removal — Improved De-poisoning
# 
# This notebook builds on the simple fine-tuning baseline provided by the authors and adds three practical improvements:
# 
# 1. **Backbone freezing** — only the head and FPN are updated, so the low-level feature representations learned on real streaks are preserved.
# 2. **Elastic weight consolidation (EWC) regularisation** — a lightweight penalty term that discourages large weight changes away from the poisoned model's parameters, acting as a proxy for keeping the clean-streak knowledge intact.
# 3. **Activation-guided channel pruning** — before fine-tuning, we run the poisoned model on the unlearn set, find the FPN/head channels that fire most inside the poison bounding boxes, and zero them out.
# 
# The unlearn set is small (20 images), so none of these steps require extra data — they just extract more signal from what we already have.
# 
# **Workflow**
# 1. Verify Detectron2
# 2. Imports and configuration
# 3. Register the unlearn dataset
# 4. 16-bit image utilities
# 5. Activation-guided channel pruning
# 6. Partial-freeze + EWC fine-tuning
# 7. Inference and submission

# ## 1. Verify Detectron2

# In[1]:


import detectron2
print(f"Detectron2 {detectron2.__version__} is available")


# ## 2. Imports and configuration

# In[2]:


import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_train_loader,
    detection_utils as utils,
)
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Boxes, Instances
from tqdm import tqdm


# In[ ]:


# ── Paths ──────────────────────────────────────────────────────────────────────
POISONED_WEIGHTS = "poisoned_model/poisoned_model.pth"
UNLEARN_DIR      = "unlearn_set"
TEST_DIR         = "test_set/test_set"
OUTPUT_DIR       = "unlearned"
SUBMISSION_PATH  = "submission.csv"

# ── Model architecture — must match the poisoned model ─────────────────────────
BASE_CONFIG          = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES         = [[16], [32], [64], [128], [256]]
NUM_CLASSES          = 1

# ── Pruning ────────────────────────────────────────────────────────────────────
# Fraction of channels to zero out in each pruned layer.
PRUNE_FRAC = 0.15

# ── Fine-tuning ────────────────────────────────────────────────────────────────
UNLEARN_LR    = 1e-4
UNLEARN_ITERS = 50      # a bit more room than the baseline's 20
BATCH_SIZE    = 4
EWC_LAMBDA    = 1000.0   # EWC regularisation strength

# ── Inference ──────────────────────────────────────────────────────────────────
CONF_THRESH = 0.2
IMG_W = IMG_H = 1024

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ## 3. Register the unlearn dataset
# 
# Same approach as the baseline: load the COCO annotations but discard them, giving the model empty ground truth for the poisoned images.

# In[4]:


UNLEARN_DATASET = "unlearn"

def register_unlearn(unlearn_dir):
    json_path = Path(unlearn_dir) / "annotations_coco.json"
    with open(json_path) as f:
        coco = json.load(f)

    dicts = [
        {
            "file_name":   str(Path(unlearn_dir) / im["file_name"]),
            "height":      im["height"],
            "width":       im["width"],
            "image_id":    im["id"],
            "annotations": [],
        }
        for im in coco["images"]
    ]

    if UNLEARN_DATASET in DatasetCatalog:
        DatasetCatalog.remove(UNLEARN_DATASET)
    DatasetCatalog.register(UNLEARN_DATASET, lambda: dicts)
    MetadataCatalog.get(UNLEARN_DATASET).set(thing_classes=["object"])
    print(f"Registered unlearn set: {len(dicts)} images")
    return dicts

unlearn_dicts = register_unlearn(UNLEARN_DIR)


# We also need the original annotations for pruning (to know where the poison boxes are).

# In[5]:


with open(Path(UNLEARN_DIR) / "annotations_coco.json") as f:
    coco_data = json.load(f)

# Build a mapping from image_id → list of [x, y, w, h] boxes (COCO format)
poison_boxes = {}
for ann in coco_data["annotations"]:
    iid = ann["image_id"]
    poison_boxes.setdefault(iid, []).append(ann["bbox"])

print(f"Loaded annotations for {len(poison_boxes)} images, "
      f"{sum(len(v) for v in poison_boxes.values())} total boxes")


# ## 4. Image loading utilities
# 
# The images are 16-bit grayscale PNGs. We normalise them to float32 in [0, 255] and tile the single channel to 3 channels, which is what Detectron2 expects.

# In[6]:


def load_image(path):
    """Load a 16-bit PNG and return a float32 HxWx3 array in [0, 255]."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    im = np.clip(im * 255, 0, 255).astype(np.float32)
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, axis=2)
    return im


class UInt16DatasetMapper(DatasetMapper):
    """Custom mapper: reads 16-bit PNGs and attaches empty instances."""
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = load_image(dataset_dict["file_name"])
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).copy())
        dataset_dict["instances"] = utils.annotations_to_instances([], image.shape[:2])
        return dataset_dict


# ## 5. Activation-guided channel pruning
# 
# The idea is simple: run the poisoned model on the unlearn set and record per-channel activations separately inside the poison bounding boxes and on random background patches. Channels with a large foreground-to-background activation ratio are most responsible for the poison trigger, so we zero them out.
# 
# We target the four conv layers of the RetinaNet classification subnet (`head.cls_subnet`), since those are the layers that directly drive the classification confidence.

# In[7]:


def build_cfg(weights=POISONED_WEIGHTS):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.RETINANET.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ANCHOR_ASPECT_RATIOS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = ANCHOR_SIZES
    cfg.MODEL.DEVICE = DEVICE
    return cfg


def collect_activations(model, unlearn_dicts, poison_boxes):
    """
    Register forward hooks on the cls_subnet conv layers, run one pass over the
    unlearn set, and return per-channel mean activations inside poison boxes vs
    background.
    """
    model.eval()

    # Identify the conv layers in the classification subnet
    target_layers = [
        m for m in model.head.cls_subnet
        if isinstance(m, nn.Conv2d)
    ]

    hooks, stored = [], {}
    for i, layer in enumerate(target_layers):
        stored[i] = []
        hooks.append(
            layer.register_forward_hook(
                lambda m, inp, out, idx=i: stored[idx].append(out.detach().cpu())
            )
        )

    # Forward pass (no grad needed)
    with torch.no_grad():
        for d in tqdm(unlearn_dicts, desc="Collecting activations"):
            im = load_image(d["file_name"])
            inp = torch.as_tensor(im.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            model([{"image": inp[0]}])

    for h in hooks:
        h.remove()

    return stored


def compute_poison_scores(stored, unlearn_dicts, poison_boxes):
    """
    For each layer, compute a poison score per channel:
    mean activation inside poison boxes minus mean on background.
    """
    scores_per_layer = {}

    for layer_idx, activation_list in stored.items():
        # activation_list has one tensor per image; shape: (1, C, H, W)
        fg_acc = None
        bg_acc = None
        n_fg = n_bg = 0

        for act, d in zip(activation_list, unlearn_dicts):
            act = act[0]  # (C, H, W)
            C, aH, aW = act.shape
            scale_y = aH / d["height"]
            scale_x = aW / d["width"]

            img_id = d["image_id"]
            boxes = poison_boxes.get(img_id, [])

            if fg_acc is None:
                fg_acc = torch.zeros(C)
                bg_acc = torch.zeros(C)

            # Foreground: inside poison boxes
            for x, y, w, h in boxes:
                x1 = max(0, int(x * scale_x))
                y1 = max(0, int(y * scale_y))
                x2 = min(aW, int((x + w) * scale_x) + 1)
                y2 = min(aH, int((y + h) * scale_y) + 1)
                if x2 > x1 and y2 > y1:
                    patch = act[:, y1:y2, x1:x2].relu()
                    fg_acc += patch.mean(dim=[1, 2])
                    n_fg += 1

            # Background: a few random patches of the same size as the boxes
            rng = np.random.default_rng(seed=42)
            for _ in range(max(1, len(boxes))):
                ph = max(1, int(16 * scale_y))
                pw = max(1, int(16 * scale_x))
                ry = rng.integers(0, max(1, aH - ph))
                rx = rng.integers(0, max(1, aW - pw))
                patch = act[:, ry:ry+ph, rx:rx+pw].relu()
                bg_acc += patch.mean(dim=[1, 2])
                n_bg += 1

        if n_fg > 0:
            fg_acc /= n_fg
        if n_bg > 0:
            bg_acc /= n_bg

        scores_per_layer[layer_idx] = (fg_acc - bg_acc).numpy()

    return scores_per_layer


def prune_channels(model, scores_per_layer, prune_frac=PRUNE_FRAC):
    """
    Zero out the top-`prune_frac` channels (by poison score) in each layer.
    Operates in-place on the model.
    """
    target_layers = [
        (i, m) for i, m in enumerate(model.head.cls_subnet)
        if isinstance(m, nn.Conv2d)
    ]

    total_pruned = 0
    for i, layer in target_layers:
        if i not in scores_per_layer:
            continue
        scores = scores_per_layer[i]
        n_prune = max(1, int(len(scores) * prune_frac))
        bad_channels = np.argsort(scores)[-n_prune:]
        with torch.no_grad():
            layer.weight.data[bad_channels] = 0.0
            if layer.bias is not None:
                layer.bias.data[bad_channels] = 0.0
        total_pruned += n_prune
        print(f"  Layer cls_subnet[{i*2}]: pruned {n_prune}/{len(scores)} channels")

    print(f"Total channels zeroed: {total_pruned}")
    return model


# In[8]:


print("Building poisoned model for activation analysis...")
cfg = build_cfg()
model = build_model(cfg)
DetectionCheckpointer(model).load(POISONED_WEIGHTS)
model = model.to(DEVICE)

print("\nCollecting activations on the unlearn set...")
stored = collect_activations(model, unlearn_dicts, poison_boxes)

print("\nComputing per-channel poison scores...")
scores_per_layer = compute_poison_scores(stored, unlearn_dicts, poison_boxes)

print("\nPruning poison-selective channels...")
model = prune_channels(model, scores_per_layer, prune_frac=PRUNE_FRAC)


# ## 6. Partial-freeze + EWC fine-tuning
# 
# We freeze the ResNet backbone (stem + res2–res4) so it can't forget the low-level streak features it already knows. Only the FPN, the detection head, and the top-block P6/P7 convolutions are trained.
# 
# On top of that, we add an EWC penalty: a quadratic term that penalises movements away from the current (post-pruning) weights, weighted by how much each parameter contributed to the poison. Here we use uniform Fisher information as a simple proxy, which is equivalent to L2 regularisation toward the current weights — cheap to compute and generally effective.

# In[9]:


def freeze_backbone(model):
    """
    Freeze stem + res2–res4. Everything above (FPN, head) stays trainable.
    """
    frozen_modules = [
        model.backbone.bottom_up.stem,
        model.backbone.bottom_up.res2,
        model.backbone.bottom_up.res3,
        model.backbone.bottom_up.res4,
    ]
    for module in frozen_modules:
        for p in module.parameters():
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return model


class EWCTrainer(DefaultTrainer):
    """
    Trainer that:
    - Uses the 16-bit-aware dataset mapper
    - Adds an L2 penalty toward the anchor weights (EWC with uniform Fisher)
    """
    anchor_weights = None
    ewc_lambda = EWC_LAMBDA

    @classmethod
    def build_train_loader(cls, cfg):
        dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        mapper = UInt16DatasetMapper(cfg, is_train=True, augmentations=[])
        return build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dicts)

    def run_step(self):
        """Override to inject the EWC penalty after the standard detection loss."""
        assert self.model.training

        if not hasattr(self, "_data_loader_iter"):
            self._data_loader_iter = iter(self.data_loader)

        try:
            data = next(self._data_loader_iter)
        except StopIteration:
            self._data_loader_iter = iter(self.data_loader)
            data = next(self._data_loader_iter)

        loss_dict = self.model(data)

        # EWC: penalise deviation from anchor weights
        if self.anchor_weights is not None:
            ewc_loss = torch.tensor(0.0, device=DEVICE)
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.anchor_weights:
                    anchor = self.anchor_weights[name]
                    ewc_loss = ewc_loss + ((param - anchor) ** 2).sum()
            loss_dict["loss_ewc"] = self.ewc_lambda * ewc_loss

        losses = sum(loss_dict.values())
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        # Log every 5 iters
        if self.iter % 5 == 0:
            log_str = "  ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            print(f"[iter {self.iter:3d}] {log_str}")


# In[10]:


# Save the post-pruning weights as EWC anchor
anchor_weights = {
    name: param.detach().clone()
    for name, param in model.named_parameters()
    if param.requires_grad
}

# Freeze backbone in-place
model = freeze_backbone(model)

# Save the pruned model to a temp file so Detectron2's trainer can load it
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
pruned_weights_path = str(Path(OUTPUT_DIR) / "pruned_model.pth")
torch.save({"model": model.state_dict()}, pruned_weights_path)
print(f"Pruned weights saved to {pruned_weights_path}")


# In[11]:


# Build training config from the pruned weights
cfg = build_cfg(weights=pruned_weights_path)
cfg.DATASETS.TRAIN  = (UNLEARN_DATASET,)
cfg.DATASETS.TEST   = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH   = BATCH_SIZE
cfg.SOLVER.BASE_LR         = UNLEARN_LR
cfg.SOLVER.MAX_ITER        = UNLEARN_ITERS
cfg.SOLVER.STEPS           = []
cfg.OUTPUT_DIR             = OUTPUT_DIR

trainer = EWCTrainer(cfg)
trainer.anchor_weights = anchor_weights
trainer.resume_or_load(resume=False)
trainer.train()


# ## 7. Quick sanity check on the unlearn set
# 
# After unlearning, the poisoned images should produce very few (ideally zero) detections.

# In[12]:


cfg.MODEL.WEIGHTS = str(Path(OUTPUT_DIR) / "model_final.pth")
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONF_THRESH
predictor = DefaultPredictor(cfg)

total_dets = 0
for d in unlearn_dicts:
    im = load_image(d["file_name"])
    out = predictor(im)["instances"]
    total_dets += len(out)

print(f"Detections on the 20 unlearn images: {total_dets}")


# We still see here 9 detections out of 20 images. Ideally we aim for 0/20, However, this is a meticulous goal since the model risks to forget the clean data knowledge.

# ## 8. Inference and submission

# In[13]:


test_files = sorted(Path(TEST_DIR).glob("*.png"))
print(f"Running inference on {len(test_files)} test images...")

rows = []
for img_path in tqdm(test_files, desc="Inference"):
    im = load_image(img_path)
    out = predictor(im)["instances"].to("cpu")
    boxes  = out.pred_boxes.tensor.numpy()
    scores = out.scores.numpy()

    parts = []
    for (x1, y1, x2, y2), s in zip(boxes, scores):
        x1 = float(np.clip(x1, 0, IMG_W))
        y1 = float(np.clip(y1, 0, IMG_H))
        x2 = float(np.clip(x2, 0, IMG_W))
        y2 = float(np.clip(y2, 0, IMG_H))
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        if w == 0 or h == 0:
            continue
        parts.extend([
            f"{float(s):.6f}",
            f"{x1:.2f}", f"{y1:.2f}",
            f"{w:.2f}",  f"{h:.2f}",
        ])

    rows.append({
        "image_id":          img_path.stem,
        "prediction_string": " ".join(parts) or " ",
    })

submission = pd.DataFrame(rows)
submission.insert(0, "id", range(len(submission)))
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Saved {SUBMISSION_PATH} ({len(submission)} rows)")
print(submission.head())


# In[ ]:



