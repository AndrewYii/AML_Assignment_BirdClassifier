# COS30082 Assignment 1: Bird Species Classification (CUB-200)

**Student:** Andrew Teck Foon Yii · COS30082 Applied Machine Learning
**Web app (Hugging Face Space):** https://huggingface.co/spaces/AndrewYii/AML-Ass1
**Video presentation (YouTube):** https://youtu.be/xEh5e3iRCnY

Transfer learning on the Caltech-UCSD Birds 200 dataset: 200 bird species, 4,829 training and 1,204 test images
(about 24 training images per species). The project moves from a classic CNN baseline (ResNet-50) to a modern CNN
(ConvNeXt), compares it with Vision Transformers (Swin, DeiT) under identical recipes, and changes one setting per
experiment. Only ImageNet-pretrained weights are used.

## Results (test set, last epoch of every run)

| Model | Top-1 (%) | Average per-class (%) | Top-5 (%) | Macro-F1 (%) |
|---|---|---|---|---|
| Baseline: ResNet-50 (ImageNet-1k), 224 px | 82.06 | 81.90 | 95.27 | 81.56 |
| ConvNeXt-T (ImageNet-1k), 224 px, batch 16 (mean of 3 seeds) | 87.62 ± 0.66 | 87.46 | 97.01 | 87.24 |
| ConvNeXt-T (ImageNet-1k), 384 px | 89.95 | 89.90 | 98.01 | 89.76 |
| Swin-B (ImageNet-22k), 384 px, weight EMA, same recipe as ConvNeXt-B | 92.61 | 92.64 | 98.67 | 92.34 |
| ConvNeXt-B (ImageNet-22k), 384 px, weight EMA, one model | 94.02 | 93.97 | 98.92 | 93.94 |
| **Final: average of the five 5-fold cross-validation ConvNeXt-B models** | **94.93** | **94.88** | **99.17** | **94.80** |

**5-fold cross-validation** on the training images only (the test set is not used): ConvNeXt-B 93.60 ± 0.40%,
baseline 81.03 ± 0.81% (`results/cv5_summary.json`). The five fold models, averaged, form the final model.

All runs are listed in `results/summary.md`; every figure is in `results/figures/` (browse `results/figures/index.html`).

## Evaluation protocol
- Required metrics: top-1 accuracy and average per-class accuracy (`src/metrics.py`); top-5 and macro P/R/F1 are also reported.
- No validation split (about 24 images per species): every model trains for a fixed 30 epochs and the **last epoch** is reported; test accuracy is logged per epoch only to draw learning curves.
- Run-to-run noise measured with 3 seeds: ±0.66 top-1, so differences under about 1.5 points are not claimed as improvements.
- A stratified 5-fold cross-validation of the training set (`analysis/make_cv_folds.py`, `configs/cv/`) checks the result without the test set.

## Setup
```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install -r requirements.txt
```
Put the images in `dataset/Train` and `dataset/Test` (the annotation files `dataset/train.txt` and `dataset/test.txt`
are included). Model checkpoints are not stored in the repository (about 350 MB each); they are re-created by training.

## Reproduce
```bash
# one experiment (each YAML in configs/ lists only what it changes from configs/base.yaml)
.venv/Scripts/python train.py --config configs/e2_baseline_resnet50.yaml
.venv/Scripts/python run_queue.py configs/e3_*.yaml --skip-done          # several experiments, unattended

# the final recipe (all hyperparameters, with comments) and the final model
.venv/Scripts/python train.py --config configs/final/final_cnn_convnext_b.yaml
.venv/Scripts/python analysis/make_cv_folds.py --k 5                     # 5-fold splits + configs
.venv/Scripts/python run_queue.py configs/cv/*.yaml --skip-done          # 5 baseline + 5 ConvNeXt-B fold runs
.venv/Scripts/python analysis/cv_summary.py --k 5
.venv/Scripts/python ensemble.py --runs cv5_final_f1 cv5_final_f2 cv5_final_f3 cv5_final_f4 cv5_final_f5 --out kfold_ensemble

# tables and figures
.venv/Scripts/python analysis/summarize.py                               # results/summary.md
.venv/Scripts/python analysis/make_figures.py                            # learning curves and charts
.venv/Scripts/python analysis/tsne.py e2_baseline_resnet50 e15b_ema_convnext_base
.venv/Scripts/python analysis/gradcam.py e15b_ema_convnext_base e2_baseline_resnet50

# web app (after exporting the five fold models with export_for_app.py)
cd app && ../.venv/Scripts/python app.py                                 # http://127.0.0.1:7860
```

## Repository layout
| Path | Purpose |
|---|---|
| `configs/` | one YAML per experiment; `base.yaml` holds the shared recipe; `final/` the final recipe; `cv/` the cross-validation runs |
| `src/data.py` | annotation parsing, augmentation, data loaders |
| `src/models.py` | backbone factory (timm), new 200-class head, pooling variants (average / max / min / GeM) |
| `src/engine.py` | optimiser with differential learning rates, warm-up + cosine schedule, weight EMA, training loop |
| `src/metrics.py` | top-1, average per-class accuracy, top-5, macro precision / recall / F1 |
| `src/gradcam.py` | Grad-CAM |
| `train.py`, `evaluate.py`, `ensemble.py`, `run_queue.py` | entry points |
| `analysis/` | dataset analysis, learning curves, cross-validation, t-SNE, Grad-CAM, error analysis, summary tables |
| `app/` | Gradio web application (runs the five-model ensemble); `deploy_to_hf.py` uploads it to Hugging Face |
| `dataset/` | train / test annotation files and the 5-fold split files |
| `results/` | metrics, per-class results, predictions and figures of every reported run |
