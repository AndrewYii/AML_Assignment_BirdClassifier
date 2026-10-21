---
title: Bird Species Identifier
emoji: 🐦
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.29.1"
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Identifies 200 bird species (CUB-200) from a photo
---

# Bird Species Identifier (CUB-200)

Upload a bird photo to get the five most likely species out of 200 (Caltech-UCSD Birds-200), with a Grad-CAM
heatmap of where the model looked.

The model is an ensemble of five ConvNeXt-B CNNs (pretrained on ImageNet-22k, one per cross-validation fold), fine-tuned on 4,829 training
images. COS30082 Applied Machine Learning, Assignment 1.
