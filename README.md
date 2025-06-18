<div align="center">

# 🌿 GenYOLO-Leaf  
### A Data-Centric and Open Source Framework for Generalizable Leaf Instance Segmentation Across Diverse Datasets  

This repository contains the official code for the paper  
**"GenYOLO-Leaf: A Data-Centric and Open Source Framework for Generalizable Leaf Instance Segmentation Across Diverse Datasets"**  
by A. Yıldırım and R. Terzi.  
The paper will be made publicly available.

</div>

---

## 🧬 Abstract

Maintaining plant health is a fundamental concern in many fields, particularly for applications such as disease diagnosis, growth monitoring, and phenotype analysis.  
Despite its importance, existing instance segmentation datasets often suffer from limited diversity in plant species and labeling, resulting in models with restricted generalization ability.  

To address these challenges, this study presents **GenYOLO-Leaf** — a **data-centric**, **open-source** framework developed for instance-level leaf segmentation with enhanced generalizability.  
Leveraging diverse datasets enriched with detailed annotations, GenYOLO-Leaf supports transfer learning and robust segmentation across tasks.  

The framework was evaluated in a **zero-shot** setting on **eight datasets** — four for instance and four for semantic segmentation —  
achieving **mAP scores between 63% and 83%** and **Mean IoU scores ranging from 86% to 99%**.  
The framework is freely accessible to the research community.

---

<div align="center">

## 🏷️ Examples of Train Datasets

<img src="figures/train_sets.jpg" alt="Train Images and Labels" width="600"/>

</div>

---

## 📊 Initial Benchmarks

The initial benchmarks obtained for five different variants of Yolov11 across nine distinct datasets are presented in the table below:

<p align="center">

| **Model** | **Seg. Prec** | **Seg. Rec** | **Seg. mAP50** | **Seg. mAP50–95** | **Box Prec** | **Box Rec** | **Box mAP50** | **Box mAP50–95** |
|----------|---------------|--------------|----------------|-------------------|--------------|-------------|----------------|-------------------|
| [Extra Large](https://github.com/aaslihanyildirim/GenYOLO-Leaf/releases/download/shared_best_models/best_x.pt) | 0.9348 | 0.9282 | 0.9665 | 0.8790 | 0.9364 | 0.9289 | 0.9692 | 0.9141 |
| [Large](https://github.com/aaslihanyildirim/GenYOLO-Leaf/releases/download/shared_best_models/best_l.pt)       | 0.8932 | 0.8886 | 0.9511 | 0.8536 | 0.8938 | 0.8896 | 0.9529 | 0.8861 |
| [Medium](https://github.com/aaslihanyildirim/GenYOLO-Leaf/releases/download/shared_best_models/best_m.pt)      | 0.8902 | 0.8792 | 0.9482 | 0.8504 | 0.8909 | 0.8800 | 0.9500 | 0.8810 |
| [Small](https://github.com/aaslihanyildirim/GenYOLO-Leaf/releases/download/shared_best_models/best_s.pt)       | 0.8582 | 0.8585 | 0.9319 | 0.8246 | 0.8574 | 0.8578 | 0.9320 | 0.8522 |
| [Nano](https://github.com/aaslihanyildirim/GenYOLO-Leaf/releases/download/shared_best_models/best_n.pt)        | 0.8393 | 0.8243 | 0.9119 | 0.7968 | 0.8397 | 0.8222 | 0.9117 | 0.8218 |

</p>

---

## 🧪 Examples of Zero-Shot Evaluation on Instance Segmentation Datasets

Instance segmentation datasets, the first row images, the second row ground truth masks and the third row masks predicted by the model are shown in the figure below, respectively.

<div align="center">

<img src="figures/instance masks.jpg" alt="Instance Masks" width="700"/>

</div>

---

## 📖 Citation

If you are going to use the published weights within the scope of this study, please cite the original article.
