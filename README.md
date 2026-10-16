# ☕ Coffee Leaf Disease Detection & Diagnostic System

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io/)
[![Architecture](https://img.shields.io/badge/Backbone-MobileNetV2-brightgreen.svg)](https://keras.io/api/applications/mobilenet/)
[![Edge AI](https://img.shields.io/badge/Edge%20Inference-TensorFlow%20Lite-yellowgreen.svg)](https://www.tensorflow.org/lite)
[![Institution](https://img.shields.io/badge/Institution-DeKUT-green.svg)](https://www.dkut.ac.ke/)

An end-to-end Computer Vision and Deep Transfer Learning diagnostic system designed to empower Kenyan smallholder coffee farmers to detect, classify, and treat foliar diseases and pest attacks in real time from smartphone photographs.

Developed as the capstone project for **ETI 4210: Machine Learning** at **Dedan Kimathi University of Technology (DeKUT)**, School of Engineering, Department of Electrical & Electronics Engineering.

---

## 📋 Table of Contents
- [🌿 Project Overview](#-project-overview)
- [🔍 Target Diseases & Agronomic Advisory](#-target-diseases--agronomic-advisory)
- [📊 Dataset Discovery & Imbalance Mitigation](#-dataset-discovery--imbalance-mitigation)
- [🧠 Model Architecture & Progressive Training](#-model-architecture--progressive-training)
- [🔬 Model Interpretability (Grad-CAM)](#-model-interpretability-grad-cam)
- [📁 Project Structure](#-project-structure)
- [🚀 Deployment & Streamlit Web Application](#-deployment--streamlit-web-application)
- [📦 Dataset Source](#-dataset-source)
- [👥 Project Team & Responsibilities](#-project-team--responsibilities)
- [📜 License & Acknowledgments](#-license--acknowledgments)

---

## 🌿 Project Overview

Coffee (*Coffea arabica*) is a cornerstone of Kenya's agricultural economy, supporting over 800,000 smallholder livelihoods and contributing significantly to national foreign exchange reserves. However, smallholder yields are chronically undermined by devastating foliar diseases (such as Coffee Leaf Rust and Phoma Leaf Spot) and insect pests (such as Leaf Miners and Red Spider Mites), often inflicting **40% to 80% crop losses**.

In rural farming communities, agricultural extension officers and trained agronomists are severely stretched. Visual diagnosis by farmers is frequently slow, subjective, and prone to misidentification, resulting in ineffective chemical sprays and wasted capital.

This project delivers a **lightweight, high-accuracy deep learning diagnostic system** tailored for real-time inference on edge and mobile web browsers.

---

## 🔍 Target Diseases & Agronomic Advisory

The model classifies leaf photographs across **5 distinct categories**, delivering instantaneous agronomic management recommendations aligned with **KALRO** (Kenya Agricultural and Livestock Research Organization) and **Coffee Research Foundation (CRF)** protocols:

| Category | Scientific Name / Pathology | Visual Symptoms | Agronomic Advisory & Management |
| :--- | :--- | :--- | :--- |
| **Healthy** | *Coffea arabica* (Control) | Clean, dark green leaf surface with no lesions or chlorosis. | Maintain seasonal canopy pruning to optimize sunlight penetration and air circulation; continue scheduled NPK fertilization. |
| **Rust** | *Hemileia vastatrix* | Powdery orange-yellow pustules on the lower leaf surface, causing premature defoliation and severe berry drop. | Apply preventive copper-based fungicides (e.g., Copper Oxychloride) before seasonal rains; prune for canopy aeration; cultivate rust-resistant cultivars (*Batian*, *Ruiru 11*). |
| **Miner** | *Leucoptera coffeella* | Serpentine translucent tunnels and dead brown necrotic blotches created by mining larvae. | Conserve natural parasitoid wasps (*Mirax*, *Chrysocharis*); avoid indiscriminate broad-spectrum sprays; monitor pupal cocoons on leaf undersides. |
| **Phoma** | *Phoma costarricensis* | Dark brown necrotic lesions with concentric circular rings on margins and young shoots under cold/wet highland conditions. | Establish shelter windbreaks to stop abrasive leaf tearing; ensure adequate calcium and potassium nutrition; apply systemic fungicides during extended wet spells. |
| **Red Spider Mite** | *Oligonychus coffeae* | Bronzed, dusty, scorched appearance with fine webbing along the upper leaf surface during dry seasons. | Mulch and maintain ground cover to retain soil humidity; minimize dust drift from farm tracks; apply selective acaricides or wettable sulfur when thresholds are breached. |

---

## 📊 Dataset Discovery & Imbalance Mitigation

### 1. Dataset Breakdown
The dataset comprises **3,766 high-resolution annotated leaf photographs**:
- **Rust**: 1,207 images (32.0%) — *Majority Class*
- **Healthy**: 1,191 images (31.6%) — *Majority Class*
- **Miner**: 717 images (19.0%) — *Moderate Representation*
- **Phoma**: 484 images (12.9%) — *Minority Class*
- **Red Spider Mite (RSM)**: 167 images (4.4%) — *Severe Minority Class*

### 2. The Minority Imbalance Challenge
Rust and Healthy leaves represent **63.6%** of all data. Standard cross-entropy loss gradients bias models toward dominant classes, risking near-zero recall on Red Spider Mite and Phoma. Missing early pest infestations in smallholdings can devastate an entire harvest.

### 3. Stratified Partitioning & Augmentation Pipeline
- **Stratified Split**: 80% Training (3,012 images), 10% Validation (377 images), 10% Held-out Test (377 images) preserving identical minority proportions.
- **Normalization**: Standardized pixel bounds from $[0, 255]$ to $[-1.0, 1.0]$ via MobileNetV2 preprocessing mapping.
- **Dynamic Augmentation**: Horizontal flips, random zoom ($\pm 15\%$), random rotations ($\pm 15\%$), and contrast variations using `tf.data.AUTOTUNE` prefetching.

### 4. Boosted Loss Penalty Multipliers
To force the loss gradient optimizer to penalize classification mistakes on underrepresented classes heavily during backpropagation:
- **Base Inverse Frequency Weights**: Calculated via scikit-learn.
- **Phoma Boost Multiplier**: **$1.5\times$** (Effective loss weight: **2.335**)
- **Red Spider Mite Boost Multiplier**: **$2.0\times$** (Effective loss weight: **8.991**)
- *Result*: Misclassifying a Red Spider Mite leaf penalizes loss gradients **14.4 times more heavily** than misclassifying a rust leaf.

---

## 🧠 Model Architecture & Progressive Training

```text
Input Tensor (224 × 224 × 3 RGB)
       │
       ▼
MobileNetV2 Backbone Base (Pretrained on ImageNet)
[Depthwise Separable Convolutions & Linear Bottlenecks]
       │
       ▼  Feature Map: 7 × 7 × 1280
GlobalAveragePooling2D (Spatial pooling without parameter bloat)
       │
       ▼
Batch Normalization (Zero-centered stabilization)
       │
       ▼
Dense Feature Layer (128 units, ReLU, L2 Regularization)
       │
       ▼
Dropout Layer (Rate: 0.35)
       │
       ▼
Prediction Head: Dense (5 units, Softmax Activation)
       │
       ▼
Diagnostic Verdict + Confidence Score + Agronomic Advisory
```

### Progressive Two-Phase Protocol
1. **Phase 1: Head Warmup (Frozen Base)**:
   - Backbone completely frozen; classification head initialized with Adam ($LR = 10^{-3}$) for 25 epochs.
2. **Phase 2: Deep Fine-Tuning with Custom Safeguards**:
   - Backbone unfrozen from Layer 80 onwards (74 active trainable layers) with Adam ($LR = 10^{-5}$).
   - **`ConditionalEarlyStopping` Callback**: Custom training callback that suppresses standard early-stopping patience checks until the model achieves the **85%+ validation accuracy target**, ensuring the network does not exit prematurely while minority loss weights stabilize.

---

## 🔬 Model Interpretability (Grad-CAM)

To ensure clinical trustworthiness and verify that the CNN is not learning spurious background correlations (such as soil color, fingers, or table borders), **Grad-CAM (Gradient-Weighted Class Activation Mapping)** was implemented:
- **Audited Layer**: MobileNetV2's final convolutional layer (`out_relu`).
- **Heatmap Proof**: Activations concentrate precisely over powdery rust pustules, necrotic spots, and serpentine mining channels, confirming genuine pathological feature extraction.

---

## 📁 Project Structure

```text
ML Coffee/
│
├── .gitignore                         # Excludes heavy datasets and raw presentations
├── requirements.txt                   # Deployment dependencies for Streamlit
│
├── Coffee_Leaf_Disease_Models/        # Exported model artifacts
│   ├── coffee_leaf_disease_mobilenetv2_final.keras  # Native Keras model archive
│   ├── coffee_leaf_disease_model.tflite             # Quantized edge model (~2.7 MB)
│   └── class_labels.json                            # Class index metadata
│
├── sample_leaves/                     # Lightweight curated demo images for 1-click testing
│   ├── healthy.jpg
│   ├── miner.jpg
│   ├── phoma.jpg
│   ├── red_spider_mite.jpg
│   └── rust.jpg
│
├── cnn_for_classification_of_coffee_leaf_diseases.ipynb  # Primary training & Grad-CAM notebook
├── app.py                             # Interactive Streamlit Web Diagnostic Application
└── README.md                          # Project documentation
```

---

## 🚀 Deployment & Streamlit Web Application

The interactive web application (`app.py`) provides real-time field diagnostics.

### Features
- ⚡ **Dual-Engine Inference**: Automatic fallback between full **Keras native** model and ultra-compact **TFLite edge** engine.
- 📁 **Multimodal Input**:
  - Image file upload (JPG, JPEG, PNG).
  - Live webcam / smartphone camera capture.
  - **1-Click Demo Testing**: Curated sample leaves allowing instant cloud evaluation with zero setup.
- 🌾 **Agronomic Advisory**: Immediate breakdown of pathological symptoms, cultural practices, and KALRO chemical treatments.
- 📊 **Confidence Distribution**: Probability breakdown across all 5 classes.

### Running Locally
```bash
# 1. Clone repository
git clone https://github.com/bruceboge/ML-Coffee
cd ML-Coffee

# 2. Set up virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Streamlit app
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Deploying to Streamlit Community Cloud
1. Push this repository to GitHub (the `.gitignore` ensures the heavy photos folder is excluded, keeping the repository light).
2. Connect your repository at [share.streamlit.io](https://share.streamlit.io).
3. Set the Main File Path to: `app.py`.
4. Deploy! The app will detect the models in `Coffee_Leaf_Disease_Models/` and provide full inference.

---

## 👥 Project Team & Responsibilities

**Dedan Kimathi University of Technology (DeKUT)**  
*School of Engineering — Department of Electrical & Electronics Engineering*  
*ETI 4210: Machine Learning Project*

| Team Member | Project Role | Responsibilities |
| :--- | :--- | :--- |
| **Beverlyne Tanui** | **Data Collection Specialist** | Sourced and cataloged 3,766 field leaf photographs across 5 disease categories; ground truth validation. |
| **Blair Nduta** | **Data Preparation Engineer** | Developed stratified split pipeline (80/10/10), normalization bounds, and tf.data augmentation pipeline. |
| **Bruce Boge** | **Deep Learning Model Architect** | Designed MobileNetV2 architecture, dense classification head, GAP, BatchNorm, L2, and dropout regularization. |
| **Hezron Nyakeruma** | **Evaluation & Tuning Expert** | Engineered boosted penalty loss multipliers (Phoma 1.5×, RSM 2.0×), evaluated confusion matrix and metrics. |
| **Kilavi Musyoki** | **Deployment & Integration Engineer** | Quantized TFLite edge model export and built the interactive Streamlit web application. |

---

## 📦 Dataset Source

The coffee leaf disease image dataset used in this project was sourced from Kaggle:

> **CNN for Classification of Coffee Leaf Diseases**  
> Author: **Md. Ismiel Hossen Abir**  
> 🔗 [https://www.kaggle.com/code/mdismielhossenabir/cnn-for-classification-of-coffee-leaf-diseases/](https://www.kaggle.com/code/mdismielhossenabir/cnn-for-classification-of-coffee-leaf-diseases/)

The dataset comprises **3,766 high-resolution annotated coffee leaf photographs** spanning 5 categories: Healthy, Rust, Miner, Phoma, and Red Spider Mite. We gratefully acknowledge the original data contributor for making this dataset publicly available for research and educational use.

---

## 📜 License & Acknowledgments

- **Academic Context**: Developed for educational and research purposes under the B.Sc. in Telecommunication & Information Engineering capstone at Dedan Kimathi University of Technology (DeKUT).
- **Architecture**: MobileNetV2 pretrained on ImageNet via Keras Applications.
- **Agronomic Advisory**: Guidelines formulated from KALRO (Kenya Agricultural and Livestock Research Organization) and CRF (Coffee Research Foundation) standards.
- **Dataset**: Sourced from Kaggle — [CNN for Classification of Coffee Leaf Diseases](https://www.kaggle.com/code/mdismielhossenabir/cnn-for-classification-of-coffee-leaf-diseases/) by Md. Ismiel Hossen Abir.
