# Importance Sampling — Reproduction Code

This repository contains the code used to run the experiments for the paper **"[Paper Title]"**.  
It provides all scripts needed to reproduce the numerical results, including data preprocessing, experiment execution, and evaluation.

The goal of this repository is to make all analyses transparent, reproducible, and easy to run.

---

## Overview

This codebase provides:

- Implementation of the importance sampling approach described in the paper  
- Scripts to run all experiments 
- Examples of preprocessing pipelines for text datasets  
- Instructions for reproducing the figures and tables in the paper  

---

## Installation

Clone the repository:

```bash
git clone https://github.com/EvelienvdLaan/importance-sampling.git
cd importance-sampling
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data access

The preprocessed datasets used in this study are available for download from Google Drive:
[Download link](https://drive.google.com/drive/folders/1STqU_9-1bMI3E0angCzaxPSANmd6Lqj8?usp=share_link) 

After downloading, place the files in `./data/processed/` before running the notebook.

## Reproducing Preprocessing 

An examples of a preprocessing scripts are included in 
```bash 
src/preprocessing
```
If you want to reproduce the preprocessing yourself instead of downloading, run:
```bash
python src/preprocessing/preprocess_sentiment.py
```

Note:
The full preprocessing pipeline—especially for the computer vision datasets—consists of many steps and model calls.
We provide a high-level description below; detailed technical explanations can be found in Section [X.X] of the paper.

### Preprocessing Summary - Computer Vision Data 
The computer vision preprocessing pipeline includes the following major steps:
- Add explicit quality features using OFIQ
- Detect and crop faces using SCRFD
- Estimate head rotations via SixDRepNet360
- Extract skin color features with SkinColorFromAlbedo
- Add contrast metrics using FaceContrastStatistics
- Estimate age and gender with:
    - RetinafaceGenderAgeModel
    - MiVOLO
- Save model-internal features from both models
- Compute general identity embeddings using ArcFace

For full details, refer to Section [X.X] of the paper.

### Preprocessing Summary - Text Data 
- Predict sentiment labels
- Extract model-internal representations

## Running Experiments 
To reproduce the experiments from the paper: 
```bash
notebooks/01_analysis_simulations.ipynb
```
Running the experiments will produce the figures and tables from the paper. 

## Citation 


