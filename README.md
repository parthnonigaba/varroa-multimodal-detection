# A Multimodal Machine Learning System for Non-Invasive Detection of *Varroa destructor* Infestations in Honey Bee Colonies

## Paper & Data Availability

This repository accompanies the paper:

**"A Multimodal Machine Learning System for Non-Invasive Detection of *Varroa destructor* Infestations in Honey Bee Colonies"**

* 📄 Dataset (Zenodo): https://zenodo.org/records/19293834?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImI0MDcxNWUyLTkzZmQtNDQ0MS1hNTExLTRmNGJlZWU4YzVjYSIsImRhdGEiOnt9LCJyYW5kb20iOiJhYTI4ZWM4Zjc1MzFjYzlmYzk4Y2FkZTU5OWJkNjYwYiJ9.KcFeXtaTEhbb4aQCAPjFTkwEKudJo47k9BoQZ_Z6RV-H6dRZs0b6zybVhUFTbsSFwgFlQiCejkjYpdGLGi94wQ
* 💻 Code (GitHub): https://github.com/parthnonigaba/varroa-multimodal-detection

The Zenodo dataset contains processed deployment data used for evaluation, while this repository contains the full multimodal inference pipeline and trained deployment models.

---

## Overview

This project presents a real-world, non-invasive system for early detection of *Varroa destructor* infestations in honey bee colonies using multimodal sensing and machine learning.

The system integrates:

* Vision-based detection of bees and mites
* Audio-based classification of colony health
* Environmental sensing using CO₂, temperature, and humidity

These signals are combined using a confidence-weighted fusion framework to generate infestation scores and early warnings.

---

## System Architecture

The system operates in three stages:

1. Vision module detects bees and classifies varroa presence using YOLO-based models
2. Audio module analyzes hive acoustics using a CNN-LSTM classifier
3. Environmental module evaluates CO₂, temperature, and humidity using a Random Forest model

Outputs from each modality are combined through confidence-weighted fusion to produce a final infestation score.

---

## Repository Structure

* `src/` — deployment and monitoring code
* `models/` — trained model artifacts used in the deployed system
* `config/` — example configuration file
* `notebooks/` — training and experimentation notebooks
* `sample_data/` — processed deployment datasets

---

## Sample Data

The `sample_data/` folder contains processed field deployment datasets:

* `healthy_hive_30day_processed.csv`
* `mild_hive_30day_processed.csv`
* `infested_hive_1day_processed.csv`

These correspond to healthy, mildly infested, and clearly infested colony conditions.

---

## Running the System

1. Review and adapt the configuration file:

   config/config.example.json

2. Install dependencies:

   pip install -r requirements.txt

3. Run the deployment system:

   python src/main_monitor.py

---

## Reproducibility

This repository provides:

* Full multimodal deployment pipeline (vision, audio, environmental sensing)
* Trained model artifacts used during evaluation
* Processed datasets aligned with reported experimental results

Due to storage constraints, raw continuous audio/video streams and intermediate training data are not included. The provided datasets and models are sufficient to reproduce system-level performance and fusion behavior.

---

## Notes

* This repository is intended to accompany the paper and public dataset release
* Included model files are deployment-ready artifacts, not full training checkpoints
* The system is designed for real-time, field-deployable monitoring on embedded hardware

---
