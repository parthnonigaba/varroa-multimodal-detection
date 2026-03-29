# Multimodal Varroa Mite Detection System

A real-time, non-invasive system for early detection of Varroa destructor infestations in honey bee colonies using multimodal machine learning.

## Why This Matters

Varroa mites are a leading cause of honey bee colony collapse. Current detection methods are manual, invasive, and often too late.

This system enables continuous, automated monitoring to detect infestations before irreversible damage occurs.

## System Overview

The system integrates three sensing modalities:

- Vision: bee and varroa detection using YOLO models  
- Audio: hive health classification using neural networks  
- Environmental: CO2, temperature, and humidity-based risk modeling  

Outputs from each modality are combined using a confidence-weighted fusion approach to generate infestation alerts.

## Performance

- Precision: 99.5 percent  
- Recall: 92.1 percent  
- Median early detection lead time: 4.6 days before critical intervention threshold  

## Repository Structure

- src/ : real-time monitoring and deployment system  
- models/ : trained model artifacts used in deployment  
- config/ : example configuration file  
- notebooks/ : training and experimentation pipelines  
- sample_data/ : processed deployment datasets  

## Sample Data

The sample_data/ folder includes:

- healthy_hive_30day_processed.csv  
- mild_hive_30day_processed.csv  
- infested_hive_1day_processed.csv  

These datasets represent real-world colony conditions across different infestation levels.

## Running the System

1. Review config/config.example.json  
2. Install dependencies:

   pip install -r requirements.txt  

3. Run:

   python src/main_monitor.py  

## Notes

- Designed for Raspberry Pi deployment with sensors and camera  
- Large raw media files are excluded  
- Included models are deployment-ready artifacts, not full training outputs  

