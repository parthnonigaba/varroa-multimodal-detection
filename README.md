# A Multimodal Machine Learning System for Non-Invasive Detection of Varroa Destructor Infestations in Honey Bee Colonies

This repository contains deployment code, trained model artifacts, research notebooks, and processed sample datasets for a multimodal honey bee colony monitoring system.

## System Overview

The system integrates three sensing modalities:

- Vision for bee and varroa detection
- Audio for colony health classification
- Environmental sensing using CO2, temperature, and humidity

These modality outputs are combined through confidence-weighted fusion to generate infestation scores and alerts.

## Repository Structure

- src/ : deployment and monitoring code
- models/ : trained model artifacts used by the deployment system
- config/ : example configuration file
- notebooks/ : training and experimentation notebooks
- sample_data/ : processed deployment datasets

## Sample Data

The sample_data/ folder contains three processed field deployment datasets:

- healthy_hive_30day_processed.csv
- mild_hive_30day_processed.csv
- infested_hive_1day_processed.csv

These files correspond to healthy, mildly infested, and clearly infested colony conditions.

## Running the System

1. Review and adapt the configuration file in config/config.example.json
2. Install dependencies with:

   pip install -r requirements.txt

3. Run the deployment system with:

   python src/main_monitor.py

## Notes

- This repository is intended to accompany the paper and public dataset release.
- Some raw deployment media and other large intermediate files are excluded for size constraints.
- Included model files are the deployment artifacts, not full training dumps.
