# A Multimodal Machine Learning System for Non-Invasive Detection of Varroa Destructor Infestations in Honey Bee Colonies

This repository contains deployment code, trained model artifacts, notebooks, and processed sample datasets for a multimodal honey bee colony monitoring system that uses:

- vision-based varroa detection
- audio-based colony health analysis
- environmental sensing using CO2, temperature, and humidity
- confidence-weighted fusion across modalities

## Repository structure

- `src/` deployment and monitoring code
- `models/` trained model artifacts used by the deployment system
- `config/` example configuration file
- `notebooks/` research and training notebooks
- `sample_data/` processed deployment datasets for healthy, mildly infested, and heavily infested colonies

## Sample datasets

The `sample_data/` folder contains processed field deployment CSV files corresponding to:

- healthy colony, 30 days
- mildly infested colony, 30 days
- heavily infested colony, 1 day

## Notes

This repository is intended to accompany the paper and public dataset release. Some raw deployment media and large intermediate files are not included.
