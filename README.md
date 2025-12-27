# Bias as a Feature for Generative AI Detection
This project investigates whether bias-related content features can improve the detection of AI-generated text beyond traditional stylometric baselines, and how these two feature families fail and succeed under different conditions.

Overview
- 1_individual_features.ipynb and .pdf: Code used during the development of the individual features. Shows the actual measurements for the differences between Human and AI for each feature.
- 2_running_models.ipynb and .pdf: Code that executes the baseline model (lightweight stylometric features), bias model (bias-related content-level features) and merged model.
- 3_custom_dataset_creation.ipynb and .pdf: Code used to compile my own dataset
- features.py: Contains the features

Data:
- As a main dataset, I used the PAN 2025 Voight-Kampff Subtask 1 dataset, for which you can request access from: https://zenodo.org/records/14962653
- As a secondary dataset, I compiled a custom migration-focused dataset for controlled in-domain analysis

To run the code, make sure to modify the paths
