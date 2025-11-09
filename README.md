# Melting Point Prediction of Organic Crystals using Hybrid Machine Learning and Graph Neural Networks

This repository contains the code implementation and data processing scripts for the study titled "Hybrid Machine Learning and Graph Neural Network Approaches for Accurate Melting Point Prediction of Organic Crystals" by Dubonos et al. The project focuses on developing and benchmarking hybrid machine learning and graph neural network models for predicting melting points of organic crystals from molecular representations.

## Overview

Melting point prediction plays a critical role in chemistry, materials science, and pharmaceuticals for compound identification, purity assessment, and design of thermally stable materials. This work presents:

- Curated a large, harmonized dataset of 117,752 unique organic crystal melting points from multiple public sources.
- Evaluated classical machine learning models using molecular embeddings derived from SMILES strings (ChemBERTa, MoLFormer, Morgan fingerprints, Uni-Mol).
- Developed advanced graph neural network (GNN) architectures with message passing and attention mechanisms.
- Proposed a hybrid TabM model integrating transformer-based embeddings for improved performance.
- Benchmarked against the state-of-the-art Chemprop framework.

## Repository Contents

- **data_preprocessing/**: Scripts to clean, harmonize, and prepare molecular datasets.
- **embeddings/**: Code to generate molecular embeddings using ChemBERTa, MoLFormer, Morgan fingerprints, and Uni-Mol.
- **models/**
  - Classical ML models using XGBoost
  - Deep learning models including TabM and GNN architectures
  - Fine-tuning scripts for transformer-based models
- **training/**: Training and evaluation pipelines with hyperparameter tuning and performance metrics.
- **results/**: Performance comparisons, metrics (MAE, RMSE), and computational efficiency reports.
- **utils/**: Utility functions for data handling, metric calculations, and visualization.

## Key Results

- Hybrid TabM model combining ChemBERTa and MoLFormer embeddings achieved a mean absolute error (MAE) of 26.5 K, outperforming individual models.
- Graph neural networks with attention mechanisms improved accuracy but required longer training time.
- Chemprop achieved the highest raw predictive performance but with significantly increased computational cost.
- The dataset and models provide a robust framework for accelerating materials and pharmaceutical development.

## Requirements

- Python 3.8+
- PyTorch
- RDKit
- XGBoost
- Additional dependencies listed in `requirements.txt`

## Usage

1. Prepare dataset using scripts in `data_preprocessing/`.
2. Generate molecular embeddings from SMILES.
3. Train models via scripts in `training/`.
4. Evaluate and compare model performance using provided metrics.

## Citation

If you use this code or dataset, please cite:

Dubonos et al., "Hybrid Machine Learning and Graph Neural Network Approaches for Accurate Melting Point Prediction of Organic Crystals," [Journal/Preprint details], 2025.

## Contact

For questions and collaboration, contact Alexander S. Novikov at novikovradio.chem.msu.ru.
