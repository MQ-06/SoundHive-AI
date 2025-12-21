# Deliverable 5.4: Sequence-Based Deep Learning Classification

## Overview
Implementation of sequence-based deep learning models (LSTM/GRU/RNN) for time-series classification using raw sequential data.

## Project Structure
```
deliverable_5_4/
├── config.py          # Configuration settings
├── data_prep.py       # Data preparation and sequence creation
├── model_builder.py   # Model architecture builders
├── train.py           # Main training script
├── outputs/           # Results and visualizations
└── README.md         # This file
```

## Usage

### Run the model:
```bash
python train.py
```

### Change model type:
Edit `config.py` and set `MODEL_TYPE = "LSTM"` (or "GRU" or "RNN")

### Adjust sequence length:
Edit `config.py` and set `SEQUENCE_LENGTH = 15` (or desired value)

## Outputs
- `results.csv` - Performance metrics
- `training_history.png` - Training curves
- `confusion_matrix.png` - Confusion matrix visualization

## Requirements
- tensorflow
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

