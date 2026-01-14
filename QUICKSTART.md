# Quick Start Guide

Get started with the Beehive Health Monitoring ML project in minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd ML_PROJECT
   ```

2. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate (Windows)
   venv\Scripts\activate
   
   # Activate (Linux/Mac)
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Run

### Step 1: Preprocess Data

```bash
python scripts/preprocess.py
```

This will:
- Load raw data from `data/raw/new_ds/temperature_2017.csv`
- Preprocess and clean the data
- Save to `data/processed/temperature_cleaned.csv`

### Step 2: Train Classical ML Models

```bash
python scripts/train_classical_ml.py
```

This will:
- Extract features from preprocessed data
- Train 5 classical ML models (LR, SVM, DT, RF, XGBoost)
- Generate results and visualizations
- Save results to `results/` directory

### Step 3: Train Deep Learning Model

```bash
python scripts/train_dnn.py
```

This will:
- Create feature vectors from raw data
- Build and train a Fully Connected DNN
- Generate training history and confusion matrix
- Save results to `results/` directory

## Using Python API

### Preprocessing

```python
from src.data import load_data, preprocess_timeseries

# Load and preprocess
df = load_data("data/raw/new_ds/temperature_2017.csv")
df_clean = preprocess_timeseries(df)
df_clean.to_csv("data/processed/temperature_cleaned.csv", index=False)
```

### Feature Engineering

```python
from src.features import extract_time_series_features, create_labels
import pandas as pd

# Load preprocessed data
df = pd.read_csv("data/processed/temperature_cleaned.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# Extract features
features = extract_time_series_features(df, window_size='60min')
labels = create_labels(features, n_classes=3)
```

### Train Models

```python
from src.models.classical_ml import train_random_forest
from sklearn.model_selection import train_test_split

# Prepare data
X = features[['temp_mean', 'temp_std', ...]]  # Select features
y = labels

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
model, results = train_random_forest(
    X_train.values, y_train.values,
    X_test.values, y_test.values
)

print(f"Accuracy: {results['accuracy']:.4f}")
```

## Expected Output

After running all scripts, you should have:

```
results/
├── figures/
│   ├── rf_confusion_matrix.png
│   ├── rf_feature_importances.png
│   ├── dnn_training_history.png
│   └── dnn_confusion_matrix.png
└── tables/
    ├── classical_ml_results.csv
    └── dnn_results.csv
```

## Troubleshooting

### Import Errors
- Make sure you're in the project root directory
- Verify virtual environment is activated
- Check that all dependencies are installed: `pip install -r requirements.txt`

### Data Not Found
- Ensure data files are in `data/raw/new_ds/`
- Check file paths in scripts match your data location

### Memory Issues
- Reduce batch size in DNN training: `--batch-size 16`
- Use smaller window size: `--window 120min`

## Next Steps

1. Explore the code in `src/` to understand the implementation
2. Modify hyperparameters in scripts for experimentation
3. Add your own models or features
4. Check out `docs/` for detailed documentation

## Getting Help

- Check `README.md` for comprehensive documentation
- Review `docs/dataset_description.md` for dataset info
- Examine code docstrings for API details

Happy coding! 🐝

