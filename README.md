# 🐝 Beehive Health Monitoring using Machine Learning

A comprehensive machine learning project for analyzing beehive sensor data to monitor colony health, detect anomalies, and predict hive conditions using time-series analysis, classical ML algorithms, and deep learning.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a complete machine learning pipeline for monitoring beehive health using sensor data. The system processes time-series data from multiple sensors (temperature, humidity, weight, flow) and applies various ML techniques to classify hive conditions and detect anomalies.

**Key Objectives:**
- Preprocess and clean time-series sensor data
- Extract meaningful features from raw sensor readings
- Build baseline models using classical ML algorithms
- Implement deep learning models for improved accuracy
- Provide insights for beekeepers to monitor colony health

## ✨ Features

- **Data Preprocessing**: Comprehensive time-series data cleaning and preparation
  - Timestamp parsing and chronological sorting
  - Missing value handling with interpolation
  - Outlier detection and data validation

- **Feature Engineering**: Advanced feature extraction from time-series data
  - Statistical features (mean, std, min, max, median, skewness)
  - Temporal features (hour, day of week with cyclical encoding)
  - Window-based aggregation (60-minute windows)

- **Classical ML Models**: Multiple baseline classifiers
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Decision Tree
  - Random Forest
  - XGBoost/Gradient Boosting

- **Deep Learning**: Fully Connected Neural Network
  - Multi-layer architecture with dropout regularization
  - Handles both binary and multi-class classification

- **Visualization**: Comprehensive result visualization
  - Label distribution plots
  - Confusion matrices
  - Training history plots
  - Feature importance analysis

## 📊 Dataset

**Source**: [Beehive Metrics - Kaggle](https://www.kaggle.com/datasets/se18m502/bee-hive)  
**Project**: HOBOS (HOneyBee Online Studies)

### Dataset Details

- **Time Period**: 2017-2019
- **Locations**: Wurzburg and Schwartau hives (Germany)
- **Total Samples**: 400,000+ time-series readings
- **Sensor Types**:
  - **Temperature**: 13 sensors monitoring hive temperature (°C)
  - **Humidity**: Environmental humidity monitoring (%)
  - **Weight**: Hive weight measurements (kg)
  - **Flow**: Bee traffic counters (arrivals/departures)

### Data Characteristics

- **Sampling Rate**: Approximately hourly readings
- **Data Quality**: High-quality sensor data with minimal missing values
- **Preprocessing**: Required (timestamp parsing, sorting, missing value handling)

For detailed dataset information, see [docs/dataset_description.md](docs/dataset_description.md)

## 📁 Project Structure

```
ML_PROJECT/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
├── .gitignore               # Git ignore rules
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── data/                 # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── preprocess.py
│   ├── features/             # Feature engineering
│   │   ├── __init__.py
│   │   └── engineering.py
│   ├── models/               # ML models
│   │   ├── __init__.py
│   │   ├── classical_ml.py
│   │   └── deep_learning.py
│   └── utils/                # Utilities
│       ├── __init__.py
│       └── visualization.py
│
├── scripts/                  # Executable scripts
│   ├── preprocess.py        # Data preprocessing pipeline
│   ├── train_classical_ml.py # Train classical ML models
│   └── train_dnn.py          # Train deep learning model
│
├── notebooks/                # Jupyter notebooks (optional)
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_classical_ml.ipynb
│   └── 04_deep_learning.ipynb
│
├── data/                     # Data directory
│   ├── raw/                  # Raw data files
│   └── processed/            # Processed data files
│
├── models/                   # Saved models
│   └── saved_models/
│
├── results/                  # Results and outputs
│   ├── figures/             # Visualization plots
│   └── tables/              # Result tables
│
└── docs/                     # Documentation
    └── dataset_description.md
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ML_PROJECT
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### 1. Data Preprocessing

Preprocess raw sensor data:

```bash
python scripts/preprocess.py
```

Or use the Python API:

```python
from src.data import load_data, preprocess_timeseries

# Load data
df = load_data("data/raw/new_ds/temperature_2017.csv")

# Preprocess
df_processed = preprocess_timeseries(df)

# Save
df_processed.to_csv("data/processed/temperature_cleaned.csv", index=False)
```

### 2. Feature Engineering

Extract features from time-series data:

```python
from src.features import extract_time_series_features, create_labels

# Extract features
features_df = extract_time_series_features(
    df_processed,
    value_column='temperature',
    window_size='60min'
)

# Create labels
labels = create_labels(features_df, value_column='temp_mean', n_classes=3)
```

### 3. Train Classical ML Models

Train baseline classifiers:

```bash
python scripts/train_classical_ml.py
```

Or programmatically:

```python
from src.models.classical_ml import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost
)
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train models
lr_model, lr_results = train_logistic_regression(X_train, y_train, X_test, y_test)
rf_model, rf_results = train_random_forest(X_train, y_train, X_test, y_test)
```

### 4. Train Deep Learning Model

Train a fully connected neural network:

```bash
python scripts/train_dnn.py
```

Or programmatically:

```python
from src.models.deep_learning import build_dnn, train_dnn
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build model
model = build_dnn(
    input_dim=X_train_scaled.shape[1],
    num_classes=3,
    hidden_layers=[256, 128, 64]
)

# Train model
trained_model, history = train_dnn(
    model,
    X_train_scaled, y_train,
    X_test_scaled, y_test,
    epochs=30,
    batch_size=32
)
```

## 📈 Results

### Model Performance

The project includes comprehensive evaluation metrics for all models:

- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and macro-averaged precision
- **Recall**: Per-class and macro-averaged recall
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrices**: Detailed classification breakdown

Results are saved in `results/tables/` and visualizations in `results/figures/`.

### Key Findings

- Random Forest and XGBoost typically achieve the best performance among classical ML models
- Deep Neural Networks can capture complex patterns in the time-series data
- Feature engineering significantly improves model performance
- Temporal features (hour, day of week) are important for classification

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Classical machine learning algorithms
- **XGBoost**: Gradient boosting framework
- **TensorFlow/Keras**: Deep learning framework
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter**: Interactive notebooks (optional)

## 📚 Documentation

- [Dataset Description](docs/dataset_description.md): Detailed dataset information
- Code documentation: Available as docstrings in source files

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **HOBOS Project**: For providing the beehive sensor dataset
- **Kaggle**: For hosting the dataset
- Open-source ML community for excellent libraries and tools

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for beekeepers and the environment**
