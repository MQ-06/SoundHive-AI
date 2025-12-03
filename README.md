# ITF22 - Time Series Data Analysis & Preprocessing
## Assignment 5.1

This project contains the implementation of Assignment 5.1 for ITF22 course, focusing on sensor-based time series data analysis and preprocessing.

## 📊 Dataset

**Source**: [Beehive Metrics - Kaggle](https://www.kaggle.com/datasets/se18m502/bee-hive)  
**Project**: HOBOS (HOneyBee Online Studies)

### Sensor Types
- **Temperature Sensors**: 13 sensors monitoring hive temperature
- **Humidity Sensors**: Environmental humidity monitoring
- **Weight Sensors**: Hive weight scale measurements
- **Flow Sensors**: Bee traffic (arrivals/departures)

### Dataset Details
- **Time Period**: 2017-2019
- **Locations**: Wurzburg and Schwartau hives
- **Format**: CSV files with timestamps
- **Samples**: 400,000+ time-series readings

## 📁 Project Structure

```
ML_PROJECT/
├── data/
│   └── raw/
│       ├── new_ds/              # Sensor dataset (temperature, humidity, weight, flow)
│       ├── zenodo_bee_dataset/  # Audio dataset (archived)
│       └── archive/             # Additional audio data (archived)
├── notebooks/
│   └── assignment_5_1.ipynb     # Main assignment notebook
├── archive_bee_audio_project/   # Previous bee audio ML project (archived)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🎯 Assignment Objectives

1. **Dataset Selection**: Find and describe a sensor-based time series dataset
2. **Data Loading**: Load dataset into Python environment
3. **Timestamp Parsing**: Convert time columns to datetime format
4. **Chronological Sorting**: Arrange data in time order
5. **Missing Value Handling**: Deal with gaps in the data
6. **Data Cleaning**: Remove duplicates, fix outliers, rename columns
7. **Documentation**: Explain preprocessing steps and their importance

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Assignment
Open and run the Jupyter notebook:
```bash
jupyter notebook notebooks/assignment_5_1.ipynb
```

## 📝 Dataset Description

The dataset comes from beehive monitoring sensors that track environmental conditions and hive health indicators. The sensors collect:

- **Temperature**: Measured in °C from multiple sensors inside the hive
- **Humidity**: Relative humidity percentage
- **Weight**: Hive weight in kilograms (includes bees, honey, and structure)
- **Flow**: Number of bees entering/exiting the hive

These measurements help beekeepers monitor colony health, detect swarming behavior, and optimize hive management.

## 📚 References

- HOBOS Project: https://www.hobos.de/
- Dataset: https://www.kaggle.com/datasets/se18m502/bee-hive
- Course: ITF22 - Time Series Data Analysis

---

**Note**: The `archive_bee_audio_project` folder contains a previous bee health monitoring project using audio analysis. It has been archived to keep this assignment focused and clean.
