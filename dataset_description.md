# Dataset Description - ITF22 Assignment 5.1

## Dataset Information

### Source
- **Name**: Beehive Metrics
- **Platform**: Kaggle
- **URL**: https://www.kaggle.com/datasets/se18m502/bee-hive
- **Project**: HOBOS (HOneyBee Online Studies)
- **License**: Public dataset, freely available

### Dataset Overview
This dataset contains sensor measurements from beehive monitoring systems deployed in Germany. The data was collected as part of the HOBOS project, which aims to study honey bee colony behavior and health through continuous environmental monitoring.

---

## Sensor Types

The dataset includes measurements from multiple sensor types:

### 1. Temperature Sensors
- **Type**: Digital temperature sensors
- **Purpose**: Monitor internal hive temperature
- **Number of sensors**: 13 sensors placed at different locations within the hive
- **Measurement unit**: Degrees Celsius (°C)
- **Importance**: Temperature regulation is critical for bee colony health and brood development

### 2. Humidity Sensors
- **Type**: Relative humidity sensors
- **Purpose**: Track moisture levels inside the hive
- **Measurement unit**: Percentage (%)
- **Importance**: Humidity affects honey production and colony comfort

### 3. Weight Sensors
- **Type**: Load cell / scale
- **Purpose**: Measure total hive weight
- **Measurement unit**: Kilograms (kg)
- **Importance**: Weight changes indicate honey production, bee population, and foraging activity

### 4. Flow Sensors
- **Type**: Bee traffic counters
- **Purpose**: Count bees entering and exiting the hive
- **Measurement unit**: Number of bees
- **Importance**: Activity patterns indicate colony health and foraging behavior

---

## Collected Variables

### Primary Dataset Used: `temperature_2017.csv`

| Variable | Type | Description | Unit |
|----------|------|-------------|------|
| `timestamp` | datetime | Date and time of measurement | YYYY-MM-DD HH:MM:SS |
| `temperature` | float | Average temperature reading | °C (Celsius) |

### Additional Available Datasets

- `humidity_2017.csv` - Humidity measurements (8,737 samples)
- `weight_2017.csv` - Hive weight measurements (524,110 samples)
- `flow_2017.csv` - Bee traffic data (arrivals/departures)
- Similar files for 2018-2019 and different hive locations (Wurzburg, Schwartau)

---

## Number of Samples

### Temperature Dataset (Primary)
- **Total samples**: 401,869 rows
- **Time period**: January 1, 2017 - December 31, 2017
- **Sampling frequency**: Approximately hourly readings
- **Missing values**: 3 missing temperature readings (0.0007% of data)
- **Data completeness**: 99.9993%

### Dataset Characteristics
- **Duration**: Full year (365 days)
- **Temporal coverage**: Continuous monitoring
- **Data quality**: High-quality sensor data with minimal missing values
- **Consistency**: Regular time intervals between measurements

---

## Data Collection Context

### Location
- **Country**: Germany
- **Hive locations**: Wurzburg and Schwartau
- **Environment**: Natural outdoor beehive settings

### Collection Method
- **Automated sensors**: Continuous data logging
- **Data storage**: Centralized database
- **Quality control**: Automated validation and error checking

### Use Cases
This dataset is valuable for:
- Understanding bee colony behavior patterns
- Detecting anomalies in hive conditions
- Predicting swarming events
- Monitoring colony health
- Studying environmental impacts on bees
- Time series analysis and forecasting

---

## Preprocessing Steps Applied

The following preprocessing steps were performed on this dataset:

### 1. Timestamp Parsing
- Converted timestamp strings to datetime objects
- Enables proper time-based operations and analysis

### 2. Chronological Sorting
- Verified and ensured data is sorted by timestamp
- Required for time series algorithms and calculations

### 3. Missing Value Handling
- Method: Linear interpolation
- Filled 3 missing temperature values
- Maintains temporal continuity in sensor readings

### 4. Data Cleaning
- Checked for and removed duplicate records
- Verified data ranges (temperature values are reasonable)
- Validated outliers using IQR method
- Ensured data quality and consistency

---

## Dataset Suitability for Assignment

This dataset is ideal for ITF22 Assignment 5.1 because:

✅ **Public source**: Freely available on Kaggle  
✅ **Sensor-based**: Real physical sensor measurements  
✅ **Time series**: Continuous temporal data with timestamps  
✅ **Multiple variables**: Temperature, humidity, weight, flow  
✅ **Sufficient samples**: 400,000+ data points  
✅ **Real-world data**: Contains missing values and requires preprocessing  
✅ **Well-documented**: Clear variable descriptions and metadata  
✅ **Relevant application**: Environmental monitoring for bee health  

---

## References

- HOBOS Project: https://www.hobos.de/
- Kaggle Dataset: https://www.kaggle.com/datasets/se18m502/bee-hive
- Related Research: Bee colony monitoring and environmental sensing

---

**Document prepared for**: ITF22 Assignment 5.1 - Time Series Data Analysis & Preprocessing  
**Date**: December 2025
