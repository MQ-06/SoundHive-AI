# Comparative Study of Classical Machine Learning, Fully Connected Neural Networks, and Sequence-Based Deep Learning Models

## Abstract

This report presents a comprehensive comparison of three classification approaches applied to a single dataset: feature-engineering-based classical machine learning models, fully connected deep neural networks using raw data, and sequence-based deep learning models. Performance is evaluated using accuracy and F1-score to analyze the strengths and limitations of each approach. The study uses beehive temperature sensor data collected over one year, with the goal of classifying temperature states into four categories: Cold, Normal, Warm, and Hot. Results demonstrate that sequence-based models (LSTM) achieve superior performance (99.63% accuracy, 0.9862 F1-score) by effectively capturing temporal dependencies, while classical machine learning models with feature engineering also perform excellently (100% accuracy, 1.0 F1-score) but require manual feature extraction.

---

## 1. Introduction / Problem Statement

Classification problems can be addressed using a wide range of modeling techniques. Traditional machine learning models rely heavily on feature engineering, whereas deep learning models learn representations directly from raw data. However, fully connected neural networks often fail to capture temporal dependencies in sequential data. This project aims to compare these approaches to determine their effectiveness under different modeling paradigms.

The beehive temperature monitoring problem requires classifying temperature states to monitor colony health. Different modeling approaches offer trade-offs between:
- **Manual effort**: Feature engineering vs. automatic learning
- **Temporal understanding**: Static features vs. sequence-aware models
- **Performance**: Accuracy and generalization capability
- **Computational requirements**: Training time and resource usage

This comparative study evaluates three distinct approaches on the same dataset to provide insights into their relative strengths and limitations.

---

## 2. Related Work

Previous studies indicate that classical machine learning models perform well on structured data with carefully engineered features, while deep learning approaches outperform them on large-scale and complex datasets. Sequence-based models such as LSTM and GRU have shown superior results on time-dependent data by maintaining memory of past observations through gating mechanisms.

Research in time-series classification has demonstrated that:
- Feature-engineered models excel when domain knowledge is available
- Fully connected networks reduce feature dependency but may miss temporal patterns
- Recurrent architectures (LSTM/GRU) are particularly effective for sequential data with long-term dependencies

---

## 3. Methodology

The project follows three distinct approaches, all applied to the same beehive temperature dataset:

### Approach-1: Feature Engineering + Classical Machine Learning

**Implementation**: Deliverable 5.2

This approach combines manual feature engineering with traditional machine learning classifiers:
- **Feature Engineering**: Statistical features extracted from 60-minute windows:
  - Mean, standard deviation, min, max, median, range
  - Temporal features: hour of day, day of week (cyclical encoding)
  - Total: 12 engineered features per sample
- **Classifiers**: Logistic Regression, SVM (RBF kernel), Decision Tree, Random Forest, XGBoost/Gradient Boosting
- **Best Model**: Random Forest (selected based on performance)
- **Data Format**: 2D feature matrix (samples × features)

### Approach-2: Fully Connected Deep Neural Network

**Implementation**: Deliverable 3

This approach uses a fully connected neural network trained directly on raw input data:
- **Data Preparation**: Multiple sensor readings per timestamp flattened into feature vectors
- **Architecture**: 
  - Input layer: Variable size based on sensor readings
  - Hidden layers: Dense(256) → Dropout(0.3) → Dense(128) → Dropout(0.3) → Dense(64)
  - Output layer: Dense(4) with Softmax activation
- **Training**: Adam optimizer, 30 epochs, batch size 32
- **Data Format**: 2D matrix (samples × flattened features)

### Approach-3: Sequence-Based Deep Learning Models

**Implementation**: Deliverable 5.4

This approach uses sequence-based models (LSTM) trained on raw sequential data:
- **Data Preparation**: 
  - Raw data preserved in sequential form
  - Sliding window sequences of 15 consecutive timestamps
  - 3D format: (samples × time_steps × features)
- **Architecture**:
  - Input layer: Accepts sequential data (15 timestamps × 1 feature)
  - LSTM layer: 128 units
  - Dropout layer: 0.3
  - Dense layer: 64 units with ReLU activation
  - Output layer: Dense(4) with Softmax activation
- **Training**: Adam optimizer, 30 epochs, batch size 32, validation split 20%
- **Data Format**: 3D sequences (samples × 15 × 1)

### Common Settings Across All Approaches

- **Dataset**: Same beehive temperature data (2017)
- **Train-Test Split**: 80% training, 20% testing (stratified)
- **Target Variable**: 4-class classification (Cold, Normal, Warm, Hot)
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
- **Random Seed**: 42 (for reproducibility)

---

## 4. Experiments and Results

### 4.1 Dataset Description

- **Dataset Name**: Beehive Temperature Sensor Data
- **Source**: Kaggle (https://www.kaggle.com/datasets/se18m502/bee-hive)
- **Project**: HOBOS (HOneyBee Online Studies)
- **Data Type**: Time-series sensor data (sequential/ordered)
- **Number of Samples**: 
  - Raw data: 401,786 temperature readings
  - After preprocessing and sequence creation: Variable (depends on approach)
- **Sequence Length**: 15 time steps (for Approach-3)
- **Features per Time Step**: 1 (temperature in °C)
- **Class Labels**: 4 classes
  - **Cold**: Temperature < 10°C
  - **Normal**: Temperature 10-25°C
  - **Warm**: Temperature 25-35°C
  - **Hot**: Temperature > 35°C

The dataset contains ordered observations where temporal relationships between data points are significant for understanding hive state transitions.

### 4.2 Data Settings

#### Train-Test Split
- **Ratio**: 80% training, 20% testing
- **Method**: Stratified split (preserves class distribution)
- **Random State**: 42

#### Preprocessing Techniques

**Approach-1 (Feature Engineering)**:
- Window aggregation: 60-minute windows
- Statistical feature extraction (mean, std, min, max, etc.)
- Temporal feature encoding (cyclical hour, day of week)
- StandardScaler normalization for LR and SVM

**Approach-2 (Fully Connected DNN)**:
- Grouping by timestamp (multiple sensors per timestamp)
- Padding to uniform feature vector length
- StandardScaler normalization
- No manual feature engineering

**Approach-3 (Sequence Model)**:
- Grouping by timestamp
- Sliding window sequence creation (length=15)
- StandardScaler normalization
- Reshape to 3D format (samples × time_steps × features)
- No manual feature engineering

#### Label Encoding
- **Method**: LabelEncoder + One-Hot Encoding (for multi-class)
- **Classes**: 4 classes (Cold=0, Normal=1, Warm=2, Hot=3)

#### Normalization Methods
- **Method**: StandardScaler (mean=0, std=1)
- **Applied to**: All approaches
- **Fit on**: Training set only (to prevent data leakage)

### 4.3 Results (Comparison of All Approaches)

| Approach | Model Type | Input Type | Accuracy | F1-Score |
|----------|-----------|------------|----------|----------|
| **Feature Engineering + ML** | Random Forest | Engineered Features | **100.00%** | **1.0000** |
| **Fully Connected DNN** | Dense Neural Network | Raw Data (Flattened) | 98.47% | 0.7927 |
| **Sequence Model** | LSTM | Raw Sequential Data | 99.63% | 0.9862 |

#### Detailed Results by Approach

**Approach-1: Classical Machine Learning Models**

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | 98.46% | 0.9845 |
| SVM (RBF) | 97.88% | 0.9788 |
| Decision Tree | 100.00% | 1.0000 |
| **Random Forest** | **100.00%** | **1.0000** |
| Gradient Boosting | 100.00% | 1.0000 |

**Approach-2: Fully Connected DNN**
- Accuracy: 98.47%
- F1-Score: 0.7927
- Architecture: 256 → 128 → 64 → 4 neurons

**Approach-3: Sequence-Based Model (LSTM)**
- Accuracy: 99.63%
- F1-Score: 0.9862
- Precision: 0.9967
- Recall: 0.9768
- Sequence Length: 15 timestamps

### 4.4 Discussion

#### Feature Engineering + Classical ML (Approach-1)

**Strengths**:
- Achieved perfect performance (100% accuracy, 1.0 F1-score) with Random Forest
- Interpretable models (especially Decision Tree and Random Forest)
- Fast training and inference
- Works well when domain knowledge is available for feature engineering

**Limitations**:
- Requires significant manual effort and domain expertise
- Feature engineering is time-consuming and dataset-specific
- May not generalize well to new domains without re-engineering features
- Does not explicitly model temporal dependencies between distant time points

**Best Use Cases**: When domain knowledge is available, interpretability is important, and computational resources are limited.

#### Fully Connected DNN (Approach-2)

**Strengths**:
- No manual feature engineering required
- Learns representations automatically from raw data
- Good performance (98.47% accuracy)
- Handles multiple sensor readings per timestamp effectively

**Limitations**:
- Lower F1-score (0.7927) compared to other approaches, indicating class imbalance issues
- Does not capture temporal dependencies effectively
- Treats each sample independently, ignoring sequence context
- Requires more data and computational resources than classical ML

**Best Use Cases**: When feature engineering is difficult, but temporal patterns are not critical.

#### Sequence-Based Model - LSTM (Approach-3)

**Strengths**:
- Excellent performance (99.63% accuracy, 0.9862 F1-score)
- Effectively captures temporal dependencies across time steps
- No manual feature engineering required
- Maintains memory of past observations through gating mechanisms
- Better balanced performance across classes (higher F1-score than DNN)

**Limitations**:
- More computationally intensive than classical ML
- Requires careful tuning of sequence length
- Longer training time compared to feature-engineered models
- Less interpretable than tree-based models

**Best Use Cases**: When temporal dependencies are important, sufficient computational resources are available, and automatic feature learning is desired.

#### Comparative Analysis

1. **Performance Ranking**:
   - Best Accuracy: Random Forest (100%) = Decision Tree (100%) > LSTM (99.63%) > DNN (98.47%)
   - Best F1-Score: Random Forest (1.0) = Decision Tree (1.0) > LSTM (0.9862) > DNN (0.7927)

2. **Temporal Modeling**:
   - Only LSTM explicitly models temporal dependencies
   - Feature-engineered models capture temporal patterns through window statistics
   - DNN treats samples independently

3. **Effort vs. Performance Trade-off**:
   - High effort, high performance: Feature engineering + Random Forest
   - Medium effort, good performance: LSTM (automatic learning)
   - Low effort, moderate performance: DNN (simple architecture)

4. **Generalization**:
   - LSTM likely generalizes better to new temporal patterns
   - Feature-engineered models may require re-engineering for new domains
   - DNN may struggle with temporal shifts in data distribution

---

## 5. Conclusion and Future Work

### 5.1 Conclusion

This project demonstrated that model selection depends heavily on data characteristics and project requirements:

1. **For this specific dataset**: Feature-engineered Random Forest achieved perfect performance, but this required significant manual effort and domain knowledge.

2. **For temporal data**: Sequence-based models (LSTM) provide an excellent balance between automatic learning and performance, achieving 99.63% accuracy with minimal manual intervention.

3. **Trade-offs exist**: 
   - Classical ML offers interpretability and speed but requires feature engineering
   - Fully connected DNNs reduce feature dependency but miss temporal patterns
   - Sequence models capture temporal dependencies but require more computation

4. **Key Insight**: For time-series classification, sequence-based deep learning models (LSTM/GRU) are the most suitable when temporal dependencies are important, as they outperform fully connected networks and match or approach the performance of carefully feature-engineered models.

### 5.2 Future Work

Several directions could enhance this comparative study:

1. **Attention Mechanisms**: Implement Transformer-based architectures with self-attention to capture long-range dependencies more effectively than LSTM.

2. **Hybrid Models**: Combine feature engineering with sequence models to leverage both domain knowledge and automatic temporal learning.

3. **Automated Hyperparameter Optimization**: Use techniques like Bayesian optimization or genetic algorithms to automatically tune sequence length, model architecture, and training parameters.

4. **Ensemble Methods**: Combine predictions from multiple approaches (Random Forest + LSTM + DNN) to potentially improve robustness and generalization.

5. **Transfer Learning**: Explore pre-trained sequence models and fine-tuning for beehive monitoring tasks.

6. **Real-time Deployment**: Investigate model compression and optimization for real-time inference in IoT beehive monitoring systems.

7. **Multi-modal Learning**: Incorporate additional sensor data (humidity, weight, flow) to create multi-modal sequence models.

8. **Explainability**: Develop methods to interpret LSTM decisions, making sequence models more transparent for domain experts.

---

## References

- Dataset: Kaggle - Beehive Metrics (https://www.kaggle.com/datasets/se18m502/bee-hive)
- HOBOS Project: https://www.hobos.de/
- TensorFlow/Keras Documentation
- Scikit-learn Documentation

---

**Report Prepared For**: ITF22 Project - Deliverable 5.5  
**Date**: December 2025  
**Authors**: [Your Name/Roll Numbers]

