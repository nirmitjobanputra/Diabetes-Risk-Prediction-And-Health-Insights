# Diabetes Risk Prediction and Health Insights

A comprehensive machine learning project for predicting diabetes risk and segmenting populations for targeted public health interventions using the CDC Diabetes Health Indicators Dataset.

## 📋 Project Overview

This project implements:
- **Classification Models**: XGBoost and Random Forest for individual diabetes risk prediction with probability outputs
- **Clustering Models**: K-Means and DBSCAN for population health segmentation
- **Public Health Recommendations**: Targeted intervention strategies based on cluster profiles

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or download this repository**

2. **Create a virtual environment (recommended)**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

5. **Open and run** `Diabetes_Risk_Prediction_Analysis.ipynb`

### Alternative: Install packages individually
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn jupyter
```

## 📁 Project Structure

```
Diabetes-Risk-Prediction-And-Health-Insights/
│
├── CDC Diabetes Dataset.csv          # Source dataset (253,682 records)
├── Diabetes_Risk_Prediction_Analysis.ipynb  # Main analysis notebook
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## 📊 Dataset

The **CDC Diabetes Health Indicators Dataset** contains:
- **253,682 survey responses** from the Behavioral Risk Factor Surveillance System (BRFSS)
- **22 features** including health indicators, lifestyle factors, and demographics
- **Target variable**: `Diabetes_012` (0 = No Diabetes, 1 = Prediabetes, 2 = Diabetes)

### Key Features
| Feature | Description |
|---------|-------------|
| HighBP | High Blood Pressure (0/1) |
| HighChol | High Cholesterol (0/1) |
| BMI | Body Mass Index |
| Smoker | Ever smoked 100+ cigarettes (0/1) |
| PhysActivity | Physical activity in past 30 days (0/1) |
| Fruits | Consume fruit 1+ times/day (0/1) |
| Veggies | Consume vegetables 1+ times/day (0/1) |
| GenHlth | General Health (1-5 scale) |
| Age | Age category (1-13) |

## 🔬 Analysis Pipeline

1. **Data Loading & Exploration** - Initial data assessment
2. **Data Quality Assessment** - Missing values, duplicates, distributions
3. **Exploratory Data Analysis** - Correlations, visualizations
4. **Feature Engineering** - HealthRiskScore, LifestyleScore, etc.
5. **Classification Models** - XGBoost & Random Forest with hyperparameter tuning
6. **Model Evaluation** - ROC curves, PR curves, calibration plots
7. **Probability Predictions** - Risk categorization (Low/Medium/High)
8. **Clustering Analysis** - K-Means & DBSCAN segmentation
9. **Public Health Recommendations** - Targeted intervention strategies

## 📈 Key Outputs

### Classification
- Diabetes probability predictions for individuals
- Risk category assignment (Low/Medium/High Risk)
- Feature importance rankings
- Model performance comparison

### Clustering
- 4 distinct population segments (K-Means)
- Density-based groupings with outlier detection (DBSCAN)
- Cluster profiles with health/lifestyle characteristics
- Targeted public health campaign recommendations

## 🛠️ Troubleshooting

### ModuleNotFoundError
If you get import errors, ensure all packages are installed:
```bash
pip install -r requirements.txt
```

### Kernel Issues in VS Code
1. Open Command Palette (Ctrl+Shift+P)
2. Select "Python: Select Interpreter"
3. Choose your virtual environment

### Memory Issues
The dataset is large. If you encounter memory problems:
- Reduce the clustering sample size in the notebook
- Close other applications

## 📝 License

This project is for educational purposes.

## 🙏 Acknowledgments

- CDC Behavioral Risk Factor Surveillance System (BRFSS)
- UCI Machine Learning Repository