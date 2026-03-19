# SME Financial Health Prediction System
## Predicting Financial Well-Being of Small Businesses in Southern Africa

**Competition Score: 0.8849 (Top 28%)**

---

## 📋 Project Overview

This project develops a machine learning system to predict the **Financial Health Index (FHI)** of Small and Medium-sized Enterprises (SMEs) across Southern Africa. The FHI classifies businesses into three categories: **Low**, **Medium**, or **High** financial health based on socio-economic, business, and financial data.

### Problem Statement
Across Southern Africa, SMEs are crucial for employment and economic growth but face significant challenges:
- Limited access to credit
- Unstable cash flow
- Exposure to economic shocks
- Exclusion from formal financial systems

Traditional metrics (revenue, profit) don't capture true financial health. This project provides a holistic measure reflecting:
- **Savings and assets**
- **Debt and repayment ability**
- **Resilience to shocks**
- **Access to credit and financial services**

### Dataset Information
- **Source Countries**: Eswatini, Lesotho, Zimbabwe, Malawi
- **Total Records**: 9,618 SME survey responses
- **Features**: 39 (socio-economic, business, and financial indicators)
- **Target Classes**: Low, Medium, High (multi-class classification)
- **Missing Data**: 35 of 39 features contain missing values (range: 0.02% - 46.67%)

---

## 🎯 Model Performance

### Validation Metrics (10% Holdout)
| Metric | Value |
|--------|-------|
| **Accuracy** | 88.88% |
| **Weighted F1-Score** | 0.8858 |
| **Macro F1-Score** | 0.8446 |

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **High** | 0.89 | 0.72 | 0.80 | 47 |
| **Low** | 0.90 | 0.96 | 0.93 | 628 |
| **Medium** | 0.86 | 0.75 | 0.80 | 287 |

### Leaderboard Score
- **Public Score**: 0.8849
- **Ranking**: Top 28% (among 450+ participants)

---

## 📁 Project Structure

```
SME-Financial-Health-Prediction/
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── notebooks/
│   ├── 01_data_analysis.py       # Exploratory data analysis
│   ├── 02_model_training.py      # Model development and training
│   └── 03_predictions.py         # Final predictions and submission
├── data/
│   ├── Train.csv                 # Training dataset (9,618 records)
│   ├── Test.csv                  # Test dataset (4,283 records)
│   └── submission.csv            # Final predictions
├── models/
│   └── trained_model.pkl         # Serialized Random Forest model
└── outputs/
    └── analysis_report.csv       # Data analysis results
```

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/SME-Financial-Health-Prediction.git
cd SME-Financial-Health-Prediction
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Data
Place the training and test CSV files in the `data/` directory:
```
data/
├── Train.csv
└── Test.csv
```

---

## 🚀 Quick Start

### Run Complete Pipeline
```bash
# 1. Exploratory Data Analysis
python notebooks/01_data_analysis.py

# 2. Train Model
python notebooks/02_model_training.py

# 3. Generate Predictions
python notebooks/03_predictions.py
```

### Run Individual Steps

**Step 1: Data Analysis & Exploration**
```bash
python notebooks/01_data_analysis.py
```
- Generates comprehensive missing value analysis
- Creates data quality report
- Outputs: `outputs/missing_value_analysis.csv`

**Step 2: Model Training**
```bash
python notebooks/02_model_training.py
```
- Handles missing data via median imputation
- Creates Word2Vec embeddings for 34 categorical features
- Trains Random Forest classifier with 800 trees
- Generates validation metrics and confusion matrix
- Outputs: `models/trained_model.pkl`, validation report

**Step 3: Generate Predictions**
```bash
python notebooks/03_predictions.py
```
- Loads trained model
- Generates predictions on test set
- Creates submission file
- Outputs: `submission.csv` (ready for competition upload)

---

## 📊 Data Analysis Summary

### Missing Data Overview
- **Total Features**: 39
- **Features with Missing Data**: 35 (89.7%)
- **Average Missing Percentage**: 21.67%
- **Maximum Missing Percentage**: 46.67% (informal lenders)

### High Missing Features (>20%)
These features were analyzed but handled via appropriate imputation strategies:

| Feature | Missing % | Strategy |
|---------|-----------|----------|
| `uses_informal_lender` | 46.67% | Word2Vec Embedding |
| `uses_friends_family_savings` | 46.66% | Word2Vec Embedding |
| `motivation_make_more_money` | 44.61% | Word2Vec Embedding |
| `funeral_insurance` | 43.54% | Word2Vec Embedding |
| `medical_insurance` | 43.54% | Word2Vec Embedding |

### Data Types
- **Numeric Features**: 5 (business age, owner age, income, expenses, turnover)
- **Categorical Features**: 34 (financial products, business behaviors, attitudes)

### Geographic Distribution
- **Eswatini, Lesotho, Zimbabwe, Malawi**: Balanced representation
- Encoded as one-hot features in final model

---

## 🤖 Model Architecture

### Feature Engineering Pipeline

#### 1. **Missing Data Handling**
- **Numeric Features**: Median imputation (preserves distribution)
- **Categorical Features**: Embedded via Word2Vec (captures semantic meaning)

#### 2. **Categorical Feature Embedding (Word2Vec)**
- **Embedding Dimension**: 30
- **Applied to**: 34 categorical features
- **Resulting Features**: 34 × 30 = 1,020 embeddings
- **Advantage**: Captures relationships between categorical values without losing information

#### 3. **Derived Numeric Features**
Created from numeric columns:
- `num_mean`: Mean across numeric features
- `num_std`: Standard deviation
- `num_min`: Minimum value
- `num_max`: Maximum value
- `missing_count`: Count of missing values per record

#### 4. **Final Feature Space**
- **Total Features**: 936 (after categorical transformation)
- **Features Used**: All numeric + Word2Vec embeddings

### Model: Random Forest Classifier

**Hyperparameters:**
```python
{
    'n_estimators': 800,          # Number of trees
    'max_depth': None,             # Unlimited tree depth
    'random_state': 42,            # Reproducibility
    'n_jobs': -1                   # Parallel processing
}
```

**Why Random Forest?**
- Handles mixed data types effectively
- Provides feature importance scores
- Robust to outliers and missing data
- No feature scaling required
- Prevents overfitting via ensemble averaging

---

## 📈 Key Findings

### Model Strengths
1. **Excellent "Low" Class Performance** (F1: 0.93)
   - Most important for financial institutions
   - Identifies high-risk businesses effectively

2. **High Precision on All Classes** (0.86-0.90)
   - False positives minimized
   - Reliable for policy decision-making

3. **Generalization** (Val F1: 0.89 vs Train F1: 1.0)
   - 11% gap indicates healthy learning
   - Not severely overfitting despite perfect training accuracy

### Model Limitations
1. **Class Imbalance Effect**
   - "High" class underrepresented (47 samples)
   - Lower recall for "High" class (0.72)
   - Recommended: Threshold adjustment or class weights for production use

2. **Feature Dependency**
   - 46% missing in some features
   - Embedding-based approach adds complexity
   - Could explore alternative imputation (KNN, MICE)

---

## 🔄 Prediction Workflow

### Input Format
Training and test files with 39 features:
- ID (unique identifier)
- Country (eswatini, lesotho, malawi, zimbabwe)
- Owner demographics (age, sex)
- Business metrics (age, turnover, expenses)
- Financial products (loan, credit card, mobile money, etc.)
- Business behaviors (marketing, record-keeping, credit offering)
- Attitudes (worry, ambition, satisfaction)

### Output Format
```csv
ID,Target
ID_5EGLKX,Low
ID_4AI7RE,Low
ID_V9OB3M,Low
ID_6OI9DI,Medium
```

---

## 🛠️ Technical Stack

| Component | Tool |
|-----------|------|
| **Language** | Python 3.8+ |
| **Data Processing** | pandas, NumPy |
| **ML Framework** | scikit-learn |
| **Text Embeddings** | Gensim (Word2Vec) |
| **Model Serialization** | pickle |
| **Testing** | pytest (optional) |

---

## 📊 File Descriptions

### `notebooks/01_data_analysis.py`
Comprehensive exploratory data analysis including:
- Missing value analysis
- Data type detection
- Unique value counts
- Numeric statistics (mean, median, range)
- Correlation analysis
- Output: `missing_value_analysis.csv`

### `notebooks/02_model_training.py`
End-to-end model training pipeline:
- Data loading and preprocessing
- Missing data imputation
- Word2Vec embedding generation
- Train-validation split (90-10)
- Model training and validation
- Performance metrics calculation
- Model serialization

### `notebooks/03_predictions.py`
Final prediction and submission generation:
- Load trained model
- Apply preprocessing to test data
- Generate class predictions
- Create submission file in required format

---

## 🎓 Learning Resources

### Data for Social Impact
- [data.org Platform](https://www.data.org)
- Courses: Fintech Literacy, Responsible Data Management, Ethical AI

### Financial Inclusion
- [FinMark Trust](https://www.finmarktrust.org.zw) - Regional financial inclusion research
- Focus: Making financial markets work for people in poverty

### Competition Context
- [India AI Impact Summit 2026](https://indiaaissummit.in/)
- Global collaboration on AI for social good

---

## 📝 Usage Examples

### Example 1: Train New Model
```python
from notebooks.train_model import train_model
model, metrics = train_model(train_path='data/Train.csv')
print(f"Validation F1: {metrics['f1_weighted']:.4f}")
```

### Example 2: Make Predictions
```python
from notebooks.make_predictions import predict
predictions = predict(
    test_path='data/Test.csv',
    model_path='models/trained_model.pkl'
)
predictions.to_csv('submission.csv', index=False)
```

### Example 3: Analyze Data
```python
from notebooks.data_analysis import analyze_data
analysis = analyze_data(train_path='data/Train.csv')
print(f"Missing values: {analysis['missing_percent'].describe()}")
```

---

## 🔐 Model Deployment

### Save Trained Model
```python
import pickle
with open('models/trained_model.pkl', 'wb') as f:
    pickle.dump(trained_model, f)
```

### Load and Use
```python
import pickle
with open('models/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)
predictions = model.predict(X_test)
```

---

## 🤝 Contributing

Improvements welcome! Consider:
1. **Alternative imputation methods** (KNN Impute, MICE)
2. **Hyperparameter optimization** (GridSearch, Bayesian)
3. **Ensemble methods** (XGBoost, LightGBM, Voting)
4. **Feature selection** (correlation, mutual information)
5. **Class imbalance handling** (SMOTE, class weights)
6. **Cross-validation** (Stratified K-Fold)

---

## 📄 License

This project is part of the data.org Financial Health Prediction Challenge for Southern African SMEs.

---

## 👥 Competition & Credits

- **Challenge Host**: data.org & FinMark Trust
- **Participants**: 450+
- **Final Score**: 0.8849 (F1-Score)
- **Rank**: Top 28%

---

## 📞 Support

For questions or issues:
1. Check documentation in README
2. Review code comments in notebooks
3. Refer to dataset exploration report
4. Consult scikit-learn and Gensim documentation

---

## 🎯 Next Steps for Production

1. **Hyperparameter Tuning**: GridSearch CV for optimal params
2. **Stratified K-Fold**: Better validation strategy
3. **SHAP Values**: Model interpretability
4. **API Development**: REST endpoint for predictions
5. **Monitoring**: Track model performance over time
6. **Retraining Pipeline**: Automated model updates

---

**Last Updated**: March 2026  
**Version**: 1.0.0  
**Python Version**: 3.8+
