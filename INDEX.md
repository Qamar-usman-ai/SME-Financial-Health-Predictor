# SME Financial Health Prediction - Complete Repository Index

**Competition Score: 0.8849 | Ranking: Top 28%**

---

## 📚 Repository Contents Overview

This is a complete, production-ready machine learning project for predicting Small and Medium-sized Enterprise (SME) financial health in Southern Africa.

### What You Get
✅ Complete ML pipeline  
✅ Data analysis scripts  
✅ Model training code  
✅ Prediction generation  
✅ Comprehensive documentation  
✅ Setup verification tools  
✅ Performance analysis  
✅ Ready for deployment  

---

## 📖 Documentation Files (Start Here!)

### 1. **README.md** - Main Documentation
- **Read this first!**
- Complete project overview
- Problem statement and objectives
- Dataset information
- Model performance metrics
- Installation instructions
- Technical stack details
- Contributing guidelines

### 2. **QUICKSTART.md** - Get Running in 5 Minutes
- Quick installation guide
- Step-by-step execution
- Expected outputs
- Troubleshooting tips
- Common questions

### 3. **PROJECT_STRUCTURE.md** - Detailed File Layout
- Complete directory tree
- File descriptions and purposes
- Data flow diagrams
- Feature statistics
- File sizes and types

### 4. **This File** - Repository Index
- Overview of all files
- Quick navigation
- File purposes
- Reading recommendations

---

## 🐍 Python Scripts (Execution Order)

### 1. **setup_verification.py** (Run First!)
**Purpose:** Verify all dependencies and configurations  
**Execution Time:** ~1 minute  
**Command:**
```bash
python setup_verification.py
```
**Checks:**
- Python version compatibility
- Required packages installed
- Directory structure
- Data files present
- Disk space availability
- System configuration

**Output:**
- Summary report
- Status of all checks
- Recommendations if issues found

---

### 2. **notebooks/01_data_analysis.py** (Optional but Recommended)
**Purpose:** Exploratory Data Analysis  
**Execution Time:** 2-3 minutes  
**Command:**
```bash
python notebooks/01_data_analysis.py
```
**Functions:**
- Load and inspect training data
- Analyze missing values
- Generate data type summaries
- Calculate statistics
- Identify feature distributions
- Analyze target variable
- Geographic and demographic breakdown
- Business characteristics analysis

**Outputs:**
- `outputs/missing_value_analysis.csv` - Data quality report
- Console printout with detailed statistics

---

### 3. **notebooks/02_model_training.py** (Required)
**Purpose:** Train the predictive model  
**Execution Time:** 15-30 minutes  
**Command:**
```bash
python notebooks/02_model_training.py
```
**Pipeline:**
1. Load training and test data
2. Remove high-missing columns (>60%)
3. Separate features and target
4. Impute numeric missing values (median)
5. Create Word2Vec embeddings for categorical features
6. Generate advanced statistical features
7. Encode target variable
8. Create train-validation split (90-10)
9. Train Random Forest classifier (800 trees)
10. Evaluate on validation set
11. Save trained model

**Key Parameters:**
```python
n_estimators = 800
max_depth = None
embedding_dim = 30
val_size = 0.1
random_state = 42
```

**Outputs:**
- `models/trained_model.pkl` - Serialized trained model
- Console: Training metrics, validation metrics, confusion matrix
- Expected Validation F1: 0.886

---

### 4. **notebooks/03_predictions.py** (Final Step)
**Purpose:** Generate predictions and create submission  
**Execution Time:** 5-10 minutes  
**Command:**
```bash
python notebooks/03_predictions.py
```
**Pipeline:**
1. Load test data
2. Load training data (for preprocessing reference)
3. Load trained model from disk
4. Apply same preprocessing as training
5. Impute numeric features
6. Create Word2Vec embeddings
7. Generate advanced features
8. Create class predictions
9. Format submission file

**Outputs:**
- `outputs/submission.csv` - Final predictions (4,283 rows)
- Format: ID, Target, (optional) Confidence
- Ready for competition submission

---

## 📊 Data Files

### Training Data: `data/Train.csv`
- **Records:** 9,618 SME businesses
- **Columns:** 39 features + ID + Target
- **Size:** 6-8 MB
- **Features:**
  - Demographics: owner_age, owner_sex, country
  - Business metrics: business_age_years, business_age_months, business_turnover, business_expenses
  - Financial products: has_loan_account, has_credit_card, has_mobile_money, has_insurance, etc.
  - Business behavior: marketing_word_of_mouth, keeps_financial_records, offers_credit_to_customers
  - Attitudes: attitude_worried_shutdown, attitude_satisfied_with_achievement
  - Access to finance: uses_informal_lender, uses_friends_family_savings, problem_sourcing_money
- **Target:** Low, Medium, High (financial health classification)

### Test Data: `data/Test.csv`
- **Records:** 4,283 SME businesses
- **Columns:** Same 39 features + ID (no Target)
- **Size:** 3-4 MB

### Generated: `outputs/submission.csv`
- **Records:** 4,283 predictions
- **Columns:** ID, Target, (optional) Confidence
- **Size:** 150-200 KB
- **Format:** CSV, ready for upload

---

## 🤖 Model & Configuration Files

### Trained Model: `models/trained_model.pkl`
- **Size:** 50-100 MB
- **Type:** Serialized RandomForestClassifier
- **Format:** Python pickle
- **Created by:** `02_model_training.py`
- **Used by:** `03_predictions.py`

### Configuration Files
- **requirements.txt** - Python dependencies with versions
- **Config classes** - Embedded in each Python script for easy modification

---

## 📈 Performance Summary

### Leaderboard Results
```
Competition: data.org Financial Health Prediction Challenge
Score: 0.8849 (F1-Score)
Ranking: Top 28% (450+ participants)
```

### Training Performance
```
Accuracy: 100.0% (perfect training fit)
F1-Score (Weighted): 1.0
F1-Score (Macro): 1.0
```

### Validation Performance (10% holdout)
```
Accuracy: 88.88%
F1-Score (Weighted): 0.8858
F1-Score (Macro): 0.8446

Per-Class Performance:
  High:   Precision=0.89, Recall=0.72, F1=0.80 (47 samples)
  Low:    Precision=0.90, Recall=0.96, F1=0.93 (628 samples) ← Best!
  Medium: Precision=0.86, Recall=0.75, F1=0.80 (287 samples)
```

---

## 🔄 Complete Workflow

```
START
  ↓
1. Verify Setup
   python setup_verification.py
  ↓
2. (Optional) Analyze Data
   python notebooks/01_data_analysis.py
   → outputs/missing_value_analysis.csv
  ↓
3. Train Model
   python notebooks/02_model_training.py
   → models/trained_model.pkl
  ↓
4. Generate Predictions
   python notebooks/03_predictions.py
   → outputs/submission.csv
  ↓
5. Submit Results
   Upload outputs/submission.csv to competition
  ↓
END (Score: 0.8849)
```

---

## ⚙️ Technical Architecture

### Feature Engineering Pipeline

```
Input Data (39 features)
    ↓
[Missing Data Handling]
    ├─ Numeric: Median imputation
    └─ Categorical: Word2Vec embedding (30D)
    ↓
[Feature Creation]
    ├─ Statistical aggregates (mean, std, min, max, median, range)
    ├─ Missing value counts
    └─ Interaction features
    ↓
[Final Features: 936 total]
    ├─ Numeric (5): original numeric features
    ├─ Embeddings (1,020): 34 features × 30 dimensions
    └─ Advanced (7): statistical aggregates + missing counts
    ↓
[Model Input]
```

### Model Architecture

```
Random Forest Classifier
├─ n_estimators: 800 trees
├─ max_depth: Unlimited
├─ Features: 936
├─ Classes: 3 (Low, Medium, High)
└─ Random State: 42 (reproducibility)
```

---

## 🎯 Key Features

### Data Handling
✅ Handles 46.67% missing data  
✅ Preserves categorical relationships  
✅ Maintains numeric distributions  
✅ Balances feature importance  

### Model Advantages
✅ No feature scaling required  
✅ Handles mixed data types  
✅ Robust to outliers  
✅ Fast prediction time  
✅ Feature importance available  

### Code Quality
✅ Comprehensive documentation  
✅ Type hints and docstrings  
✅ Error handling  
✅ Modular functions  
✅ Reproducible (fixed random seed)  
✅ Parallel processing enabled  

---

## 📋 Requirements

### System Requirements
- **Python:** 3.8 or higher
- **RAM:** 4 GB minimum, 8 GB recommended
- **Disk:** 2 GB available
- **CPU:** Multi-core processor recommended

### Dependencies (See requirements.txt)
- pandas ≥ 1.3.0
- numpy ≥ 1.21.0
- scikit-learn ≥ 0.24.0
- gensim ≥ 4.0.0
- matplotlib, seaborn (optional, for visualization)

---

## 🚀 Quick Start (3 Steps)

### Step 1: Install
```bash
pip install -r requirements.txt
```

### Step 2: Verify
```bash
python setup_verification.py
```

### Step 3: Run Complete Pipeline
```bash
python notebooks/01_data_analysis.py  # Optional
python notebooks/02_model_training.py # Required (15-30 min)
python notebooks/03_predictions.py    # Required (5-10 min)
```

**Result:** `outputs/submission.csv` ready for upload!

---

## 🔍 File Navigation Guide

### I want to...

**...understand the project**
→ Read `README.md`

**...get it running quickly**
→ Read `QUICKSTART.md`

**...understand the file structure**
→ Read `PROJECT_STRUCTURE.md`

**...verify my setup**
→ Run `python setup_verification.py`

**...analyze the data**
→ Run `python notebooks/01_data_analysis.py`

**...train a model**
→ Run `python notebooks/02_model_training.py`

**...make predictions**
→ Run `python notebooks/03_predictions.py`

**...modify the code**
→ Edit any `notebooks/*.py` file directly

**...understand the model**
→ Check `Config` class in `02_model_training.py`

---

## 📊 Output Files Generated

| File | Size | Created By | Purpose |
|------|------|-----------|---------|
| missing_value_analysis.csv | ~50 KB | 01_data_analysis.py | Data quality report |
| trained_model.pkl | 50-100 MB | 02_model_training.py | Trained model |
| submission.csv | 150-200 KB | 03_predictions.py | Competition submission |

---

## ✅ Verification Checklist

Before running, ensure:
- [ ] Python 3.8+ installed
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Setup verified: `python setup_verification.py`
- [ ] `data/Train.csv` exists (9,618 rows)
- [ ] `data/Test.csv` exists (4,283 rows)
- [ ] Directories created: data/, models/, outputs/
- [ ] ~2 GB disk space available
- [ ] Multi-core CPU available (for faster training)

---

## 🎓 Learning Path

1. **Beginner:** Read README.md → QUICKSTART.md
2. **Intermediate:** Review PROJECT_STRUCTURE.md → Run scripts in order
3. **Advanced:** Modify Config → Try different models/parameters
4. **Expert:** Add new features → Implement hyperparameter tuning → Deploy API

---

## 🤝 Common Modifications

### Adjust Model Complexity
Edit `Config.N_ESTIMATORS` in `02_model_training.py`
```python
Config.N_ESTIMATORS = 500  # Faster, less accurate
Config.N_ESTIMATORS = 1000  # Slower, more accurate
```

### Change Embedding Dimension
Edit `Config.EMBEDDING_DIM`
```python
Config.EMBEDDING_DIM = 20  # Smaller, faster
Config.EMBEDDING_DIM = 50  # Larger, slower but richer
```

### Adjust Validation Split
Edit `Config.VAL_SIZE`
```python
Config.VAL_SIZE = 0.2  # 80-20 split
Config.VAL_SIZE = 0.1  # 90-10 split (current)
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| ImportError | Run: `pip install -r requirements.txt` |
| File not found | Check: `data/Train.csv` and `data/Test.csv` exist |
| Out of memory | Reduce `Config.N_ESTIMATORS` or `Config.EMBEDDING_DIM` |
| Slow training | Ensure multi-core CPU; reduce n_estimators |
| Model not loading | Ensure `models/trained_model.pkl` exists |

---

## 📞 Support & Questions

### Getting Help
1. Check QUICKSTART.md for common issues
2. Review README.md for detailed explanations
3. Check script docstrings for function details
4. Run `setup_verification.py` for system diagnostics

### Common Questions
- **"How long does training take?"** 15-30 minutes (normal)
- **"Can I modify the model?"** Yes, edit Config classes
- **"How do I improve the score?"** Try different models/parameters
- **"Is GPU supported?"** No, current code uses CPU only

---

## 📜 License & Credits

- **Challenge Host:** data.org & FinMark Trust
- **Data Source:** Southern African SME financial inclusion research
- **Benchmark:** Top 28% of 450+ competition participants
- **Status:** Production-ready code

---

## 🎉 You're Ready!

You have everything you need to:
1. ✅ Understand the problem
2. ✅ Prepare the data
3. ✅ Train a competitive model
4. ✅ Generate predictions
5. ✅ Submit results

**Start with:** `python setup_verification.py`

---

**Last Updated:** March 2026  
**Version:** 1.0.0  
**Status:** Production Ready ✨

**Happy Modeling! 🚀**
