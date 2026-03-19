"""
SME Financial Health Prediction - Model Training Module
========================================================
Complete machine learning pipeline for training Random Forest classifier

Features:
- Missing data handling (median imputation + Word2Vec embeddings)
- Categorical feature encoding via Word2Vec
- Random Forest model training with 800 trees
- Train-validation evaluation (90-10 split)
- Model serialization and persistence
- Comprehensive performance metrics

Author: Data Science Team
Version: 1.0.0
Last Updated: March 2026
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report, 
    confusion_matrix
)
from gensim.models import Word2Vec

# Set random seeds for reproducibility
np.random.seed(42)
pd.np.random.seed(42)

# =========================================================
# 1. CONFIGURATION
# =========================================================
class Config:
    """Configuration parameters for model training"""
    
    # Data paths
    DATA_DIR = 'data/'
    OUTPUT_DIR = 'models/'
    LOGS_DIR = 'outputs/'
    
    # Model parameters
    N_ESTIMATORS = 800
    MAX_DEPTH = None
    RANDOM_STATE = 42
    N_JOBS = -1
    
    # Word2Vec parameters
    EMBEDDING_DIM = 30
    W2V_WINDOW = 2
    W2V_MIN_COUNT = 1
    W2V_WORKERS = 4
    W2V_SG = 1
    
    # Train-validation split
    VAL_SIZE = 0.1
    
    # Missing data threshold
    MISSING_THRESHOLD = 60  # Drop columns with >60% missing


# =========================================================
# 2. DATA LOADING
# =========================================================
def load_data(data_dir=Config.DATA_DIR):
    """
    Load training and test datasets
    
    Parameters:
    -----------
    data_dir : str
        Directory containing Train.csv and Test.csv
        
    Returns:
    --------
    tuple
        (train_df, test_df)
    """
    train_path = os.path.join(data_dir, 'Train.csv')
    test_path = os.path.join(data_dir, 'Test.csv')
    
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Data files not found in {data_dir}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"✓ Training data: {train_df.shape[0]:,d} rows × {train_df.shape[1]} columns")
    print(f"✓ Test data: {test_df.shape[0]:,d} rows × {test_df.shape[1]} columns")
    
    return train_df, test_df


# =========================================================
# 3. DATA PREPROCESSING
# =========================================================
def preprocess_data(train_df, test_df, missing_threshold=Config.MISSING_THRESHOLD):
    """
    Preprocess data by removing high-missing-value columns
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataset
    test_df : pd.DataFrame
        Test dataset
    missing_threshold : float
        Drop columns with missing % > threshold
        
    Returns:
    --------
    tuple
        (train_df, test_df) - preprocessed
    """
    print("\n" + "=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)
    
    # Calculate missing percentages
    missing_percent = train_df.isnull().mean() * 100
    cols_drop = missing_percent[missing_percent > missing_threshold].index.tolist()
    
    if cols_drop:
        print(f"\nRemoving {len(cols_drop)} columns with >{missing_threshold}% missing:")
        for col in cols_drop:
            print(f"  - {col}: {missing_percent[col]:.2f}% missing")
        
        train_df = train_df.drop(columns=cols_drop)
        test_df = test_df.drop(columns=cols_drop)
    else:
        print(f"\nNo columns with >{missing_threshold}% missing data")
    
    print(f"\n✓ Training data shape: {train_df.shape}")
    print(f"✓ Test data shape: {test_df.shape}")
    
    return train_df, test_df


# =========================================================
# 4. SEPARATE FEATURES AND TARGET
# =========================================================
def separate_features_target(train_df, test_df):
    """
    Separate features and target variable
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataset with target
    test_df : pd.DataFrame
        Test dataset without target
        
    Returns:
    --------
    tuple
        (X, y, X_test, train_ids, test_ids)
    """
    print("\n" + "=" * 80)
    print("SEPARATING FEATURES AND TARGET")
    print("=" * 80)
    
    train_ids = train_df["ID"]
    test_ids = test_df["ID"]
    
    y = train_df["Target"]
    X = train_df.drop(columns=["ID", "Target"])
    X_test = test_df.drop(columns=["ID"])
    
    print(f"\nFeature matrix X: {X.shape}")
    print(f"Target vector y: {y.shape}")
    print(f"Test feature matrix X_test: {X_test.shape}")
    
    print(f"\nTarget distribution:")
    print(y.value_counts())
    
    return X, y, X_test, train_ids, test_ids


# =========================================================
# 5. IDENTIFY NUMERIC AND CATEGORICAL COLUMNS
# =========================================================
def identify_column_types(X):
    """
    Identify numeric and categorical columns
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
        
    Returns:
    --------
    tuple
        (num_cols, cat_cols)
    """
    print("\n" + "=" * 80)
    print("IDENTIFYING COLUMN TYPES")
    print("=" * 80)
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nNumeric columns: {len(num_cols)}")
    for col in num_cols:
        print(f"  - {col}")
    
    print(f"\nCategorical columns: {len(cat_cols)}")
    for col in cat_cols:
        print(f"  - {col}")
    
    return num_cols, cat_cols


# =========================================================
# 6. HANDLE NUMERIC MISSING VALUES
# =========================================================
def impute_numeric_features(X, X_test, num_cols):
    """
    Impute numeric features using median strategy
    
    Parameters:
    -----------
    X : pd.DataFrame
        Training feature matrix
    X_test : pd.DataFrame
        Test feature matrix
    num_cols : list
        Numeric column names
        
    Returns:
    --------
    tuple
        (X, X_test) - with imputed values
    """
    print("\n" + "=" * 80)
    print("IMPUTING NUMERIC FEATURES")
    print("=" * 80)
    
    if not num_cols:
        print("No numeric columns to impute")
        return X, X_test
    
    print(f"\nImputing {len(num_cols)} numeric columns using median strategy...")
    
    imputer = SimpleImputer(strategy="median")
    X[num_cols] = imputer.fit_transform(X[num_cols])
    X_test[num_cols] = imputer.transform(X_test[num_cols])
    
    print("✓ Numeric imputation complete")
    
    return X, X_test


# =========================================================
# 7. WORD2VEC EMBEDDING FOR CATEGORICAL FEATURES
# =========================================================
def train_word2vec_model(df, col, embedding_dim=Config.EMBEDDING_DIM):
    """
    Train Word2Vec model for a categorical feature
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data containing the categorical column
    col : str
        Column name
    embedding_dim : int
        Embedding dimension
        
    Returns:
    --------
    Word2Vec
        Trained Word2Vec model
    """
    # Create sentences (each value as a sentence)
    sentences = [[str(val)] for val in df[col].fillna("MISSING")]
    
    # Train Word2Vec
    model = Word2Vec(
        sentences,
        vector_size=embedding_dim,
        window=Config.W2V_WINDOW,
        min_count=Config.W2V_MIN_COUNT,
        workers=Config.W2V_WORKERS,
        sg=Config.W2V_SG,
        seed=Config.RANDOM_STATE
    )
    
    return model


def get_word2vec_embedding(model, val):
    """
    Get Word2Vec embedding for a value
    
    Parameters:
    -----------
    model : Word2Vec
        Trained Word2Vec model
    val : any
        Value to embed
        
    Returns:
    --------
    np.array
        Embedding vector
    """
    try:
        return model.wv[str(val)]
    except KeyError:
        # Return zero vector for out-of-vocabulary words
        return np.zeros(model.vector_size)


def embed_categorical_features(X, X_test, cat_cols, embedding_dim=Config.EMBEDDING_DIM):
    """
    Apply Word2Vec embeddings to categorical features
    
    Parameters:
    -----------
    X : pd.DataFrame
        Training feature matrix
    X_test : pd.DataFrame
        Test feature matrix
    cat_cols : list
        Categorical column names
    embedding_dim : int
        Embedding dimension
        
    Returns:
    --------
    tuple
        (X, X_test) - with embeddings added, original cat columns dropped
    """
    print("\n" + "=" * 80)
    print("APPLYING WORD2VEC EMBEDDINGS TO CATEGORICAL FEATURES")
    print("=" * 80)
    
    print(f"\nEmbedding {len(cat_cols)} categorical columns...")
    print(f"Embedding dimension: {embedding_dim}")
    
    for idx, col in enumerate(cat_cols, 1):
        print(f"\n  [{idx}/{len(cat_cols)}] Processing: {col}")
        
        # Train Word2Vec model for this column
        w2v_model = train_word2vec_model(X, col, embedding_dim)
        
        # Transform training data
        embeddings_train = np.array([
            get_word2vec_embedding(w2v_model, v) 
            for v in X[col].fillna("MISSING")
        ])
        
        # Add embeddings as new columns
        for i in range(embedding_dim):
            X[f"{col}_w2v_{i}"] = embeddings_train[:, i]
        
        # Transform test data
        embeddings_test = np.array([
            get_word2vec_embedding(w2v_model, v) 
            for v in X_test[col].fillna("MISSING")
        ])
        
        # Add embeddings to test data
        for i in range(embedding_dim):
            X_test[f"{col}_w2v_{i}"] = embeddings_test[:, i]
    
    # Drop original categorical columns
    X = X.drop(columns=cat_cols)
    X_test = X_test.drop(columns=cat_cols)
    
    print(f"\n✓ Categorical embedding complete")
    print(f"  Features after embedding: {X.shape[1]}")
    
    return X, X_test


# =========================================================
# 8. CREATE ADVANCED NUMERIC FEATURES
# =========================================================
def create_advanced_features(X, X_test):
    """
    Create advanced numeric features from existing features
    
    Parameters:
    -----------
    X : pd.DataFrame
        Training feature matrix
    X_test : pd.DataFrame
        Test feature matrix
        
    Returns:
    --------
    tuple
        (X, X_test) - with new features
    """
    print("\n" + "=" * 80)
    print("CREATING ADVANCED FEATURES")
    print("=" * 80)
    
    # Get only numeric columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    print(f"\nCreating aggregate features from {len(numeric_cols)} numeric columns...")
    
    for df in [X, X_test]:
        # Statistical aggregates
        df["num_mean"] = df[numeric_cols].mean(axis=1)
        df["num_std"] = df[numeric_cols].std(axis=1).fillna(0)
        df["num_min"] = df[numeric_cols].min(axis=1)
        df["num_max"] = df[numeric_cols].max(axis=1)
        df["num_range"] = df["num_max"] - df["num_min"]
        df["num_median"] = df[numeric_cols].median(axis=1)
        
        # Missing value indicator
        df["missing_count"] = df[numeric_cols].isnull().sum(axis=1)
    
    print("✓ Advanced features created:")
    print("  - num_mean, num_std, num_min, num_max, num_range, num_median")
    print("  - missing_count")
    
    return X, X_test


# =========================================================
# 9. ENCODE TARGET VARIABLE
# =========================================================
def encode_target(y):
    """
    Encode target variable to numeric labels
    
    Parameters:
    -----------
    y : pd.Series
        Target variable
        
    Returns:
    --------
    tuple
        (y_encoded, label_encoder)
    """
    print("\n" + "=" * 80)
    print("ENCODING TARGET VARIABLE")
    print("=" * 80)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"\nLabel mapping:")
    for i, label in enumerate(le.classes_):
        print(f"  {label:8s} -> {i}")
    
    return y_encoded, le


# =========================================================
# 10. TRAIN-VALIDATION SPLIT
# =========================================================
def create_train_val_split(X, y_encoded, val_size=Config.VAL_SIZE):
    """
    Create train-validation split with stratification
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y_encoded : np.array
        Encoded target
    val_size : float
        Validation set size
        
    Returns:
    --------
    tuple
        (X_train, X_val, y_train, y_val)
    """
    print("\n" + "=" * 80)
    print("CREATING TRAIN-VALIDATION SPLIT")
    print("=" * 80)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded,
        test_size=val_size,
        random_state=Config.RANDOM_STATE,
        stratify=y_encoded
    )
    
    print(f"\nTrain set: {X_train.shape[0]:,d} samples ({100*(1-val_size):.1f}%)")
    print(f"Validation set: {X_val.shape[0]:,d} samples ({100*val_size:.1f}%)")
    
    return X_train, X_val, y_train, y_val


# =========================================================
# 11. TRAIN RANDOM FOREST MODEL
# =========================================================
def train_random_forest(X_train, y_train):
    """
    Train Random Forest classifier
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : np.array
        Training labels
        
    Returns:
    --------
    RandomForestClassifier
        Trained model
    """
    print("\n" + "=" * 80)
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("=" * 80)
    
    print(f"\nModel Configuration:")
    print(f"  n_estimators: {Config.N_ESTIMATORS}")
    print(f"  max_depth: {Config.MAX_DEPTH}")
    print(f"  random_state: {Config.RANDOM_STATE}")
    print(f"  n_jobs: {Config.N_JOBS}")
    
    print(f"\nTraining on {X_train.shape[0]:,d} samples with {X_train.shape[1]} features...")
    
    model = RandomForestClassifier(
        n_estimators=Config.N_ESTIMATORS,
        max_depth=Config.MAX_DEPTH,
        random_state=Config.RANDOM_STATE,
        n_jobs=Config.N_JOBS
    )
    
    model.fit(X_train, y_train)
    
    print("✓ Model training complete")
    
    return model


# =========================================================
# 12. EVALUATE MODEL
# =========================================================
def evaluate_model(model, X_train, X_val, y_train, y_val, label_encoder):
    """
    Evaluate model on training and validation sets
    
    Parameters:
    -----------
    model : RandomForestClassifier
        Trained model
    X_train : pd.DataFrame
        Training features
    X_val : pd.DataFrame
        Validation features
    y_train : np.array
        Training labels
    y_val : np.array
        Validation labels
    label_encoder : LabelEncoder
        Label encoder for class names
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    # Training metrics
    print("\nTRAINING METRICS:")
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1w = f1_score(y_train, y_train_pred, average="weighted")
    train_f1m = f1_score(y_train, y_train_pred, average="macro")
    
    print(f"  Accuracy:     {train_acc:.4f}")
    print(f"  F1-Score (Weighted): {train_f1w:.4f}")
    print(f"  F1-Score (Macro):    {train_f1m:.4f}")
    
    # Validation metrics
    print("\nVALIDATION METRICS:")
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1w = f1_score(y_val, y_val_pred, average="weighted")
    val_f1m = f1_score(y_val, y_val_pred, average="macro")
    
    print(f"  Accuracy:     {val_acc:.4f}")
    print(f"  F1-Score (Weighted): {val_f1w:.4f}")
    print(f"  F1-Score (Macro):    {val_f1m:.4f}")
    
    # Classification report
    print("\nCLASSIFICATION REPORT:")
    target_names = label_encoder.classes_
    print(classification_report(y_val, y_val_pred, target_names=target_names))
    
    # Confusion matrix
    print("CONFUSION MATRIX:")
    cm = confusion_matrix(y_val, y_val_pred)
    print(cm)
    
    metrics = {
        'train_accuracy': train_acc,
        'train_f1_weighted': train_f1w,
        'train_f1_macro': train_f1m,
        'val_accuracy': val_acc,
        'val_f1_weighted': val_f1w,
        'val_f1_macro': val_f1m,
        'confusion_matrix': cm
    }
    
    return metrics


# =========================================================
# 13. SAVE MODEL
# =========================================================
def save_model(model, output_dir=Config.OUTPUT_DIR, filename='trained_model.pkl'):
    """
    Serialize and save trained model
    
    Parameters:
    -----------
    model : RandomForestClassifier
        Trained model
    output_dir : str
        Output directory
    filename : str
        Model filename
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, filename)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n✓ Model saved to: {model_path}")


# =========================================================
# 14. MAIN TRAINING PIPELINE
# =========================================================
def main():
    """
    Main training pipeline orchestration
    """
    print("\n" + "=" * 80)
    print("SME FINANCIAL HEALTH PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 80)
    print(f"Start Time: {pd.Timestamp.now()}\n")
    
    # 1. Load data
    train_df, test_df = load_data(Config.DATA_DIR)
    
    # 2. Preprocess
    train_df, test_df = preprocess_data(train_df, test_df, Config.MISSING_THRESHOLD)
    
    # 3. Separate features and target
    X, y, X_test, train_ids, test_ids = separate_features_target(train_df, test_df)
    
    # 4. Identify column types
    num_cols, cat_cols = identify_column_types(X)
    
    # 5. Impute numeric features
    X, X_test = impute_numeric_features(X, X_test, num_cols)
    
    # 6. Embed categorical features
    X, X_test = embed_categorical_features(X, X_test, cat_cols, Config.EMBEDDING_DIM)
    
    print(f"\n✓ Final feature matrix shape: {X.shape}")
    
    # 7. Create advanced features
    X, X_test = create_advanced_features(X, X_test)
    
    print(f"✓ Feature matrix shape after advanced features: {X.shape}")
    
    # 8. Encode target
    y_encoded, le = encode_target(y)
    
    # 9. Train-validation split
    X_train, X_val, y_train, y_val = create_train_val_split(X, y_encoded)
    
    # 10. Train model
    model = train_random_forest(X_train, y_train)
    
    # 11. Evaluate model
    metrics = evaluate_model(model, X_train, X_val, y_train, y_val, le)
    
    # 12. Save model
    save_model(model, Config.OUTPUT_DIR, 'trained_model.pkl')
    
    print("\n" + "=" * 80)
    print("✓ MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"End Time: {pd.Timestamp.now()}")
    print("\nNext Step: Generate predictions with notebook 03_predictions.py")


if __name__ == "__main__":
    main()
