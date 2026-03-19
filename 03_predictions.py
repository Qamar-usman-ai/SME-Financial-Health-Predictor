"""
SME Financial Health Prediction - Predictions Module
=====================================================
Generate predictions on test dataset and create submission file

Features:
- Load trained model from disk
- Apply preprocessing to test data
- Generate class predictions
- Create submission file in competition format
- Confidence score calculation (optional)

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

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec

# Set random seeds for reproducibility
np.random.seed(42)

# =========================================================
# 1. CONFIGURATION
# =========================================================
class Config:
    """Configuration parameters for predictions"""
    
    # Data paths
    DATA_DIR = 'data/'
    MODEL_DIR = 'models/'
    OUTPUT_DIR = 'outputs/'
    
    # Model file
    MODEL_FILE = 'trained_model.pkl'
    
    # Word2Vec parameters (must match training)
    EMBEDDING_DIM = 30
    W2V_WINDOW = 2
    W2V_MIN_COUNT = 1
    W2V_WORKERS = 4
    W2V_SG = 1
    
    # Missing data threshold
    MISSING_THRESHOLD = 60


# =========================================================
# 2. LOAD DATA
# =========================================================
def load_test_data(data_dir=Config.DATA_DIR):
    """
    Load test dataset
    
    Parameters:
    -----------
    data_dir : str
        Directory containing Test.csv
        
    Returns:
    --------
    pd.DataFrame
        Test dataset
    """
    test_path = os.path.join(data_dir, 'Test.csv')
    
    print("=" * 80)
    print("LOADING TEST DATA")
    print("=" * 80)
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")
    
    test_df = pd.read_csv(test_path)
    print(f"✓ Test data loaded: {test_df.shape[0]:,d} rows × {test_df.shape[1]} columns")
    
    return test_df


def load_training_data(data_dir=Config.DATA_DIR):
    """
    Load training data for preprocessing reference
    
    Parameters:
    -----------
    data_dir : str
        Directory containing Train.csv
        
    Returns:
    --------
    pd.DataFrame
        Training dataset
    """
    train_path = os.path.join(data_dir, 'Train.csv')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")
    
    train_df = pd.read_csv(train_path)
    print(f"✓ Training data loaded: {train_df.shape[0]:,d} rows × {train_df.shape[1]} columns")
    
    return train_df


# =========================================================
# 3. LOAD TRAINED MODEL
# =========================================================
def load_model(model_dir=Config.MODEL_DIR, model_file=Config.MODEL_FILE):
    """
    Load trained model from disk
    
    Parameters:
    -----------
    model_dir : str
        Directory containing model file
    model_file : str
        Model filename
        
    Returns:
    --------
    RandomForestClassifier
        Loaded trained model
    """
    print("\n" + "=" * 80)
    print("LOADING TRAINED MODEL")
    print("=" * 80)
    
    model_path = os.path.join(model_dir, model_file)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✓ Model loaded from: {model_path}")
    print(f"✓ Model type: {type(model).__name__}")
    
    return model


# =========================================================
# 4. DATA PREPROCESSING FOR PREDICTIONS
# =========================================================
def preprocess_test_data(test_df, train_df, missing_threshold=Config.MISSING_THRESHOLD):
    """
    Apply same preprocessing to test data as training
    
    Parameters:
    -----------
    test_df : pd.DataFrame
        Test dataset
    train_df : pd.DataFrame
        Training dataset (reference)
    missing_threshold : float
        Drop columns with missing % > threshold
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed test data
    """
    print("\n" + "=" * 80)
    print("PREPROCESSING TEST DATA")
    print("=" * 80)
    
    # Calculate missing percentages based on training data
    missing_percent = train_df.isnull().mean() * 100
    cols_drop = missing_percent[missing_percent > missing_threshold].index.tolist()
    
    # Remove ID and Target if present
    cols_drop = [col for col in cols_drop if col not in ['ID', 'Target']]
    
    if cols_drop:
        print(f"\nRemoving {len(cols_drop)} high-missing columns...")
        test_df = test_df.drop(columns=cols_drop, errors='ignore')
    
    print(f"✓ Test data shape after preprocessing: {test_df.shape}")
    
    return test_df


# =========================================================
# 5. SEPARATE TEST IDs AND FEATURES
# =========================================================
def separate_test_ids_features(test_df):
    """
    Separate IDs from features
    
    Parameters:
    -----------
    test_df : pd.DataFrame
        Test dataset
        
    Returns:
    --------
    tuple
        (test_ids, X_test)
    """
    print("\n" + "=" * 80)
    print("SEPARATING TEST IDs AND FEATURES")
    print("=" * 80)
    
    test_ids = test_df["ID"]
    X_test = test_df.drop(columns=["ID"])
    
    print(f"✓ Test IDs: {len(test_ids):,d}")
    print(f"✓ Test features: {X_test.shape[0]:,d} rows × {X_test.shape[1]} columns")
    
    return test_ids, X_test


# =========================================================
# 6. IDENTIFY NUMERIC AND CATEGORICAL COLUMNS
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
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    return num_cols, cat_cols


# =========================================================
# 7. IMPUTE NUMERIC FEATURES
# =========================================================
def impute_numeric_test(X_test, num_cols, train_df):
    """
    Impute numeric features using statistics from training data
    
    Parameters:
    -----------
    X_test : pd.DataFrame
        Test feature matrix
    num_cols : list
        Numeric column names
    train_df : pd.DataFrame
        Training data for computing statistics
        
    Returns:
    --------
    pd.DataFrame
        Test data with imputed numeric values
    """
    print("\n" + "=" * 80)
    print("IMPUTING NUMERIC FEATURES")
    print("=" * 80)
    
    if not num_cols:
        print("No numeric columns to impute")
        return X_test
    
    print(f"Imputing {len(num_cols)} numeric columns using training medians...")
    
    # Get numeric columns from training data
    train_numeric = train_df.select_dtypes(include=['int64', 'float64']).columns
    num_cols_filtered = [col for col in num_cols if col in train_numeric]
    
    # Fit imputer on training data
    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_df[num_cols_filtered])
    
    # Apply to test data
    X_test[num_cols_filtered] = imputer.transform(X_test[num_cols_filtered])
    
    print("✓ Numeric imputation complete")
    
    return X_test


# =========================================================
# 8. WORD2VEC EMBEDDING FOR CATEGORICAL FEATURES
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
    sentences = [[str(val)] for val in df[col].fillna("MISSING")]
    
    model = Word2Vec(
        sentences,
        vector_size=embedding_dim,
        window=Config.W2V_WINDOW,
        min_count=Config.W2V_MIN_COUNT,
        workers=Config.W2V_WORKERS,
        sg=Config.W2V_SG,
        seed=42
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
        return np.zeros(model.vector_size)


def embed_categorical_test(X_test, train_df, cat_cols, embedding_dim=Config.EMBEDDING_DIM):
    """
    Apply Word2Vec embeddings to test categorical features
    
    Parameters:
    -----------
    X_test : pd.DataFrame
        Test feature matrix
    train_df : pd.DataFrame
        Training data for learning embeddings
    cat_cols : list
        Categorical column names
    embedding_dim : int
        Embedding dimension
        
    Returns:
    --------
    pd.DataFrame
        Test data with embeddings, original categorical columns dropped
    """
    print("\n" + "=" * 80)
    print("APPLYING WORD2VEC EMBEDDINGS")
    print("=" * 80)
    
    # Filter categorical columns that exist in training data
    train_cat = train_df.select_dtypes(include=['object']).columns
    cat_cols_filtered = [col for col in cat_cols if col in train_cat and col not in ['ID', 'Target']]
    
    print(f"\nEmbedding {len(cat_cols_filtered)} categorical columns...")
    
    for idx, col in enumerate(cat_cols_filtered, 1):
        print(f"  [{idx}/{len(cat_cols_filtered)}] {col}")
        
        # Train Word2Vec model on training data
        w2v_model = train_word2vec_model(train_df, col, embedding_dim)
        
        # Transform test data
        embeddings_test = np.array([
            get_word2vec_embedding(w2v_model, v) 
            for v in X_test[col].fillna("MISSING")
        ])
        
        # Add embeddings as new columns
        for i in range(embedding_dim):
            X_test[f"{col}_w2v_{i}"] = embeddings_test[:, i]
    
    # Drop original categorical columns
    X_test = X_test.drop(columns=cat_cols_filtered, errors='ignore')
    
    print(f"✓ Embedding complete")
    
    return X_test


# =========================================================
# 9. CREATE ADVANCED FEATURES
# =========================================================
def create_advanced_features_test(X_test):
    """
    Create advanced numeric features for test data
    
    Parameters:
    -----------
    X_test : pd.DataFrame
        Test feature matrix
        
    Returns:
    --------
    pd.DataFrame
        Test data with advanced features
    """
    print("\n" + "=" * 80)
    print("CREATING ADVANCED FEATURES")
    print("=" * 80)
    
    # Get only numeric columns
    numeric_cols = X_test.select_dtypes(include=['int64', 'float64']).columns
    
    print(f"Creating aggregate features from {len(numeric_cols)} numeric columns...")
    
    # Statistical aggregates
    X_test["num_mean"] = X_test[numeric_cols].mean(axis=1)
    X_test["num_std"] = X_test[numeric_cols].std(axis=1).fillna(0)
    X_test["num_min"] = X_test[numeric_cols].min(axis=1)
    X_test["num_max"] = X_test[numeric_cols].max(axis=1)
    X_test["num_range"] = X_test["num_max"] - X_test["num_min"]
    X_test["num_median"] = X_test[numeric_cols].median(axis=1)
    
    # Missing value indicator
    X_test["missing_count"] = X_test[numeric_cols].isnull().sum(axis=1)
    
    print("✓ Advanced features created")
    
    return X_test


# =========================================================
# 10. GENERATE PREDICTIONS
# =========================================================
def generate_predictions(model, X_test, label_encoder):
    """
    Generate class predictions on test data
    
    Parameters:
    -----------
    model : RandomForestClassifier
        Trained model
    X_test : pd.DataFrame
        Test feature matrix
    label_encoder : LabelEncoder
        Label encoder from training
        
    Returns:
    --------
    tuple
        (predicted_labels, predicted_probs)
    """
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)
    
    print(f"\nGenerating predictions on {X_test.shape[0]:,d} test samples...")
    
    # Get class predictions
    y_pred_encoded = model.predict(X_test)
    
    # Decode predictions
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
    
    # Get prediction probabilities
    y_pred_probs = model.predict_proba(X_test)
    
    print("✓ Predictions generated successfully")
    print(f"\nPrediction distribution:")
    unique, counts = np.unique(y_pred_labels, return_counts=True)
    for label, count in zip(unique, counts):
        pct = (count / len(y_pred_labels)) * 100
        print(f"  {label:8s}: {count:6,d} ({pct:6.2f}%)")
    
    return y_pred_labels, y_pred_probs


# =========================================================
# 11. CREATE SUBMISSION FILE
# =========================================================
def create_submission(test_ids, predictions, confidence=None, output_dir=Config.OUTPUT_DIR):
    """
    Create submission file in competition format
    
    Parameters:
    -----------
    test_ids : pd.Series
        Test sample IDs
    predictions : np.array
        Predicted class labels
    confidence : np.array, optional
        Prediction confidence scores
    output_dir : str
        Output directory
        
    Returns:
    --------
    pd.DataFrame
        Submission dataframe
    """
    print("\n" + "=" * 80)
    print("CREATING SUBMISSION FILE")
    print("=" * 80)
    
    submission = pd.DataFrame({
        "ID": test_ids,
        "Target": predictions
    })
    
    # Add confidence if provided
    if confidence is not None:
        submission["Confidence"] = confidence.max(axis=1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save submission
    submission_path = os.path.join(output_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    
    print(f"\n✓ Submission file created: {submission_path}")
    print(f"\nSubmission Preview (first 10 rows):")
    print(submission.head(10))
    
    print(f"\nSubmission Statistics:")
    print(f"  Total predictions: {len(submission):,d}")
    print(f"  File size: {os.path.getsize(submission_path) / 1024:.2f} KB")
    
    return submission


# =========================================================
# 12. MAIN PREDICTION PIPELINE
# =========================================================
def main():
    """
    Main prediction pipeline orchestration
    """
    print("\n" + "=" * 80)
    print("SME FINANCIAL HEALTH PREDICTION - PREDICTION PIPELINE")
    print("=" * 80)
    print(f"Start Time: {pd.Timestamp.now()}\n")
    
    # 1. Load data
    train_df = load_training_data(Config.DATA_DIR)
    test_df = load_test_data(Config.DATA_DIR)
    
    # 2. Preprocess test data
    test_df = preprocess_test_data(test_df, train_df, Config.MISSING_THRESHOLD)
    
    # 3. Separate IDs and features
    test_ids, X_test = separate_test_ids_features(test_df)
    
    # 4. Identify column types
    num_cols, cat_cols = identify_column_types(X_test)
    
    # 5. Impute numeric features
    X_test = impute_numeric_test(X_test, num_cols, train_df)
    
    # 6. Embed categorical features
    X_test = embed_categorical_test(X_test, train_df, cat_cols, Config.EMBEDDING_DIM)
    
    print(f"\n✓ Test feature matrix shape: {X_test.shape}")
    
    # 7. Create advanced features
    X_test = create_advanced_features_test(X_test)
    
    print(f"✓ Test feature matrix shape after advanced features: {X_test.shape}")
    
    # 8. Load trained model
    model = load_model(Config.MODEL_DIR, Config.MODEL_FILE)
    
    # 9. Create label encoder (for inverse transform)
    # Get classes from training data Target variable
    le = LabelEncoder()
    le.fit(train_df['Target'])
    
    # 10. Generate predictions
    predictions, probabilities = generate_predictions(model, X_test, le)
    
    # 11. Create submission file
    submission = create_submission(test_ids, predictions, probabilities, Config.OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("✓ PREDICTION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"End Time: {pd.Timestamp.now()}")
    print(f"\n📝 Submission file ready: {os.path.join(Config.OUTPUT_DIR, 'submission.csv')}")
    print("✅ File ready for competition submission!")


if __name__ == "__main__":
    main()
