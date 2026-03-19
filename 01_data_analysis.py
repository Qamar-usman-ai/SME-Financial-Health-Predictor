"""
SME Financial Health Prediction - Data Analysis Module
========================================================
Comprehensive exploratory data analysis of training dataset
Analyzes missing values, data types, distributions, and feature characteristics

Author: Data Science Team
Version: 1.0.0
Last Updated: March 2026
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Configuration
np.random.seed(42)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# =========================================================
# 1. LOAD DATA
# =========================================================
def load_data(base_path='data/'):
    """
    Load training dataset for analysis
    
    Parameters:
    -----------
    base_path : str
        Path to data directory
        
    Returns:
    --------
    pd.DataFrame
        Loaded training dataset
    """
    train_file = os.path.join(base_path, 'Train.csv')
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data not found at {train_file}")
    
    print(f"Loading data from: {train_file}")
    df = pd.read_csv(train_file)
    print(f"✓ Data loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
    
    return df


# =========================================================
# 2. MISSING VALUE ANALYSIS
# =========================================================
def analyze_missing_values(df):
    """
    Comprehensive analysis of missing values
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Analysis results sorted by missing percentage
    """
    print("\n" + "=" * 80)
    print("MISSING VALUE ANALYSIS")
    print("=" * 80)
    
    analysis_results = []
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        unique_count = df[col].nunique()
        
        analysis_results.append({
            'Column': col,
            'Data Type': df[col].dtype,
            'Missing Count': missing_count,
            'Missing %': round(missing_percentage, 2),
            'Unique Count': unique_count,
            'Non-Null Count': len(df) - missing_count
        })
    
    # Convert to dataframe and sort by missing percentage
    analysis_df = pd.DataFrame(analysis_results)
    analysis_df = analysis_df.sort_values('Missing %', ascending=False)
    
    return analysis_df


# =========================================================
# 3. DATA TYPE ANALYSIS
# =========================================================
def analyze_data_types(df):
    """
    Analyze data types and their distribution
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Dictionary with data type statistics
    """
    print("\n" + "=" * 80)
    print("DATA TYPE ANALYSIS")
    print("=" * 80)
    
    dtype_counts = df.dtypes.value_counts()
    print("\nData Type Distribution:")
    print(dtype_counts)
    
    stats = {
        'numeric_cols': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categorical_cols': df.select_dtypes(include=['object']).columns.tolist(),
        'numeric_count': len(df.select_dtypes(include=['int64', 'float64']).columns),
        'categorical_count': len(df.select_dtypes(include=['object']).columns)
    }
    
    print(f"\nNumeric Features: {stats['numeric_count']}")
    print(f"Categorical Features: {stats['categorical_count']}")
    
    return stats


# =========================================================
# 4. DETAILED FEATURE ANALYSIS
# =========================================================
def analyze_feature_details(df):
    """
    Detailed analysis for each feature
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Detailed feature information
    """
    print("\n" + "=" * 80)
    print("DETAILED FEATURE ANALYSIS")
    print("=" * 80)
    
    feature_info = {}
    
    for col in df.columns:
        col_data = df[col].dropna()
        
        info = {
            'dtype': df[col].dtype,
            'missing': df[col].isnull().sum(),
            'missing_pct': round(df[col].isnull().mean() * 100, 2),
            'unique': col_data.nunique(),
        }
        
        # Numeric features
        if df[col].dtype in ['int64', 'float64']:
            info.update({
                'min': col_data.min(),
                'max': col_data.max(),
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'skewness': col_data.skew(),
                'kurtosis': col_data.kurtosis()
            })
        
        # Categorical features
        else:
            top_values = col_data.value_counts().head(3)
            info['top_values'] = top_values.to_dict()
        
        feature_info[col] = info
    
    return feature_info


# =========================================================
# 5. TARGET VARIABLE ANALYSIS
# =========================================================
def analyze_target(df):
    """
    Analyze target variable distribution
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.Series
        Target value counts
    """
    print("\n" + "=" * 80)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 80)
    
    target_counts = df['Target'].value_counts()
    target_pcts = df['Target'].value_counts(normalize=True) * 100
    
    print("\nTarget Distribution:")
    for label in target_counts.index:
        count = target_counts[label]
        pct = target_pcts[label]
        print(f"  {label:8s}: {count:6,d} ({pct:6.2f}%)")
    
    print(f"\nClass Imbalance Ratio (High:Low): {target_counts['High']/target_counts['Low']:.3f}")
    
    return target_counts


# =========================================================
# 6. NUMERIC FEATURES STATISTICS
# =========================================================
def numeric_features_summary(df):
    """
    Summary statistics for numeric features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics
    """
    print("\n" + "=" * 80)
    print("NUMERIC FEATURES SUMMARY STATISTICS")
    print("=" * 80)
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    summary = df[numeric_cols].describe().T
    
    # Add skewness and kurtosis
    summary['skewness'] = df[numeric_cols].skew()
    summary['kurtosis'] = df[numeric_cols].kurtosis()
    
    print("\n", summary.round(3))
    
    return summary


# =========================================================
# 7. CATEGORICAL FEATURES SUMMARY
# =========================================================
def categorical_features_summary(df):
    """
    Summary for categorical features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Categorical feature summaries
    """
    print("\n" + "=" * 80)
    print("CATEGORICAL FEATURES SUMMARY")
    print("=" * 80)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    cat_summary = {}
    
    for col in categorical_cols:
        if col != 'Target':  # Skip target for now
            value_counts = df[col].value_counts()
            cat_summary[col] = {
                'unique': df[col].nunique(),
                'missing': df[col].isnull().sum(),
                'top_3': value_counts.head(3).to_dict()
            }
            
            print(f"\n{col}:")
            print(f"  Unique values: {df[col].nunique()}")
            print(f"  Missing: {df[col].isnull().sum()}")
            print(f"  Top 3 values:")
            for val, count in value_counts.head(3).items():
                pct = (count / len(df)) * 100
                print(f"    {val:30s}: {count:6,d} ({pct:5.2f}%)")
    
    return cat_summary


# =========================================================
# 8. CORRELATION ANALYSIS (Numeric Features)
# =========================================================
def correlation_analysis(df):
    """
    Correlation analysis for numeric features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Correlation matrix
    """
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS (Numeric Features)")
    print("=" * 80)
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = df[numeric_cols].corr()
    
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))
    
    # Find highly correlated pairs
    print("\n\nHighly Correlated Pairs (|corr| > 0.7, excluding self-correlation):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                print(f"  {col1:25s} <-> {col2:25s}: {corr_val:7.3f}")
    
    return corr_matrix


# =========================================================
# 9. COUNTRY ANALYSIS
# =========================================================
def country_analysis(df):
    """
    Analyze data distribution by country
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Country statistics
    """
    print("\n" + "=" * 80)
    print("GEOGRAPHIC DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    country_stats = df.groupby('country').agg({
        'ID': 'count',
        'Target': lambda x: x.value_counts().to_dict(),
        'owner_age': ['mean', 'std'],
        'business_age_years': ['mean', 'std'],
        'business_turnover': ['mean', 'median']
    }).round(2)
    
    print("\nSample Count by Country:")
    country_counts = df['country'].value_counts()
    for country, count in country_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {country:15s}: {count:6,d} ({pct:6.2f}%)")
    
    print("\nTarget Distribution by Country:")
    target_by_country = pd.crosstab(df['country'], df['Target'], margins=True)
    print(target_by_country)
    
    return country_stats


# =========================================================
# 10. OWNER DEMOGRAPHICS ANALYSIS
# =========================================================
def demographics_analysis(df):
    """
    Analyze owner demographics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Demographics statistics
    """
    print("\n" + "=" * 80)
    print("OWNER DEMOGRAPHICS ANALYSIS")
    print("=" * 80)
    
    # Age analysis
    print("\nOwner Age Distribution:")
    age_stats = df['owner_age'].describe()
    print(age_stats.round(2))
    
    # Gender analysis
    print("\n\nOwner Gender Distribution:")
    gender_counts = df['owner_sex'].value_counts()
    for gender, count in gender_counts.items():
        pct = (count / df['owner_sex'].notna().sum()) * 100
        print(f"  {gender:15s}: {count:6,d} ({pct:6.2f}%)")
    
    # Age groups
    print("\n\nAge Group Distribution:")
    age_bins = [0, 25, 35, 45, 55, 65, 110]
    age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    age_groups = pd.cut(df['owner_age'], bins=age_bins, labels=age_labels)
    age_group_counts = age_groups.value_counts().sort_index()
    for group, count in age_group_counts.items():
        pct = (count / age_group_counts.sum()) * 100
        print(f"  {group:10s}: {count:6,d} ({pct:6.2f}%)")
    
    return {
        'age_stats': age_stats,
        'gender_counts': gender_counts,
        'age_groups': age_group_counts
    }


# =========================================================
# 11. BUSINESS CHARACTERISTICS ANALYSIS
# =========================================================
def business_analysis(df):
    """
    Analyze business characteristics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Business statistics
    """
    print("\n" + "=" * 80)
    print("BUSINESS CHARACTERISTICS ANALYSIS")
    print("=" * 80)
    
    # Business age
    print("\nBusiness Age (Years) Distribution:")
    age_stats = df['business_age_years'].describe()
    print(age_stats.round(2))
    
    # Business turnover
    print("\n\nBusiness Turnover Distribution:")
    turnover_stats = df['business_turnover'].describe()
    print(turnover_stats.round(2))
    
    # Business expenses
    print("\n\nBusiness Expenses Distribution:")
    expenses_stats = df['business_expenses'].describe()
    print(expenses_stats.round(2))
    
    # Financial records
    print("\n\nKeeps Financial Records:")
    records_counts = df['keeps_financial_records'].value_counts()
    for val, count in records_counts.items():
        pct = (count / records_counts.sum()) * 100
        print(f"  {val:25s}: {count:6,d} ({pct:6.2f}%)")
    
    return {
        'age': age_stats,
        'turnover': turnover_stats,
        'expenses': expenses_stats
    }


# =========================================================
# 12. GENERATE COMPREHENSIVE REPORT
# =========================================================
def generate_analysis_report(train_df):
    """
    Generate comprehensive analysis report
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataset
    """
    print("\n" + "=" * 80)
    print("SME FINANCIAL HEALTH PREDICTION - DATA ANALYSIS REPORT")
    print("=" * 80)
    print(f"Analysis Date: {pd.Timestamp.now()}")
    print(f"Dataset Shape: {train_df.shape[0]} rows × {train_df.shape[1]} columns")
    
    # Run all analyses
    missing_df = analyze_missing_values(train_df)
    dtypes_stats = analyze_data_types(train_df)
    feature_info = analyze_feature_details(train_df)
    target_counts = analyze_target(train_df)
    num_summary = numeric_features_summary(train_df)
    cat_summary = categorical_features_summary(train_df)
    corr_matrix = correlation_analysis(train_df)
    country_stats = country_analysis(train_df)
    demo_stats = demographics_analysis(train_df)
    biz_stats = business_analysis(train_df)
    
    return {
        'missing': missing_df,
        'dtypes': dtypes_stats,
        'features': feature_info,
        'target': target_counts,
        'numeric_summary': num_summary,
        'categorical_summary': cat_summary,
        'correlations': corr_matrix,
        'country': country_stats,
        'demographics': demo_stats,
        'business': biz_stats
    }


# =========================================================
# 13. EXPORT RESULTS
# =========================================================
def export_analysis_results(missing_df, output_dir='outputs/'):
    """
    Export analysis results to CSV files
    
    Parameters:
    -----------
    missing_df : pd.DataFrame
        Missing value analysis results
    output_dir : str
        Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'missing_value_analysis.csv')
    missing_df.to_csv(output_file, index=False)
    print(f"\n✓ Analysis exported to: {output_file}")


# =========================================================
# 14. MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    """
    Main execution function
    Loads data and generates comprehensive analysis report
    """
    
    # Load data
    train_df = load_data(base_path='data/')
    
    # Generate comprehensive report
    results = generate_analysis_report(train_df)
    
    # Export results
    export_analysis_results(results['missing'], output_dir='outputs/')
    
    print("\n" + "=" * 80)
    print("✓ DATA ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nNext Steps:")
    print("1. Review analysis results in outputs/missing_value_analysis.csv")
    print("2. Run model training: python notebooks/02_model_training.py")
    print("3. Generate predictions: python notebooks/03_predictions.py")
