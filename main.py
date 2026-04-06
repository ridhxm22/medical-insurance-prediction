"""
Medical Insurance Cost Prediction - Main Script
This script loads data, preprocesses it, trains multiple models, and evaluates their performance.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import sys

# Import custom modules
from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model, evaluate_model
from src.utils import plot_correlation_matrix, plot_actual_vs_predicted, print_metrics

def main():
    """Main function to run the medical insurance prediction pipeline."""
    
    print("="*60)
    print("Medical Insurance Cost Prediction")
    print("="*60)
    
    # Step 1: Load Data
    print("\n[Step 1] Loading data...")
    try:
        df = load_data('data/insurance.csv')
        print(f"✓ Data loaded successfully! Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError:
        print("✗ Error: data/insurance.csv not found!")
        print("Please ensure the dataset is in the data/ folder.")
        sys.exit(1)
    
    # Step 2: Data Preprocessing
    print("\n[Step 2] Preprocessing data...")
    df_processed = preprocess_data(df.copy())
    print(f"✓ Data preprocessed! New shape: {df_processed.shape}")
    
    # Step 3: Prepare Features and Target
    print("\n[Step 3] Preparing features and target...")
    X = df_processed.drop('charges', axis=1)
    y = df_processed['charges']
    print(f"✓ Features shape: {X.shape}, Target shape: {y.shape}")
    
    # Step 4: Train-Test Split
    print("\n[Step 4] Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✓ Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Step 5: Feature Scaling
    print("\n[Step 5] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features scaled successfully!")
    
    # Step 6: Train Models
    print("\n[Step 6] Training models...")
    
    # Linear Regression
    print("  - Training Linear Regression...")
    lr_model = train_model(X_train_scaled, y_train, LinearRegression())
    lr_pred = lr_model.predict(X_test_scaled)
    print("    ✓ Linear Regression trained!")
    
    # Random Forest
    print("  - Training Random Forest...")
    rf_model = train_model(X_train_scaled, y_train, RandomForestRegressor(n_estimators=100, random_state=42))
    rf_pred = rf_model.predict(X_test_scaled)
    print("    ✓ Random Forest trained!")
    
    # Step 7: Evaluate Models
    print("\n[Step 7] Evaluating models...")
    print("\n--- Linear Regression Results ---")
    print_metrics(y_test, lr_pred)
    
    print("--- Random Forest Results ---")
    print_metrics(y_test, rf_pred)
    
    print("\n" + "="*60)
    print("Pipeline Execution Completed!")
    print("="*60)

if __name__ == "__main__":
    main()