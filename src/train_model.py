import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_and_evaluate(df):
    """
    Splits the data, trains 3 models with class imbalance handling, and returns evaluation metrics.
    """
    print("Preparing data for training...")
    
    # --- THE FIX: Convert all text columns into numbers (One-Hot Encoding) ---
    df_encoded = pd.get_dummies(df, drop_first=True)

    # 1. Separate Features (X) and Target (y)
    X = df_encoded.drop('survival_status', axis=1)
    y = df_encoded['survival_status']

    # 2. Split data into Training (80%) and Testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Calculate exact imbalance ratio for XGBoost
    ratio = float(y_train.value_counts()[0]) / y_train.value_counts()[1]

    # 3. Initialize the 3 required models
    models = {
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
        "SVM": SVC(class_weight='balanced', probability=True, random_state=42),
        "XGBoost": XGBClassifier(scale_pos_weight=ratio, random_state=42, eval_metric='logloss')
    }

    # 4. Train and test each model
    results = {}
    for name, model in models.items():
        # Train the AI
        model.fit(X_train, y_train)
        
        # Make predictions on the unseen test data
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] # Get probabilities for ROC-AUC

        # Grade the AI
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba)
        }
        
    # Format the grades into a clean Pandas table
    results_df = pd.DataFrame(results).T
    print("\n--- Model Evaluation Results ---")
    print(results_df.round(3))
    
    return models, results_df

# --- IGNITION SWITCH FOR TERMINAL EXECUTION ---
if __name__ == "__main__":
    import os
    import pandas as pd
    from scipy.io import arff
    
    data_path = "data/bone-marrow.arff"
    
    if os.path.exists(data_path):
        print(f"Loading ARFF data from {data_path}...")
        data, meta = arff.loadarff(data_path)
        df = pd.DataFrame(data)
        
        # Decode ARFF byte strings
        for col in df.select_dtypes([object]).columns:
            df[col] = df[col].str.decode('utf-8')
            
        # --- NEW: Clean Missing Values (NaNs) ---
        # Fills missing numbers with the median, and missing text with the most frequent value
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
                
        # Run One-Hot Encoding
        df_encoded = pd.get_dummies(df, drop_first=True, dtype=int)
        
        # Drop target leakage
        if 'survival_time' in df_encoded.columns:
            df_encoded = df_encoded.drop('survival_time', axis=1)
            
        # Train the models!
        train_and_evaluate(df_encoded)
    else:
        print(f"Error: Could not find dataset at {data_path}. Please check the path.")