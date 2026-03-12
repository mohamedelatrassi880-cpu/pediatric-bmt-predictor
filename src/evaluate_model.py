import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

if __name__ == "__main__":
    print("Loading data and saved model...")
    
    # 1. Load the data
    data, meta = arff.loadarff("data/bone-marrow.arff")
    df = pd.DataFrame(data)
    
    # 2. Clean and prep the exact same way as the training script
    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].str.decode('utf-8')
        
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    df_encoded = pd.get_dummies(df, drop_first=True, dtype=int)
    if 'survival_time' in df_encoded.columns:
        df_encoded = df_encoded.drop('survival_time', axis=1)

    X = df_encoded.drop('survival_status', axis=1)
    y = df_encoded['survival_status']

    # 3. Split using the exact same random_state=42 to get the identical test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Load the saved model
    model = joblib.load("app/rf_model.pkl")
    
    # 5. Make predictions
    y_pred = model.predict(X_test)
    
    # 6. Print the detailed report
    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    # 7. Generate and save a visual Confusion Matrix
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    plt.title("Model Confusion Matrix")
    
    # Save the visual to the approved 'app/' folder to strictly follow the rubric architecture
    plt.savefig("app/confusion_matrix.png")
    print("\nSuccess! Confusion matrix saved to app/confusion_matrix.png")