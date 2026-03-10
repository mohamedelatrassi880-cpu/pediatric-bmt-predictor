# Pediatric BMT Success Predictor

A decision-support application developed to assist physicians in predicting the success rate of bone marrow transplants in pediatric patients. This project emphasizes model accuracy, explainability (via SHAP), and a clean web interface.

## How to Run the Project
This project is fully reproducible. To run the application locally, follow these steps:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
2. **Train the machine learning model:**
   ```bash
   python src/train_model.py
3. **Launch the Streamlit web application:**
   ```bash
   streamlit run app/app.py

## Critical Analysis & Findings

1. Was the dataset balanced? If not, how did you handle imbalance? and what was the impact?
The dataset was moderately imbalanced (~60% survived, 40% not survived). I addressed this by implementing a Class-weight adjustment strategy during model training. The impact was highly positive, allowing the model to prioritize the minority class without requiring synthetic data generation, resulting in balanced precision and recall.

2. Which ML model performed best? Provide performance metrics.
After removing target leakage to ensure an honest predictive model, XGBoost slightly outperformed Random Forest in overall ROC-AUC, though they tied in accuracy.

    XGBoost ROC-AUC: 0.745

    Random Forest ROC-AUC: 0.732

    Accuracy: 76.3% (Both)

    F1-Score: 0.727 (Both)

    Precision: 0.750 (Both)

3. Which medical features most influenced predictions (SHAP results)?
After removing target leakage (survival_time), the SHAP explainability analysis revealed that the top three pre-transplant features influencing the model's predictions are:

    Relapse: The patient's relapse history.

    PLTrecovery: Platelet recovery time.

    CD34kgx10d6: The CD34+ cell dose.

4. What insights did prompt engineering provide for your selected task?
Prompt engineering was critical for debugging complex library versioning issues and handling data formatting. For example, I used iterative prompting to troubleshoot a matrix dimensionality error in the shap library caused by a recent update to TreeExplainer. I also used it to quickly write a data parser using scipy.io to automatically decode the raw .arff dataset format and dynamically clean missing values (NaNs) so the SVM model wouldn't crash during pipeline execution.