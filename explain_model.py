import joblib
import shap
import matplotlib.pyplot as plt

# Load model and data
model, X_train, X_test, _, _ = joblib.load('D:\agentic AI\customer_churn_dataset.csv')

# Create SHAP explainer
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Explain 1st prediction
shap.plots.waterfall(shap_values[0])

# Summary plot for feature importance
shap.plots.beeswarm(shap_values)
