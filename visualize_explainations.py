import joblib
import lime
import lime.lime_tabular
import numpy as np

# Load model and data
model, X_train, X_test, _, _ = joblib.load('D:\agentic AI\customer_churn_dataset.csv')

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['No Churn', 'Churn'],
    mode='classification'
)

# Explain a prediction
exp = explainer.explain_instance(
    data_row=X_test.iloc[1],
    predict_fn=model.predict_proba
)
exp.show_in_notebook()
