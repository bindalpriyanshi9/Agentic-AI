import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# Load data
data = pd.read_csv('D:\agentic AI\customer_churn_dataset.csv')
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Save model
joblib.dump((model, X_train, X_test, y_train, y_test), 'churn_model.pkl')

