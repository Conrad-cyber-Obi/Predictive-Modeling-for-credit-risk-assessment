import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb

# Step 1: Data Collection
# Assuming data is in a CSV file named 'credit_data.csv'
# Load the dataset
data = pd.read_csv('credit_data.csv')

# Step 2: Data Preprocessing
# Handling missing values
# Impute numerical features with median, categorical with most frequent
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = data.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Step 3: Feature Selection
# Assuming all features are relevant after preprocessing

# Step 4: Model Building
# Split the data into training and testing sets
X = data.drop('target', axis=1)  # Replace 'target' with the actual target variable name
y = data['target']  # Replace 'target' with the actual target variable name

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
log_reg = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LogisticRegression())])

# Random Forest
rf = Pipeline(steps=[('preprocessor', preprocessor),
                     ('classifier', RandomForestClassifier(random_state=42))])

# Gradient Boosting
gb = Pipeline(steps=[('preprocessor', preprocessor),
                     ('classifier', GradientBoostingClassifier(random_state=42))])

# XGBoost
xgboost = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', xgb.XGBClassifier(random_state=42))])

# Train models
models = {'Logistic Regression': log_reg,
          'Random Forest': rf,
          'Gradient Boosting': gb,
          'XGBoost': xgboost}

# Step 5: Evaluation
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"Model: {name}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))
    print("="*60)

# Step 6: Business Impact Analysis
# Assuming that an analysis report would be written outside this script
print("Predictive modeling improves risk management by identifying high-risk borrowers, allowing for more informed lending decisions and better allocation of financial resources.")
