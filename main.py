# %% Import
import pandas as pd
df = pd.read_csv('data/heart_disease_uci.csv')
df.head()
# %% Inspect
df.shape
df.columns
df.info()
# %% Create target and drop leaky/useless columns
df['target'] = (df['num'] > 0).astype(int)

X = df.drop(['id', 'num', 'target', 'dataset'], axis=1)
y = df['target']

print(y.value_counts(normalize=True))
# %%
# %% Separate column types
categorical_cols = [col for col in X.columns if X[col].dtype == "object" or X[col].dtype.name == "str"]
numerical_cols = [col for col in X.columns if X[col].dtype in ["int64", "float64"]]

print("Categorical:", categorical_cols)
print("Numerical:", numerical_cols)
# %%
# %% Build preprocessing pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

numerical_transformer = SimpleImputer(strategy='median')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])
# %% Split
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=42, stratify=y
)

# %% Train
clf.fit(X_train, y_train)
# %% Evaluate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

preds = clf.predict(X_valid)
print("Accuracy:", accuracy_score(y_valid, preds))
print(classification_report(y_valid, preds))
print(confusion_matrix(y_valid, preds))
# %% Cross-validate
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print("Fold scores:", cv_scores)
print("Mean accuracy:", cv_scores.mean())
print("Std:", cv_scores.std())
# %%
# %% Try XGBoost for comparison
from xgboost import XGBClassifier

xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.05, random_state=42)

xgb_clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])

xgb_cv_scores = cross_val_score(xgb_clf, X, y, cv=5, scoring='accuracy')
print("XGBoost fold scores:", xgb_cv_scores)
print("Mean accuracy:", xgb_cv_scores.mean())
print("Std:", xgb_cv_scores.std())
# %%
# %% Check if missing-data rows behave differently
df['has_missing'] = df[['ca', 'thal', 'slope']].isnull().any(axis=1)
print(df['has_missing'].value_counts())
print(df.groupby('has_missing')['target'].mean())
# %%
# %% Add has_missing as a real feature
X = df.drop(['id', 'num', 'target', 'dataset'], axis=1).copy()
X['has_missing'] = df[['ca', 'thal', 'slope']].isnull().any(axis=1)

categorical_cols = [col for col in X.columns if X[col].dtype == "object" or X[col].dtype.name == "str"]
numerical_cols = [col for col in X.columns if X[col].dtype in ["int64", "float64", "bool"]]

print("Categorical:", categorical_cols)
print("Numerical:", numerical_cols)
# %%
# %% Rebuild pipeline with the new feature
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])                  
# %%
# %% Re-run cross-validation with the new feature included
cv_scores_v2 = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print("Fold scores:", cv_scores_v2)
print("Mean accuracy:", cv_scores_v2.mean())
print("Std:", cv_scores_v2.std())
# %%
# %% Check disease-class recall with the final model
from sklearn.metrics import make_scorer, recall_score

disease_recall_scorer = make_scorer(recall_score, pos_label=1)

recall_cv_scores = cross_val_score(clf, X, y, cv=5, scoring=disease_recall_scorer)
print("Fold scores:", recall_cv_scores)
print("Mean disease recall:", recall_cv_scores.mean())
print("Std:", recall_cv_scores.std())
# %%
# %% Try higher weight on the disease class specifically
from sklearn.ensemble import RandomForestClassifier

clf_v2 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42, class_weight={0: 1, 1: 3}))
])

recall_cv_v2 = cross_val_score(clf_v2, X, y, cv=5, scoring=disease_recall_scorer)
print("Fold scores:", recall_cv_v2)
print("Mean disease recall:", recall_cv_v2.mean())
print("Std:", recall_cv_v2.std())
# %%
# %% Check precision and accuracy trade-off for this weighting
precision_cv_v2 = cross_val_score(clf_v2, X, y, cv=5, scoring=make_scorer(recall_score, pos_label=0))
accuracy_cv_v2 = cross_val_score(clf_v2, X, y, cv=5, scoring='accuracy')

print("No-disease recall (specificity):", precision_cv_v2.mean())
print("Overall accuracy:", accuracy_cv_v2.mean())
# %%
# %% Train final model on full dataset and save
# Use balanced class weights for the API model. The experimental {0: 1, 1: 3}
# weighting above is useful when prioritizing disease recall, but it pushes
# predict_proba upward and makes the API probability read too aggressive.
clf_final = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

clf_final.fit(X, y)

import joblib
joblib.dump(clf_final, 'model_pipeline.pkl')
print("Model saved.")
# %%
# %% Set up SHAP for the final model
import shap

# Extract the trained RandomForest from inside the pipeline
rf_model = clf_final.named_steps['model']

# Transform X through just the preprocessor, so SHAP sees the same numeric format the model was trained on
X_transformed = clf_final.named_steps['preprocessor'].transform(X)

# Get readable feature names after one-hot encoding
feature_names = clf_final.named_steps['preprocessor'].get_feature_names_out()

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_transformed)
# %%
# %% Check shap_values structure
print(type(shap_values))
if isinstance(shap_values, list):
    print("List of length:", len(shap_values))
    print("Shape of each:", shap_values[0].shape)
else:
    print("Shape:", shap_values.shape)
# %%
# %% Extract SHAP values for the disease class (index 1)
shap_values_disease = shap_values[:, :, 1]
print(shap_values_disease.shape)  # should be (920, 26)
# %%
# %% SHAP global summary plot
import matplotlib.pyplot as plt

shap.summary_plot(shap_values_disease, X_transformed, feature_names=feature_names)
plt.show()
# %%
# %% Explain one specific patient's prediction
patient_index = 0  # let's start with the first patient in the dataset

# Get this patient's SHAP values (one number per feature) and their actual prediction
patient_shap = shap_values_disease[patient_index]
patient_prediction = clf_final.predict_proba(X.iloc[[patient_index]])[0][1]

print("Predicted disease probability:", patient_prediction)
print("Actual outcome (target):", y.iloc[patient_index])
# %%
# %% Show this patient's actual feature values alongside their SHAP impact
import pandas as pd

patient_explanation = pd.DataFrame({
    'feature': feature_names,
    'value': X_transformed[patient_index],
    'shap_impact': patient_shap
}).sort_values('shap_impact', key=abs, ascending=False)

print(patient_explanation.head(10))
# %%
# %% Build a reusable prediction + explanation function
def explain_patient(patient_data, top_n=5):
    """
    Takes a single patient's raw data (as a DataFrame with the right columns),
    returns the predicted probability and the top contributing factors.
    """
    # 1. Get the prediction probability
    probability = clf_final.predict_proba(patient_data)[0][1]

    # 2. Transform the raw input the same way training data was transformed
    patient_transformed = clf_final.named_steps['preprocessor'].transform(patient_data)

    # 3. Run SHAP on this transformed input
    patient_shap_values = explainer.shap_values(patient_transformed)
    patient_shap_disease = patient_shap_values[:, :, 1][0]  # disease class, first (only) patient

    # 4. Build the explanation table, same as before
    explanation = pd.DataFrame({
        'feature': feature_names,
        'value': patient_transformed[0],
        'shap_impact': patient_shap_disease
    }).sort_values('shap_impact', key=abs, ascending=False)

    return {
        'probability': probability,
        'top_factors': explanation.head(top_n)
    }
# %%
# %% Test the function on an existing patient
test_patient = X.iloc[[5]]  # patient at index 5, double brackets to keep it as a DataFrame
result = explain_patient(test_patient)

print("Probability:", result['probability'])
print(result['top_factors'])
# %%
# %% Test with a completely new, made-up patient
new_patient = pd.DataFrame([{
    'age': 58,
    'sex': 'Male',
    'cp': 'asymptomatic',
    'trestbps': 150.0,
    'chol': 280.0,
    'fbs': True,
    'restecg': 'lv hypertrophy',
    'thalch': 120.0,
    'exang': True,
    'oldpeak': 2.0,
    'slope': 'flat',
    'ca': 2.0,
    'thal': 'fixed defect',
    'has_missing': False
}])

result_new = explain_patient(new_patient)
print("Probability:", result_new['probability'])
print(result_new['top_factors'])
# %%
# %% Get exact category values for validation
for col in ['sex', 'cp', 'restecg', 'slope', 'thal']:
    print(col, ":", df[col].dropna().unique())
# %%
