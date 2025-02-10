import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import optuna
from imblearn.over_sampling import SMOTE
import sys

if len(sys.argv) != 3:
    print("Usage: python random_forest.py <feature_file> <output-folder>")
    sys.exit(1)

feature_file = sys.argv[1]
folder = sys.argv[2]
if not os.path.exists(folder):
    os.makedirs(folder)

rna_names = []
with open(feature_file, 'r') as f:
    for line in f:
        rna_names.append(line.strip().replace("\n", ""))

data = pd.read_csv('matrix/training.csv')
data = data[['Samples', 'label'] + [col for col in rna_names if col in data.columns]]
X = data.drop(columns=['Samples', 'label'])
y = data['label']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

def objective(trial):
    max_depth = trial.suggest_int('max_depth', 5, 30, step=5)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

    model = RandomForestClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)

    y_eval_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_eval_pred)

    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("Best Parameters:\n", best_params)

with open(f"{folder}/best_params_optuna.txt", "w") as file:
    file.write("Best Parameters:\n")
    file.write(str(best_params))

best_model = RandomForestClassifier(
    **best_params,
    random_state=42,
    n_jobs=-1
)
best_model.fit(X_train, y_train)

joblib.dump(best_model, f"{folder}/best_random_forest_model_optuna.pkl")

class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
with open(f"{folder}/class_mapping.txt", "w") as file:
    file.write("Class Mapping (Original Labels to Encoded Classes):\n")
    for key, value in class_mapping.items():
        file.write(f"{key}: {value}\n")

y_eval_pred = best_model.predict(X_val)
print("Evaluation Set Classification Report:\n", classification_report(y_val, y_eval_pred))
with open(f"{folder}/evaluation_report.txt", "w") as file:
    file.write("Evaluation Set Classification Report:\n")
    file.write(classification_report(y_val, y_eval_pred))

y_test_pred = best_model.predict(X_test)
print("Test Set Classification Report:\n", classification_report(y_test, y_test_pred))
with open(f"{folder}/classification_report.txt", "w") as file:
    file.write("Test Set Classification Report:\n")
    file.write(classification_report(y_test, y_test_pred))

conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:\n", conf_matrix)
with open(f"{folder}/confusion_matrix_optuna.txt", "w") as file:
    file.write("Confusion Matrix:\n")
    file.write(str(conf_matrix))

feature_importances = best_model.feature_importances_
important_features = X.columns[np.argsort(feature_importances)[::-1]]
feature_importance_df = pd.DataFrame({
    'Feature': important_features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:\n", feature_importance_df)
feature_importance_df.to_csv(f"{folder}/feature_importances_optuna.csv", index=False)