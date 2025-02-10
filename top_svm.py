import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
import os, sys
from imblearn.over_sampling import SMOTE

if len(sys.argv) != 3:
    print("Usage: python svm_top.py <feature_file> <output-folder>")
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

X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

def objective(trial):
    C = trial.suggest_float('C', 0.1, 100.0, log=True)
    kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid', 'linear'])
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

    if kernel == 'poly':
        degree = trial.suggest_int('degree', 2, 5)
    else:
        degree = 3

    model = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, probability=True)
    model.fit(X_train_scaled, y_train)

    val_preds = model.predict(X_val_scaled)
    accuracy = np.mean(val_preds == y_val)
    return accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, n_jobs=3)

best_params = study.best_params
print("\nBest Hyperparameters:")
print(best_params)
with open(f"{folder}/best_hyperparameters.txt", "w") as f:
    f.write(str(best_params))

final_model = SVC(
    C=best_params['C'], 
    kernel=best_params['kernel'], 
    gamma=best_params['gamma'], 
    degree=best_params.get('degree', 3),
    probability=True
)
final_model.fit(X_train_scaled, y_train)

y_pred = final_model.predict(X_test_scaled)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
with open(f"{folder}/classification_report.txt", "w") as file:
    file.write("Classification Report:\n")
    file.write(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig(f"{folder}/confusion_matrix.png")