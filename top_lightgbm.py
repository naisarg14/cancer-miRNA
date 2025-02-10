import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, StandardScaler 
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
import lightgbm as lgb
import os, sys
from lightgbm import early_stopping, log_evaluation
from imblearn.over_sampling import SMOTE

if len(sys.argv) != 3:
    print("Usage: python top_lightgbm.py <feature_file> <output-folder>")
    sys.exit(1)
feature_file = sys.argv[1]
folder = sys.argv[2]

rna_names = []
with open(feature_file, 'r') as f:
    for line in f:
        rna_names.append(line.strip().replace("\n", ""))

if not os.path.exists(folder):
    os.makedirs(folder)

smote = SMOTE(random_state=42)

data = pd.read_csv('matrix/training.csv')
data = data[['Samples', 'label'] + [col for col in rna_names if col in data.columns]]
X = data.drop(columns=['Samples', 'label'])
y = data['label']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train, y_train = smote.fit_resample(X_train, y_train)


def objective(trial):
    param = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y_train)),
        'boosting_type': 'gbdt',
        'metric': 'multi_logloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.01),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150, step=10),
        'max_depth': trial.suggest_int('max_depth', -1, 15),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0, step=0.1),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0, step=0.1),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 200, step=10),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 10, step=0.5),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 10, step=0.5),
        'verbosity': -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        param,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'validation'],
        num_boost_round=1000,
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(period=10)
        ]
    )

    val_preds = model.predict(X_val)
    val_pred_labels = np.argmax(val_preds, axis=1)
    accuracy = np.mean(val_pred_labels == y_val)
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("\nBest Hyperparameters:")
print(best_params)
with open(f"{folder}/best_hyperparameters.txt", "w") as f:
    f.write(str(best_params))

final_model = lgb.LGBMClassifier(**best_params, objective='multiclass', num_class=len(np.unique(y_encoded)))
final_model.fit(X_train, y_train)

y_pred_prob = final_model.predict_proba(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

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

feature_importance = pd.DataFrame({
    'Feature': data.drop(columns=['Samples', 'label']).columns,
    'Importance': final_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

feature_importance.to_csv(f"{folder}/feature_importances.csv", index=False)