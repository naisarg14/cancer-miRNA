from sklearn.metrics import roc_curve, auc
import os
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

fs = 'combined'

CANCER_CODES = {
    'OV': 'Ovarian Cancer',
    'GA': 'Gastric Cancer',
    'PR': 'Prostate Cancer',
    'NC': 'No Cancer',
    'BT': 'Biliary Tract Cancer',
    'CR': 'Colorectal Cancer',
    'PA': 'Pancreatic Cancer',
    'LU': 'Lung Cancer',
    'BR': 'Breast Cancer',
    'ES': 'Esophageal Squamous Cell Cancer',
    'BL': 'Bladder Cancer',
    'HC': 'Hepatocellular Cancer',
    'SA': 'Sarcoma',
}

model = load_model('mlp-cv_combined_if/final_model.keras')

folder = f"mlp-analysis_{fs}_if"
os.makedirs(folder, exist_ok=True)

rna_names = []
with open(f'{fs}_if.txt', 'r') as f:
    for line in f:
        rna_names.append(line.strip())

data = pd.read_csv('matrix/training.csv')
data = data[['Samples', 'label'] + [col for col in rna_names if col in data.columns]]
X = data.drop(columns=['Samples', 'label'])
y = data['label']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, stratify=y_encoded, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_pred_prob = model.predict(X_test)
num_classes = y_categorical.shape[1]

plt.figure(figsize=(12, 8))

for class_idx in range(num_classes):
    fpr, tpr, _ = roc_curve(y_test[:, class_idx], y_pred_prob[:, class_idx])
    roc_auc = auc(fpr, tpr)
    class_label = label_encoder.inverse_transform([class_idx])[0]
    plt.plot(fpr, tpr, lw=2, label=f'{CANCER_CODES[class_label]} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Guessing')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Classes')
plt.legend(loc='lower right')

save_path = os.path.join(folder, "roc_curves_all_classes.png")
plt.savefig(save_path, dpi=300)
plt.close()

"""
#This plots all AUC graphs in seperate files->

from sklearn.metrics import roc_curve, auc
import os
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

fs = 'combined'

CANCER_CODES = {
    'OV': 'Ovarian Cancer',
    'GA': 'Gastric Cancer',
    'PR': 'Prostate Cancer',
    'NC': 'No Cancer',
    'BT': 'Biliary Tract Cancer',
    'CR': 'Colorectal Cancer',
    'PA': 'Pancreatic Cancer',
    'LU': 'Lung Cancer',
    'BR': 'Breast Cancer',
    'ES': 'Esophageal Squamous Cell Cancer',
    'BL': 'Bladder Cancer',
    'HC': 'Hepatocellular Cancer',
    'SA': 'Sarcoma',
}

model = load_model('mlp-cv_combined_if/final_model.keras')

folder = f"mlp-analysis_{fs}_if"
os.makedirs(folder, exist_ok=True)

rna_names = []
with open(f'{fs}_if.txt', 'r') as f:
    for line in f:
        rna_names.append(line.strip())

data = pd.read_csv('matrix/training.csv')
data = data[['Samples', 'label'] + [col for col in rna_names if col in data.columns]]
X = data.drop(columns=['Samples', 'label'])
y = data['label']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split into train-test before cross-validation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, stratify=y_encoded, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_pred_prob = model.predict(X_test)


num_classes = y_categorical.shape[1]
for class_idx in range(num_classes):
    fpr, tpr, _ = roc_curve(y_test[:, class_idx], y_pred_prob[:, class_idx])
    roc_auc = auc(fpr, tpr)

    class_label = label_encoder.inverse_transform([class_idx])[0]

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {CANCER_CODES[class_label]}')
    plt.legend(loc='lower right')

    save_path = os.path.join(folder, f"roc_curve_{class_label}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

"""