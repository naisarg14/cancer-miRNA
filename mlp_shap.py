import os
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shap

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

fs = 'combined'

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

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

#summary plot
for i in range(shap_values.shape[2]):
    plt.figure()
    class_label = label_encoder.inverse_transform([i])[0]
    shap.summary_plot(shap_values[:, :, i], X_test, feature_names=X.columns.tolist(), show=False)
    plt.title(f'SHAP Summary Plot - {CANCER_CODES[class_label]}')
    plt.savefig(os.path.join(folder, f'shap_summary_plot_{class_label}.png'))
    plt.close()

#heatmap
for class_idx in range(shap_values.shape[2]):
    plt.figure(figsize=(15, 8))
    shap_values_class = shap_values[:, :, class_idx]
    shap_values_class.feature_names = rna_names
    class_label = label_encoder.inverse_transform([class_idx])[0]
    shap.plots.heatmap(shap_values_class)
    plt.title(f'SHAP Heatmap - {CANCER_CODES[class_label]}')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f'shap_heatmap_{class_label}.png'))
    plt.close()