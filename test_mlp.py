import sys
from sklearn.metrics import roc_curve, auc
import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import classification_report
import math

model = load_model('mlp-cv_combined_if/final_model.keras')

rna_names = []
with open(f'combined_if.txt', 'r') as f:
    for line in f:
        rna_names.append(line.strip())

id = sys.argv[1] 

data = pd.read_csv(f'{id}/{id}_final_matrix.csv')
data = data[['Samples', 'label'] + [col for col in rna_names if col in data.columns]]

data = data.dropna(subset=['label'])

X = data.drop(columns=['Samples', 'label'])
y = data['label']

# Load the saved label encoder
label_encoder = joblib.load('label_encoder.pkl')
y_encoded = label_encoder.transform(y)
y_categorical = to_categorical(y_encoded)

# Load the saved standard scaler and scale the features
scaler = joblib.load('standard_scaler.pkl')
X_scaled = scaler.transform(X)

y_pred_prob = model.predict(X_scaled)
num_classes = y_categorical.shape[1]

# Get actual predictions from the model
y_pred_classes = y_pred_prob.argmax(axis=1)
y_pred_labels = label_encoder.inverse_transform(y_pred_classes)

# Get unique labels present in the test set
unique_labels = sorted(y.unique())

# classification report with only the labels present in test set
print("\nClassification Report:")
print(classification_report(y, y_pred_labels, labels=unique_labels, target_names=unique_labels))


# Calculate ROC AUC for each class
roc_auc = {}
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_categorical[:, i], y_pred_prob[:, i])
    roc_auc[label_encoder.classes_[i]] = auc(fpr, tpr)

print("ROC AUC scores:")
for class_name, score in roc_auc.items():
    print(f"{class_name}: {score:.4f}")

valid_aucs = [v for v in roc_auc.values() if not math.isnan(v)]
average_auc = sum(valid_aucs) / len(valid_aucs) if valid_aucs else float('nan')
print(f"Average AUC: {average_auc:.4f}")