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
y_pred_prob = model.predict(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

auc_scores = roc_auc_score(y_test, y_pred_prob, average=None)
auc_df = pd.DataFrame({
    'Class': label_encoder.classes_,
    'AUC Score': auc_scores
})

y_pred = y_pred_prob.argmax(axis=1)
y_test_classes = y_test.argmax(axis=1)

conf_matrix = confusion_matrix(y_test_classes, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_)
conf_matrix_df.to_csv(os.path.join(folder, 'confusion_matrix.csv'))

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig(f"{folder}/confusion_matrix.png")

report = classification_report(y_test_classes, y_pred, target_names=label_encoder.classes_, output_dict=True)

specificity_scores = {}
false_positives = {}
for i, class_name in enumerate(label_encoder.classes_):
    tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
    fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
    false_positives[class_name] = fp
    specificity = tn / (tn + fp)
    specificity_scores[class_name] = specificity

report_df = pd.DataFrame(report).transpose()
report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'])
report_df = report_df.drop(['support'], axis=1)
specificity_df = pd.DataFrame.from_dict(specificity_scores, orient='index', columns=['Specificity'])
false_positives_df = pd.DataFrame.from_dict(false_positives, orient='index', columns=['False Positives'])

combined_df = report_df.merge(specificity_df, left_index=True, right_index=True, how='left')
combined_df = combined_df.merge(auc_df.set_index('Class'), left_index=True, right_index=True, how='left')
combined_df = combined_df.merge(false_positives_df, left_index=True, right_index=True, how='left')

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

combined_df.index = combined_df.index.map(lambda x: CANCER_CODES[x])
combined_df.to_csv(os.path.join(folder, 'classification_report.csv'))

print(combined_df)