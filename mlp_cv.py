import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import os
import tensorflow as tf
tf.random.set_seed(42)

fs = 'combined'
in_folder = f"mlp_{fs}_if"
with open(f"{in_folder}/best_hyperparameters.txt", "r") as f:
    best_params = eval(f.read())

folder = f"mlp-cv_{fs}_if"
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
X_scaled = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
smote = SMOTE(random_state=42)

all_reports = []
all_conf_matrices = np.zeros((len(label_encoder.classes_), len(label_encoder.classes_)))
fold = 1

for train_idx, val_idx in kf.split(X_train, np.argmax(y_train, axis=1)):
    print(f"\nTraining fold {fold}...")
    
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    X_fold_train_resampled, y_fold_train_resampled = smote.fit_resample(X_fold_train, np.argmax(y_fold_train, axis=1))
    y_fold_train_resampled = to_categorical(y_fold_train_resampled)

    model = Sequential()
    model.add(Input(shape=(X_fold_train_resampled.shape[1],)))
    for _ in range(best_params['num_layers']):
        model.add(Dense(best_params['units'], activation='relu'))
        model.add(Dropout(best_params['dropout_rate']))
    model.add(Dense(y_fold_train_resampled.shape[1], activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1)
    
    history = model.fit(X_fold_train_resampled, y_fold_train_resampled,
                        validation_data=(X_fold_val, y_fold_val),
                        batch_size=best_params['batch_size'], epochs=100,
                        verbose=1, callbacks=[early_stopping])
    
    y_val_pred = np.argmax(model.predict(X_fold_val), axis=1)
    y_val_true = np.argmax(y_fold_val, axis=1)
    
    print(f"\nClassification Report for Fold {fold}:")
    report = classification_report(y_val_true, y_val_pred, target_names=label_encoder.classes_)
    print(report)
    all_reports.append(report)
    
    conf_matrix = confusion_matrix(y_val_true, y_val_pred)
    all_conf_matrices += conf_matrix
    
    fold += 1

with open(f"{folder}/classification_report_cv.txt", "w") as file:
    for i, report in enumerate(all_reports):
        file.write(f"Fold {i+1}:\n")
        file.write(report + "\n")

print("\nTraining final model on full training set...")
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, np.argmax(y_train, axis=1))
y_train_resampled = to_categorical(y_train_resampled)

final_model = Sequential()
final_model.add(Input(shape=(X_train_resampled.shape[1],)))
for _ in range(best_params['num_layers']):
    final_model.add(Dense(best_params['units'], activation='relu'))
    final_model.add(Dropout(best_params['dropout_rate']))
final_model.add(Dense(y_train_resampled.shape[1], activation='softmax'))

final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']),
                    loss='categorical_crossentropy', metrics=['accuracy'])

final_model.fit(X_train_resampled, y_train_resampled,
                batch_size=best_params['batch_size'], epochs=100,
                verbose=1, callbacks=[early_stopping])

final_model.save(f"{folder}/final_model.keras")

y_test_pred = np.argmax(final_model.predict(X_test), axis=1)
y_test_true = np.argmax(y_test, axis=1)

test_report = classification_report(y_test_true, y_test_pred, target_names=label_encoder.classes_)
print("\nFinal Test Set Classification Report:\n")
print(test_report)
conf_matrix_test = confusion_matrix(y_test_true, y_test_pred)

with open(f"{folder}/classification_report_test.txt", "w") as file:
    file.write(test_report)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Reds', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Final Test Set Confusion Matrix')
plt.savefig(f"{folder}/confusion_matrix_test.png")