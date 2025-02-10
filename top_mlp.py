import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import optuna
import os, sys

if len(sys.argv) != 3:
    print("Usage: python mlp_tensor.py <feature_file> <output-folder>")
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
y_categorical = to_categorical(y_encoded)

X_train, X_temp, y_train, y_temp = train_test_split(X, y_categorical, test_size=0.4, random_state=42, stratify=y_categorical)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, np.argmax(y_train, axis=1))
y_train_resampled = to_categorical(y_train_resampled)

def objective(trial):
    num_layers = trial.suggest_int('num_layers', 2, 6)
    units = trial.suggest_int('units', 16, 512, step=16)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 512, step=16)

    model = Sequential()
    model.add(Input(shape=(X_train_resampled.shape[1],)))
    for _ in range(num_layers):
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(y_train_resampled.shape[1], activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)

    history = model.fit(
        X_train_resampled, y_train_resampled,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=50,
        verbose=1,
        callbacks=[early_stopping]
    )

    val_accuracy = max(history.history['val_accuracy'])
    return val_accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("\nBest Hyperparameters:")
print(best_params)
with open(f"{folder}/best_hyperparameters.txt", "w") as f:
    f.write(str(best_params))


final_model = Sequential()
final_model.add(Input(shape=(X_train_resampled.shape[1],)))
for _ in range(best_params['num_layers']):
    final_model.add(Dense(best_params['units'], activation='relu'))
    final_model.add(Dropout(best_params['dropout_rate']))
final_model.add(Dense(y_train_resampled.shape[1], activation='softmax'))

final_model.compile(
    optimizer=Adam(learning_rate=best_params['learning_rate']),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1)

history = final_model.fit(
    X_train_resampled, y_train_resampled,
    validation_data=(X_val, y_val),
    batch_size=best_params['batch_size'],
    epochs=100,
    verbose=1,
    callbacks=[early_stopping]
)

y_pred_prob = final_model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)


print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
with open(f"{folder}/classification_report.txt", "w") as file:
    file.write("Classification Report:\n")
    file.write(classification_report(y_true, y_pred, target_names=label_encoder.classes_))


conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig(f"{folder}/confusion_matrix.png")