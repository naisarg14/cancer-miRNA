import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import optuna
import os, sys

if len(sys.argv) != 3:
    print("Usage: python cnn.py <feature_file> <output-folder>")
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
y_categorical = to_categorical(y_encoded)

X_train, X_temp, y_train, y_temp = train_test_split(X, y_categorical, test_size=0.4, random_state=42, stratify=y_encoded)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=np.argmax(y_temp, axis=1))

X_train, y_train = smote.fit_resample(X_train, np.argmax(y_train, axis=1))
y_train = to_categorical(y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train_cnn = X_train[..., np.newaxis]
X_val_cnn = X_val[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]


def objective(trial):
    num_filters_1 = trial.suggest_categorical('num_filters_1', [32, 64, 128])
    num_filters_2 = trial.suggest_categorical('num_filters_2', [64, 128, 256])
    kernel_size = trial.suggest_int('kernel_size', 3, 5)
    dense_units = trial.suggest_int('dense_units', 64, 256)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    model = Sequential([
        Conv1D(num_filters_1, kernel_size=kernel_size, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Dropout(dropout_rate),
        Conv1D(num_filters_2, kernel_size=kernel_size, activation='relu'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Dropout(dropout_rate),
        Flatten(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(y_categorical.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train_cnn, y_train,
        validation_data=(X_val_cnn, y_val),
        epochs=50,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stopping]
    )

    _, val_accuracy = model.evaluate(X_val_cnn, y_val, verbose=0)
    return val_accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("\nBest Hyperparameters:\n", best_params)

num_filters_1 = best_params['num_filters_1']
num_filters_2 = best_params['num_filters_2']
kernel_size = best_params['kernel_size']
dense_units = best_params['dense_units']
dropout_rate = best_params['dropout_rate']
batch_size = best_params['batch_size']

model = Sequential([
    Conv1D(num_filters_1, kernel_size=kernel_size, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    BatchNormalization(),
    Dropout(dropout_rate),
    Conv1D(num_filters_2, kernel_size=kernel_size, activation='relu'),
    MaxPooling1D(pool_size=2),
    BatchNormalization(),
    Dropout(dropout_rate),
    Flatten(),
    Dense(dense_units, activation='relu'),
    Dropout(dropout_rate),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train_cnn, y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=50,
    batch_size=batch_size,
    verbose=1,
    callbacks=[early_stopping]
)

test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

y_pred_prob = model.predict(X_test_cnn)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
print("\nClassification Report:\n")
print(report)

with open(f"{folder}/classification_report.txt", "w") as file:
    file.write("Final Classification Report:\n")
    file.write(report)

conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig(f"{folder}/final_confusion_matrix.png")

model.save(f"{folder}/best_cnn_model.keras")