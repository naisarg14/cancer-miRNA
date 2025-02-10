import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE


data = pd.read_csv('matrix/training.csv')

X = data.drop(columns=['Samples', 'label'])
y = data['label']
print(f"Features shape before selection: {X.shape}")

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
selector = RFE(model, n_features_to_select=100, step=100, verbose=1)
selector.fit(X_train, y_train)

selected_features = X.columns[selector.get_support()]
print(f"Top {len(selected_features)} selected features: {list(selected_features)}")

with open('lasso_if_2.txt', 'w') as f:
    for item in selected_features:
        f.write(f"{item}\n")