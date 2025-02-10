import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler


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

lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, verbose=1)
rfe = RFE(estimator=lr, n_features_to_select=100, step=100, verbose=1)
X_train = rfe.fit_transform(X_train, y_train)

selected_features = X.columns[rfe.support_]
print(f"Top 100 selected features: {list(selected_features)}")
with open('lr_if.txt', 'w') as f:
    for item in selected_features:
        f.write(f"{item}\n")
