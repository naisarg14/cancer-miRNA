import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

data = pd.read_csv('matrix/training.csv')
X = data.drop(columns=['Samples', 'label'])
y = data['label']
print(f"Features shape before selection: {X.shape}")

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

svm = SVC(kernel='linear', random_state=42)
selector = RFE(estimator=svm, n_features_to_select=100, step=100, verbose=1)
X_train_selected = selector.fit_transform(X_train, y_train)

selected_features = X.columns[selector.support_]
print(f"Top 100 selected features: {list(selected_features)}")
with open('svm_if.txt', 'w') as f:
    for item in selected_features:
        f.write(f"{item}\n")