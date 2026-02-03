"""
######################################################################
## This repository holds the codes for the research paper:          ##
##                                                                  ##
## Paper Title:                                                     ##
##   Serum-MiR-CanPred: Deep Learning Framework for Pan-Cancer      ##
##   Classification and miRNA-Targeted Drug Discovery               ##
##                                                                  ##
## Author:                                                          ##
##   Naisarg Patel                                                  ##
##                                                                  ##
## Journal: RNA Biology                                             ##
## Year: 2025                                                       ##
## DOI: https://doi.org/10.1080/15476286.2025.2577433               ##
##                                                                  ##
## License: GNU General Public License v3.0 (GPL-3.0)               ##
## Contact: naisargbpatel14<at>gmail<dot>com                        ##
######################################################################      
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE

data = pd.read_csv('matrix/training.csv')
X = data.drop(columns=['Samples', 'label'])
y = data['label']
print(f"Features shape before selection: {X.shape}")

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

lgbm = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=0)
rfe = RFE(estimator=lgbm, n_features_to_select=100, step=100, verbose=1)
X_train = rfe.fit_transform(X_train, y_train)


selected_features = X.columns[rfe.support_]
print(f"Top 100 selected features: {list(selected_features)}")
with open('lgbm_if.txt', 'w') as f:
    for item in selected_features:
        f.write(f"{item}\n")