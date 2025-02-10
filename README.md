## Feature Selection Scripts
- FS_lasso.py – Selects top 100 features using Recursive Feature Elimination (RFE) with a Lasso estimator.
- FS_LGBM.py – Selects top 100 features using RFE with a LightGBM estimator.
- FS_LR.py – Selects top 100 features using RFE with a linear regression estimator.
- FS_RF.py – Selects top 100 features using RFE with a random forest estimator.
- FS_SVM.py – Selects top 100 features using RFE with a support vector machine (SVM) estimator.

## Model Training & Hyperparameter Tuning
- top_knn.py – Uses Optuna to hyperparameter tune and train a K-Nearest Neighbors (KNN) model.
- top_lightgbm.py – Hyperparameter tunes and trains a LightGBM model using Optuna.
- top_mlp.py – Hyperparameter tunes and trains a Multi-Layer Perceptron (MLP) neural network.
- top_rf.py – Hyperparameter tunes and trains a Random Forest model using Optuna.
- top_svm.py – Hyperparameter tunes and trains a Support Vector Machine (SVM) model using Optuna.
- top_cnn.py – Hyperparameter tunes and trains a Convolutional Neural Network (CNN) model.

## MLP Analysis & Evaluation
- mlp_cv.py – Performs 5-fold cross-validation for an MLP model.
- mlp_analysis.py – Analyzes MLP model performance using various general metrics.
- mlp_shap.py – Computes and visualizes SHAP summary and heatmap plots for MLP feature importance.
- mlp_auc.py – Computes and visualizes the AUC (Area Under Curve) for MLP model predictions.