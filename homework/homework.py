
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#

import os
import json
import gzip
import shutil
import pickle
import joblib
import zipfile
import numpy as np
import pandas as pd
from glob import glob

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    make_scorer, balanced_accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix
)


def read_zip_data(type_of_data):
    zip_path = f"files/input/{type_of_data}_data.csv.zip"
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        file_names = zip_file.namelist()
        with zip_file.open(file_names[0]) as file:
            file_df = pd.read_csv(file)
    return file_df

def clean_data(df):
    cleaned_df = df.copy()

    cleaned_df = cleaned_df.rename(columns = {"default payment next month": "default"})
    cleaned_df = cleaned_df.drop(columns = "ID")
    cleaned_df = cleaned_df.loc[cleaned_df["MARRIAGE"] != 0]
    cleaned_df = cleaned_df.loc[cleaned_df["EDUCATION"] != 0]
    cleaned_df["EDUCATION"] = cleaned_df["EDUCATION"].apply(lambda x: x if x < 4 else 4)
    
    return cleaned_df


def build_pipeline(cat_features, numerical_columns) :
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'), cat_features), 
            ("scaler", StandardScaler(with_mean=True, with_std=True), numerical_columns),
        ],
        remainder='passthrough'  
    )
    return Pipeline([
        ('preprocessor', preprocessor),       
        ('pca', PCA()),              
        ('feature_selection', SelectKBest(score_func=f_classif)),       
        ('classifier', SVC(kernel="rbf", random_state=12345, max_iter=-1))                       
    ])

def optimize_pipeline(pipeline, X_train, y_train) :
    param_grid = {
        "pca__n_components": [20, X_train.shape[1] - 2],  
        'feature_selection__k': [12],           
        'classifier__kernel': ["rbf"],           
        'classifier__gamma': [0.1],                  
    }   
    cv = StratifiedKFold(n_splits=10)
    scorer = make_scorer(balanced_accuracy_score)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=3 
    )
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    return grid_search, grid_search.best_estimator_


def create_output_directory(output_directory):
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

def save_model(path, model):
    create_output_directory("files/models/")

    with gzip.open(path, "wb") as f:
        joblib.dump(model, f)

    print(f"Model saved successfully at {path}")


def evaluate_model(model, X, y, dataset_name):

    y_pred = model.predict(X)

    metrics = {
        "type" : "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y, y_pred, average="weighted"),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "recall": recall_score(y, y_pred, average="weighted"),
        "f1_score": f1_score(y, y_pred, average="weighted"),
    }
    
    return metrics

def compute_confusion_matrix(model, X, y, dataset_name):
    """
    Computes the confusion matrix and returns it as a dictionary.
    """
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    cm_dict = {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {
            "predicted_0": int(cm[0, 0]), 
            "predicted_1": int(cm[0, 1])
        },
        "true_1": {
            "predicted_0": int(cm[1, 0]), 
            "predicted_1": int(cm[1, 1])
        },
    }

    return cm_dict


def run_job():

    train_data = read_zip_data("train")
    test_data = read_zip_data("test")
    train_data_clean = clean_data(train_data)
    test_data_clean = clean_data(test_data)


    X_train = train_data_clean.drop("default", axis = 1)
    X_test = test_data_clean.drop("default", axis = 1)

    y_train = train_data_clean["default"]
    y_test = test_data_clean["default"] 






    categorical_features = ["SEX","EDUCATION", "MARRIAGE"]

    numerical_features = [x for x in X_train.columns if x not in categorical_features]

    pipeline = build_pipeline(categorical_features, numerical_features)

    grid_search, best_model = optimize_pipeline(pipeline, X_train, y_train)






    os.makedirs("files/models/", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", 'wb') as f:
        pickle.dump(grid_search, f)
    
    train_cm = compute_confusion_matrix(best_model, X_train, y_train, "train")
    test_cm = compute_confusion_matrix(best_model, X_test, y_test, "test")

    # Add "type" field to confusion matrices
    train_cm["type"] = "cm_matrix"
    test_cm["type"] = "cm_matrix"

    metrics = []

    train_metrics = evaluate_model(best_model, X_train, y_train, "train")
    test_metrics = evaluate_model(best_model, X_test, y_test, "test")
    
    metrics.append(train_metrics)
    metrics.append(test_metrics)
    # Append new confusion matrices
    metrics.append(train_cm)
    metrics.append(test_cm)

    os.makedirs("files/output/", exist_ok=True)
    with open("files/output/metrics.json", "w") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + "\n")
    print("metricas y modelo guardado :)" )


if __name__ == "__main__":
    run_job()
