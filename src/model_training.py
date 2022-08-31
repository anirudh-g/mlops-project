# %%
import pandas as pd
import numpy as np
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import compose
from sklearn import metrics
from sklearn import model_selection
from sklearn import ensemble
import xgboost as xgb
import optuna

# %%
df = pd.read_csv('/home/ubuntu/mlops-project/data/heart_failure_clinical_records_dataset.csv')
y = df['DEATH_EVENT']
X = df.drop(['DEATH_EVENT'], axis=1)

X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, random_state=42, test_size=0.2)

# %%
numeric_transformer = pipeline.Pipeline(steps=[
        ('outliers', preprocessing.RobustScaler(quantile_range=(5,95))),
        ('scale', preprocessing.MinMaxScaler())
    ])

categorical_tranformer = pipeline.Pipeline(steps=[
        ('encode', preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=99999))
    ])

preprocess_pipeline = compose.ColumnTransformer(
        transformers=
        [
            ('num', numeric_transformer, compose.make_column_selector(dtype_exclude='object')), 
            ('cat', categorical_tranformer, compose.make_column_selector(dtype_include='object')), 
        ],

        remainder='passthrough')


clf = pipeline.Pipeline(steps=[
        ('preprocessor', preprocess_pipeline),      
        ('model', ensemble.RandomForestClassifier(random_state=42))
        ]
    )

# %%
def objective(trial, clf, X_train, y_train):
    params = {
        #'model__eta': trial.suggest_float('eta', 0.2, 0.8),
        'model__max_depth': trial.suggest_int('max_depth', 2, 10)
    }
    clf.train(X_train, y_train)
    return metrics.accuracy_score()

# %%
sampler = optuna.samplers.TPESampler(seed=10)
study = optuna.create_study(sampler = sampler, direction='maximize')

func = lambda trial: objective(trial, clf, X_train, y_train)
study.optimize(func, n_trials=20)

# %%



