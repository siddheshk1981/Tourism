import pandas as pd
import sklearn
from imblearn.pipeline import Pipeline,make_pipeline
from imblearn.over_sampling import SMOTE
from feature_engine.outliers import Winsorizer
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder,OrdinalEncoder
from sklearn.compose import make_column_transformer
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score,precision_score,make_scorer
# for model serialization
import joblib
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import time
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()


Xtrain_path = "hf://datasets/siddhesh1981/tourism-package-predict/Xtrain.csv"
Xtest_path = "hf://datasets/siddhesh1981/tourism-package-predict/Xtest.csv"
ytrain_path = "hf://datasets/siddhesh1981/tourism-package-predict/ytrain.csv"
ytest_path = "hf://datasets/siddhesh1981/tourism-package-predict/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

preprocessor=make_column_transformer((Winsorizer(capping_method='quantiles',fold=0.05),['MonthlyIncome']),
                                     (OneHotEncoder(handle_unknown='ignore',drop='first'),['TypeofContact','Gender','MaritalStatus']),
                                     (OrdinalEncoder(categories=[['Free Lancer','Salaried','Small Business','Large Business']],handle_unknown='use_encoded_value',unknown_value=-1),['Occupation']),
                                     (OrdinalEncoder(categories=[['Basic','Standard','King','Deluxe','Super Deluxe']],handle_unknown='use_encoded_value',unknown_value=-1),['ProductPitched']),
                                     (OrdinalEncoder(categories=[['AVP','VP','Manager','Senior Manager','Executive']],handle_unknown='use_encoded_value',unknown_value=-1),['Designation']),remainder='passthrough'
                                        )
smote=SMOTE(sampling_strategy=0.75,k_neighbors=5,random_state=1)

gb_model=GradientBoostingClassifier(random_state=42)

pipeline=make_pipeline(preprocessor,smote,gb_model)

param_grid={
    'gradientboostingclassifier__n_estimators':[100,300,500],
    'gradientboostingclassifier__max_depth':[2,7,5],
    'gradientboostingclassifier__learning_rate':[0.0001,0.001,0.01,0.1],
    'gradientboostingclassifier__subsample':[0.5,0.7,0.9],
    'gradientboostingclassifier__max_features':[0.6,0.8,0.9],
    'gradientboostingclassifier__min_samples_split':[4,12,20],
    'gradientboostingclassifier__min_samples_leaf':[2,6,10],
    'gradientboostingclassifier__tol':[0.0001,0.001,0.01]
}

with mlflow.start_run():
    # Hyperparameter tuning
    precision_scorer = make_scorer(precision_score)
    random_search = RandomizedSearchCV(pipeline, param_grid,scoring=precision_scorer,cv=5, n_jobs=-1)
    random_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = random_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)
            time.sleep(2)

    # Log best parameters separately in main run
    mlflow.log_params(random_search.best_params_)

    # Store and evaluate the best model
    best_model = random_search.best_estimator_
    best_model.fit(Xtrain,ytrain)
    y_pred_train = best_model.predict_proba(Xtrain)[:,1]>0.60
    y_pred_test = best_model.predict_proba(Xtest)[:,1]>0.60

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    print(train_report)
    print(test_report)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

     # Save the model locally
    model_path = "gb_tourism_package_predict_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "siddhesh1981/tourism-package-predict-model"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="gb_tourism_package_predict_model_v1.joblib",
        path_in_repo="gb_tourism_package_predict_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
