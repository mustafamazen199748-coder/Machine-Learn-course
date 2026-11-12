import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

DATA_PATH = "data/diabetes_dataset.csv"
OUTPUT_FILE = "DiaScreen AI.bin"


def train():
    print(f"Loading dataset from {DATA_PATH}...")
    data = pd.read_csv(DATA_PATH)

    data.drop(
        columns=["ethnicity", "education_level", "income_level", "employment_status"],
        inplace=True,
    )

    categorical_col = list(data.dtypes[data.dtypes == "object"].index)
    for col in categorical_col:
        data[col] = data[col].str.lower().str.replace(" ", "_")

    train_val_data, test_data = train_test_split(
        data, test_size=0.2, random_state=1, stratify=data["diagnosed_diabetes"]
    )
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=0.25,
        random_state=1,
        stratify=train_val_data["diagnosed_diabetes"],
    )

    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    y_train = train_data["diagnosed_diabetes"].values
    y_val = val_data["diagnosed_diabetes"].values
    y_test = test_data["diagnosed_diabetes"].values

    cols_to_drop = ["diagnosed_diabetes", "diabetes_stage"]
    train_data.drop(columns=cols_to_drop, inplace=True)
    val_data.drop(columns=cols_to_drop, inplace=True)
    test_data.drop(columns=cols_to_drop, inplace=True)

    categorical_col = categorical_col + [
        "cardiovascular_history",
        "hypertension_history",
        "family_history_diabetes",
    ]
    if "diabetes_stage" in categorical_col:
        categorical_col.remove("diabetes_stage")

    numeric_col = list(data.dtypes[data.dtypes != "object"].index)
    to_remove = [
        "cardiovascular_history",
        "hypertension_history",
        "family_history_diabetes",
        "diagnosed_diabetes",
    ]
    numeric_col = [col for col in numeric_col if col not in to_remove]

    dv = DictVectorizer(sparse=False)

    train_dict = train_data[categorical_col + numeric_col].to_dict(orient="records")
    val_dict = val_data[categorical_col + numeric_col].to_dict(orient="records")
    test_dict = test_data[categorical_col + numeric_col].to_dict(orient="records")

    X_train = dv.fit_transform(train_dict)
    X_val = dv.transform(val_dict)
    X_test = dv.transform(test_dict)

    X_train_val_final = np.vstack((X_train, X_val))
    y_train_val_final = np.concatenate((y_train, y_val))

    dtrain_val = xgb.DMatrix(X_train_val_final, label=y_train_val_final)
    dtest = xgb.DMatrix(X_test, label=y_test)

    xgb_params = {
        "eta": 0.1,
        "max_depth": 3,
        "min_child_weight": 10,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "nthread": 8,
        "seed": 1,
        "verbosity": 0,
    }

    print("Training final XGBoost model...")
    model = xgb.train(xgb_params, dtrain_val, num_boost_round=40)

    y_pred_xgb_test = model.predict(dtest)
    test_auc_xgb = roc_auc_score(y_test, y_pred_xgb_test)

    t_selected = 0.2
    y_pred_test_xgb = (y_pred_xgb_test > t_selected).astype(int)

    prec_xgb = precision_score(y_test, y_pred_test_xgb, zero_division=0)
    rec_xgb = recall_score(y_test, y_pred_test_xgb)
    f1_xgb = f1_score(y_test, y_pred_test_xgb)
    f2_xgb = fbeta_score(y_test, y_pred_test_xgb, beta=2)

    print("\n--- FINAL XGBOOST TEST RESULTS ---")
    print(f"Test AUC:   {test_auc_xgb:.4f}")
    print(f"Threshold:  {t_selected}")
    print(f"Precision:  {prec_xgb:.4f}")
    print(f"Recall:     {rec_xgb:.4f}")
    print(f"F1-Score:   {f1_xgb:.4f}")
    print(f"F2-Score:   {f2_xgb:.4f}\n")

    with open(OUTPUT_FILE, "wb") as f_out:
        pickle.dump((dv, model), f_out)

    print(f"Model saved successfully to '{OUTPUT_FILE}'")


if __name__ == "__main__":
    train()