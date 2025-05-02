
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb


def train_model(data, selected_features):
    X = data[selected_features]
    y = data['Response']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=30,
        max_depth=3,
        learning_rate=0.2,
        verbosity=0,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, report, X_test, y_test, selected_features


def predict_customers(data, idx1, idx2, session_state):
    selected_features = session_state.selected_features
    model = session_state.model

    X = data[selected_features]

    customer1 = X.iloc[[idx1]]
    customer2 = X.iloc[[idx2]]

    pred1 = model.predict_proba(customer1)[0][1]  # Probability of class 1 (purchase)
    pred2 = model.predict_proba(customer2)[0][1]

    return pred1, pred2
