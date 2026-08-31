import pandas as pd
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = (
    MODEL_DIR /
    "placement_model.joblib"
)


def prepare_ml_data(df):
    data = df.copy()

    y = data["placement_status"].map(
        {
            "Not Placed": 0,
            "Placed": 1,
        }
    )

    X = data.drop(
        columns=[
            "student_id",
            "placement_status",
            "salary_package_lpa",
        ]
    )

    return X, y


def get_column_groups(X):

    categorical_columns = (
        X.select_dtypes(
            include=["object", "string"]
        )
        .columns
        .tolist()
    )

    numeric_columns = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    return (
        categorical_columns,
        numeric_columns,
    )


def build_preprocessor(
    categorical_columns,
    numeric_columns,
):

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_columns,
            ),
            (
                "numeric",
                StandardScaler(),
                numeric_columns,
            ),
        ]
    )


def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    categorical_columns,
    numeric_columns,
):

    preprocessor = build_preprocessor(
        categorical_columns,
        numeric_columns,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(
        X_test
    )

    probabilities = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
        ),
        "recall": recall_score(
            y_test,
            predictions,
        ),
        "f1": f1_score(
            y_test,
            predictions,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    return pipeline, metrics


def compare_models(df):

    X, y = prepare_ml_data(df)

    (
        categorical_columns,
        numeric_columns,
    ) = get_column_groups(X)

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    logistic_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    random_forest_model = (
        RandomForestClassifier(
            n_estimators=80,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
    )

    gradient_boosting_model = (
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
    )

    voting_ensemble = VotingClassifier(
        estimators=[
            (
                "logistic",
                logistic_model,
            ),
            (
                "random_forest",
                random_forest_model,
            ),
            (
                "gradient_boosting",
                gradient_boosting_model,
            ),
        ],
        voting="soft",
    )

    models = {
        "Logistic Regression":
            logistic_model,

        "Random Forest":
            random_forest_model,

        "Gradient Boosting":
            gradient_boosting_model,

        "Soft Voting Ensemble":
            voting_ensemble,
    }

    results = {}

    best_auc = -1
    best_model_name = None
    best_pipeline = None

    for model_name, model in models.items():

        pipeline, metrics = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            categorical_columns,
            numeric_columns,
        )

        results[model_name] = metrics

        if metrics["roc_auc"] > best_auc:

            best_auc = metrics["roc_auc"]

            best_model_name = (
                model_name
            )

            best_pipeline = pipeline

    return {
        "results": results,
        "best_model_name":
            best_model_name,
        "best_pipeline":
            best_pipeline,
    }


def save_model(model):

    joblib.dump(
        model,
        MODEL_PATH,
    )


def load_model():

    if not MODEL_PATH.exists():
        return None

    return joblib.load(
        MODEL_PATH
    )


def get_or_train_model(df):

    saved_model = load_model()

    if saved_model is not None:

        return {
            "pipeline": saved_model,
            "source": "saved",
        }

    ml_result = compare_models(df)

    best_model = ml_result[
        "best_pipeline"
    ]

    save_model(
        best_model
    )

    return {
        "pipeline": best_model,
        "source": "trained",
        "comparison": ml_result,
    }


def predict_student_probability(
    pipeline,
    student,
):

    excluded_columns = [
        "student_id",
        "placement_status",
        "salary_package_lpa",
    ]

    feature_columns = [
        column
        for column in student.index
        if column not in excluded_columns
    ]

    student_features = pd.DataFrame(
        [
            student[
                feature_columns
            ].to_dict()
        ]
    )

    probability = (
        pipeline.predict_proba(
            student_features
        )[0][1]
    )

    prediction = pipeline.predict(
        student_features
    )[0]

    return {
        "prediction": (
            "Placed"
            if prediction == 1
            else "Not Placed"
        ),
        "placement_probability":
            probability * 100,
    }