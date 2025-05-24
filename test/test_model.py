"""Test module for loan prediction model."""

import os
from typing import List

import numpy as np
import pandas as pd
import pytest
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample data for testing with the actual column structure."""
    return pd.DataFrame(
        {
            "Loan_ID": ["LP001002", "LP001003", "LP001005"],
            "Gender": ["Male", "Male", "Male"],
            "Married": ["No", "Yes", "Yes"],
            "Dependents": ["0", "1", "0"],
            "Education": ["Graduate", "Graduate", "Graduate"],
            "Self_Employed": ["No", "No", "Yes"],
            "ApplicantIncome": [5849, 4583, 3000],
            "CoapplicantIncome": [0, 1508, 0],
            "LoanAmount": [np.nan, 128, 66],
            "Loan_Amount_Term": [360, 360, 360],
            "Credit_History": [1, 1, 1],
            "Property_Area": ["Urban", "Rural", "Urban"],
            "Loan_Status": ["Y", "N", "Y"],
        }
    )


def test_data_loading(
    sample_data: pd.DataFrame, tmp_path: pytest.TempPathFactory
) -> None:
    """Test if data can be loaded correctly."""
    csv_path = tmp_path / "test_loan_data.csv"
    sample_data.to_csv(csv_path, index=False)

    loaded_data = pd.read_csv(csv_path)

    assert loaded_data.shape[0] == 3
    assert len(loaded_data.columns) == 13
    assert all(
        col in loaded_data.columns
        for col in [
            "Loan_ID",
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
            "Property_Area",
            "Loan_Status",
        ]
    )


def test_data_preprocessing(sample_data: pd.DataFrame) -> None:
    """Test data preprocessing steps."""
    # Drop Loan_ID
    data = sample_data.drop(["Loan_ID"], axis=1)

    # Split features and target
    X = data.drop(["Loan_Status"], axis=1)
    y = data["Loan_Status"]

    assert "Loan_ID" not in data.columns
    assert X.shape[1] == 11  # Number of features after dropping Loan_ID and Loan_Status
    assert len(y) == 3  # Number of target values
    assert all(status in ["Y", "N"] for status in y)


def test_train_test_split(sample_data: pd.DataFrame) -> None:
    """Test train-test split functionality."""
    # Prepare data
    data = sample_data.drop(["Loan_ID"], axis=1)
    X = data.drop(["Loan_Status"], axis=1)
    y = data["Loan_Status"]

    # Perform split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)


def test_handle_missing_values(sample_data: pd.DataFrame) -> None:
    """Test handling of missing values."""
    data = sample_data.drop(["Loan_ID"], axis=1)

    # Check for missing values
    assert data.isnull().any().any()  # Verify that there are missing values

    # Fill missing values (you might want to implement your own strategy)
    data_filled = data.fillna(data.mean(numeric_only=True))

    assert not data_filled.isnull().any().any()  # Verify no missing values remain


def test_model_training(sample_data: pd.DataFrame) -> None:
    """Test model training with actual data structure."""
    # Prepare data
    data = sample_data.drop(["Loan_ID"], axis=1)
    data_filled = data.fillna(data.mean(numeric_only=True))

    # Convert categorical variables to numeric
    categorical_columns: List[str] = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]

    data_encoded = pd.get_dummies(data_filled, columns=categorical_columns)

    X = data_encoded.drop(["Loan_Status"], axis=1)
    y = data_filled["Loan_Status"]

    # Create and train model
    model = LogisticRegression()
    model.fit(X, y)

    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")


def test_model_saving(
    sample_data: pd.DataFrame, tmp_path: pytest.TempPathFactory
) -> None:
    """Test model saving functionality with processed data."""
    # Prepare data
    data = sample_data.drop(["Loan_ID"], axis=1)
    data_filled = data.fillna(data.mean(numeric_only=True))

    # Convert categorical variables
    categorical_columns: List[str] = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]

    data_encoded = pd.get_dummies(data_filled, columns=categorical_columns)

    X = data_encoded.drop(["Loan_Status"], axis=1)
    y = data_filled["Loan_Status"]

    # Train model
    model = LogisticRegression()
    model.fit(X, y)

    # Save model
    model_path = tmp_path / "test_model.pkl"
    dump(model, model_path)

    assert os.path.exists(model_path)
    loaded_model = load(model_path)
    assert isinstance(loaded_model, LogisticRegression)


def test_model_predictions(sample_data: pd.DataFrame) -> None:
    """Test model predictions with processed data."""
    # Prepare data
    data = sample_data.drop(["Loan_ID"], axis=1)
    data_filled = data.fillna(data.mean(numeric_only=True))

    # Convert categorical variables
    categorical_columns: List[str] = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]

    data_encoded = pd.get_dummies(data_filled, columns=categorical_columns)

    X = data_encoded.drop(["Loan_Status"], axis=1)
    y = data_filled["Loan_Status"]

    # Train model
    model = LogisticRegression()
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert all(pred in ["Y", "N"] for pred in predictions)
