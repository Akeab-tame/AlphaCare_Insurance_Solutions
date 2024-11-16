# scripts/data_preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt


class DataPreprocessor:
    """
    A utility class for executing data preprocessing tasks on transactional data.

    Methods
    -------
    compute_aggregates(data: pd.DataFrame) -> pd.DataFrame
        Derives aggregate metrics like sum, mean, count, and standard deviation for transaction data.

    add_transactional_metrics(data: pd.DataFrame) -> pd.DataFrame
        Computes features based on transaction behavior such as net amount and debit/credit ratios.

    generate_time_features(data: pd.DataFrame) -> pd.DataFrame
        Extracts detailed temporal features like hour, day, month, and year.

    transform_categorical(data: pd.DataFrame, categorical_columns: list) -> pd.DataFrame
        Converts categorical data into numerical format through encoding.

    fill_missing_data(data: pd.DataFrame, method: str = 'mean') -> pd.DataFrame
        Addresses missing values using specified strategies.

    scale_numerical_features(data: pd.DataFrame, numeric_columns: list, mode: str = 'standard') -> pd.DataFrame
        Scales numeric features using standardization or normalization.
    """

    @staticmethod
    def compute_aggregates(data: pd.DataFrame) -> pd.DataFrame:
        # Create aggregate metrics grouped by customer ID
        aggregate_data = data.groupby('CustomerId').agg(
            Total_Transactions=('Amount', 'sum'),
            Average_Transaction=('Amount', 'mean'),
            Transaction_Volume=('TransactionId', 'count'),
            StdDev_Transaction=('Amount', 'std')
        ).reset_index()

        # Merge aggregate data into the original dataset
        data = data.merge(aggregate_data, on='CustomerId', how='left')
        return data

    @staticmethod
    def add_transactional_metrics(data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates derived metrics based on transaction flows.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to be augmented.

        Returns
        -------
        pd.DataFrame
            DataFrame enriched with transaction-level metrics.
        """
        metrics = data.groupby('CustomerId').agg(
            Net_Amount=('Amount', 'sum'),
            Total_Debits=('Amount', lambda values: (values > 0).sum()),
            Total_Credits=('Amount', lambda values: (values < 0).sum())
        ).reset_index()

        # Compute debit-to-credit ratios and handle zero-credit scenarios
        metrics['Debit_to_Credit_Ratio'] = metrics['Total_Debits'] / (metrics['Total_Credits'] + 1)

        # Merge metrics back into the dataset
        data = pd.merge(data, metrics, on='CustomerId', how='left')
        return data

    @staticmethod
    def generate_time_features(data: pd.DataFrame) -> pd.DataFrame:
        # Parse timestamps into datetime format
        data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])

        # Extract various time-based attributes
        data['Hour_of_Transaction'] = data['TransactionStartTime'].dt.hour
        data['Day_of_Transaction'] = data['TransactionStartTime'].dt.day
        data['Month_of_Transaction'] = data['TransactionStartTime'].dt.month
        data['Year_of_Transaction'] = data['TransactionStartTime'].dt.year

        return data

    @staticmethod
    def transform_categorical(data: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
        """
        Converts categorical fields into numerical values using encoding techniques.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset with categorical fields.
        categorical_columns : list
            List of categorical fields to transform.

        Returns
        -------
        pd.DataFrame
            Dataset with encoded categorical variables.
        """
        encoder = OneHotEncoder(drop='first', sparse_output=False)

        for column in categorical_columns:
            # Apply one-hot encoding and capture encoded categories
            encoded_array = encoder.fit_transform(data[[column]].astype(str))
            categories = encoder.get_feature_names_out([column])
            
            # Create DataFrame from encoded categories
            encoded_df = pd.DataFrame(encoded_array, columns=categories)

            # Merge back with the original dataset
            data = pd.concat([data, encoded_df], axis=1)
            data.drop(columns=[column], inplace=True)

        return data

    @staticmethod
    def fill_missing_data(data: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
        if method in ['mean', 'median', 'most_frequent']:
            imputer = SimpleImputer(strategy=method)
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
        elif method == 'remove':
            data.dropna(inplace=True)
        return data

    @staticmethod
    def scale_numerical_features(data: pd.DataFrame, numeric_columns: list, mode: str = 'standard') -> pd.DataFrame:
        scaler = StandardScaler() if mode == 'standard' else MinMaxScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        return data