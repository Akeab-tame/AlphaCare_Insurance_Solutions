# Enhanced scripts/credit_risk_eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

sns.set(style="darkgrid", palette="muted")

class CreditRiskEDA:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the EDA class with the DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataset to be analyzed.
        """
        self.df = df
        self.numeric_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        self.classify_columns()

    def classify_columns(self):
        """Automatically classify columns into numeric, categorical, and datetime types."""
        self.numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = self.df.select_dtypes(include=['datetime']).columns.tolist()

    def data_overview(self):
        """Display an overview of the dataset including shape, types, and missing values."""
        print(f"Dataset contains {self.df.shape[0]} rows and {self.df.shape[1]} columns.\n")
        print("Data Types:")
        print(self.df.dtypes)
        print("\nSample Data:")
        print(self.df.head())
        print("\nMissing Values:")
        print(self.df.isnull().sum())

    def summary_statistics(self):
        """Return detailed summary statistics for numeric columns."""
        numeric_df = self.df[self.numeric_cols]
        summary_stats = numeric_df.describe().T
        summary_stats['median'] = numeric_df.median()
        summary_stats['mode'] = numeric_df.mode().iloc[0]
        summary_stats['skewness'] = numeric_df.skew()
        summary_stats['kurtosis'] = numeric_df.kurtosis()
        print("Summary Statistics:\n")
        return summary_stats
    def plot_numerical_distribution(self, cols=None):
    
    # Default to all numeric columns if no specific columns are provided
        if cols is None:
            cols = self.df.select_dtypes(include='number').columns.tolist()

        if not cols:
            print("No numeric columns to plot.")
            return

        num_plots = len(cols)
        n_rows = math.ceil(num_plots**0.5)
        n_cols = math.ceil(num_plots / n_rows)
    
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            sns.histplot(self.df[col], bins=15, kde=True, color='skyblue', edgecolor='black', ax=axes[i])
            axes[i].set_title(f'Distribution of {col}', fontsize=14)
            axes[i].set_xlabel(col, fontsize=12)
            axes[i].set_ylabel('Frequency', fontsize=12)
            axes[i].axvline(self.df[col].mean(), color='red', linestyle='dashed', linewidth=1)
            axes[i].axvline(self.df[col].median(), color='green', linestyle='dashed', linewidth=1)
            axes[i].legend({'Mean': self.df[col].mean(), 'Median': self.df[col].median()})

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    # Function to plot skewness for each numerical feature
    def plot_skewness(self, cols=None):
        """
        Plot skewness of numerical features in the dataset.
        """
        numeric_df = self.df.select_dtypes(include='number')

        if numeric_df.empty:
            print("No numeric columns available for skewness analysis.")
            return

        skewness = numeric_df.skew().sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=skewness.index, y=skewness.values, palette='coolwarm')
        plt.title("Skewness of Numerical Features", fontsize=16)
        plt.xlabel("Features", fontsize=12)
        plt.ylabel("Skewness", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_numerical_distribution(self, cols=None):
        """
        Plot distribution of numerical columns.
        
        Parameters:
        -----------
        cols : list, optional
            List of columns to plot. If None, plots all numeric columns.
        """
        cols = cols or self.numeric_cols
        num_plots = len(cols)
        n_rows = math.ceil(num_plots**0.5)
        n_cols = math.ceil(num_plots / n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            sns.histplot(self.df[col], bins=20, kde=True, ax=axes[i], color='steelblue')
            axes[i].set_title(f"Distribution of {col}")
            axes[i].axvline(self.df[col].mean(), color='red', linestyle='--', label='Mean')
            axes[i].axvline(self.df[col].median(), color='green', linestyle='--', label='Median')
            axes[i].legend()

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def plot_categorical_distribution(self, cols=None, max_categories=10):
        """
        Plot distribution of categorical features with limited unique values.

        Parameters:
        -----------
        cols : list, optional
            List of columns to plot. If None, plots selected categorical columns.
        max_categories : int
            Maximum number of unique values to include in plots.
        """
        cols = cols or [col for col in self.categorical_cols if self.df[col].nunique() <= max_categories]

        num_plots = len(cols)
        n_rows = math.ceil(num_plots**0.5)
        n_cols = math.ceil(num_plots / n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            sns.countplot(data=self.df, x=col, ax=axes[i], palette="viridis")
            axes[i].set_title(f"Distribution of {col}")
            axes[i].tick_params(axis='x', rotation=45)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self, method='pearson'):
        """
        Plot correlation matrix of numeric columns.

        Parameters:
        -----------
        method : str, optional
            Correlation method, default is 'pearson'.
        """
        if not self.numeric_cols:
            print("No numeric columns to calculate correlations.")
            return

        corr_matrix = self.df[self.numeric_cols].corr(method=method)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt='.2f', linewidths=0.5)
        plt.title(f"{method.capitalize()} Correlation Matrix")
        plt.show()

    def check_missing_values(self, visualize=True):
        """
        Display and optionally visualize missing values.

        Parameters:
        -----------
        visualize : bool, optional
            If True, display a heatmap of missing values.
        """
        missing = self.df.isnull().sum()
        print("Missing Values:\n")
        print(missing[missing > 0])

        if visualize:
            plt.figure(figsize=(12, 6))
            sns.heatmap(self.df.isnull(), cbar=False, cmap='viridis')
            plt.title("Missing Value Heatmap")
            plt.show()

    def impute_missing_values(self, strategy='mean'):
        """
        Impute missing values in numeric columns.

        Parameters:
        -----------
        strategy : str
            Imputation strategy ('mean', 'median', or 'mode').
        """
        for col in self.numeric_cols:
            if self.df[col].isnull().sum() > 0:
                if strategy == 'mean':
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median':
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode().iloc[0], inplace=True)
        print(f"Imputed missing values using {strategy}.")