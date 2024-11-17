import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pytz


class RFMAnalysis:
    """
    A utility for analyzing customer transaction data using Recency, Frequency, and Monetary metrics.
    Includes scoring, labeling, and visualization features.
    """

    def __init__(self, transaction_data):
        """
        Initialize the RFMAnalysis class with transaction data.

        Parameters:
        -----------
        transaction_data : pd.DataFrame
            A DataFrame containing customer transaction records.
        """
        self.data = transaction_data

    def compute_rfm_metrics(self):
        """
        Generate Recency, Frequency, and Monetary metrics for each customer.

        Returns:
            pd.DataFrame: A summary DataFrame containing RFM metrics.
        """
        self.data['TransactionStartTime'] = pd.to_datetime(self.data['TransactionStartTime'])
        current_date = pd.Timestamp.now(tz=pytz.UTC)

        # Calculate metrics
        self.data['LastPurchaseDate'] = self.data.groupby('CustomerId')['TransactionStartTime'].transform('max')
        self.data['Recency'] = (current_date - self.data['LastPurchaseDate']).dt.days
        self.data['Frequency'] = self.data.groupby('CustomerId')['TransactionId'].transform('count')
        self.data['Monetary'] = self.data.groupby('CustomerId')['Amount'].transform('sum')

        # Create an RFM summary table
        rfm_summary = self.data[['CustomerId', 'Recency', 'Frequency', 'Monetary']].drop_duplicates()
        return rfm_summary
    
    

    def assign_rfm_scores(self, rfm_summary):
        """
        Assign scores to Recency, Frequency, and Monetary values using quantiles.

        Parameters:
        -----------
        rfm_summary : pd.DataFrame
            DataFrame with computed RFM metrics.

        Returns:
            pd.DataFrame: DataFrame with additional score columns and a composite RFM score.
        """
        rfm_summary['Recency_Score'] = pd.qcut(rfm_summary['Recency'], 4, labels=[4, 3, 2, 1])
        rfm_summary['Frequency_Score'] = pd.qcut(rfm_summary['Frequency'], 4, labels=[1, 2, 3, 4])
        rfm_summary['Monetary_Score'] = pd.qcut(rfm_summary['Monetary'], 4, labels=[1, 2, 3, 4])

        rfm_summary['RFM_Score'] = (
            rfm_summary['Recency_Score'].astype(int) * 0.2 +
            rfm_summary['Frequency_Score'].astype(int) * 0.4 +
            rfm_summary['Monetary_Score'].astype(int) * 0.4
        )
        return rfm_summary

    def label_customers(self, rfm_summary):
        """
        Classify customers into categories based on their RFM score.

        Parameters:
        -----------
        rfm_summary : pd.DataFrame
            DataFrame with RFM scores.

        Returns:
            pd.DataFrame: Updated DataFrame with a category label for each customer.
        """
        threshold = rfm_summary['RFM_Score'].median()
        rfm_summary['CustomerCategory'] = rfm_summary['RFM_Score'].apply(
            lambda score: 'High Value' if score >= threshold else 'Low Value'
        )
        return rfm_summary

    def calculate_woe_and_iv(self, grouped_data):
        """
        Compute Weight of Evidence (WoE) and Information Value (IV) for customer segments.

        Parameters:
        -----------
        grouped_data : pd.DataFrame
            Grouped data containing counts of high-value and low-value customers.

        Returns:
            tuple: WoE and IV values for each segment.
        """
        total_high = grouped_data['HighValueCount'].sum()
        total_low = grouped_data['LowValueCount'].sum()

        epsilon = 1e-10
        high_rate = grouped_data['HighValueCount'] / (total_high + epsilon)
        low_rate = grouped_data['LowValueCount'] / (total_low + epsilon)

        woe = np.log((high_rate + epsilon) / (low_rate + epsilon))
        iv = np.sum((high_rate - low_rate) * woe)
        return woe, iv

    def visualize_distributions(self):
        """
        Visualize the distribution of Recency, Frequency, and Monetary values.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics = ['Recency', 'Frequency', 'Monetary']
        titles = ['Recency Distribution', 'Frequency Distribution', 'Monetary Distribution']

        for ax, metric, title in zip(axes, metrics, titles):
            self.data[metric].hist(bins=20, ax=ax)
            ax.set_title(title)

        plt.tight_layout()
        plt.show()

    def plot_relationships(self):
        """
        Create scatter plots to explore relationships among RFM metrics.
        """
        sns.pairplot(self.data[['Recency', 'Frequency', 'Monetary']])
        plt.suptitle('Relationships Between RFM Metrics', y=1.02)
        plt.show()

    def plot_correlation_matrix(self):
        """
        Display a heatmap of the correlation matrix for RFM metrics.
        """
        correlation_matrix = self.data[['Recency', 'Frequency', 'Monetary']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap of RFM Metrics')
        plt.show()
