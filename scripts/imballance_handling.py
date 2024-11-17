from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np

class ImbalanceHandler:
    """
    A class to address class imbalance using techniques like SMOTE, re-sampling, and cost-sensitive modeling.
    """

    def __init__(self, features, target):
        """
        Initialize the ImbalanceHandler with features and target.

        Parameters:
        -----------
        features : pd.DataFrame
            The feature set for the model.
        target : pd.Series
            The target variable with class labels.
        """
        self.features = features
        self.target = target

    def apply_smote(self, random_state=42):
        """
        Apply SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset.

        Parameters:
        -----------
        random_state : int, optional
            Seed for reproducibility. Default is 42.

        Returns:
        --------
        X_resampled : np.ndarray
            Resampled feature set.
        y_resampled : np.ndarray
            Resampled target set.
        """
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(self.features, self.target)
        return X_resampled, y_resampled

    def train_with_resampling(self, oversample=True):
        """
        Train a model using over-sampling or under-sampling.

        Parameters:
        -----------
        oversample : bool, optional
            Whether to apply over-sampling. Default is True. If False, under-sampling is applied.

        Returns:
        --------
        model : RandomForestClassifier
            Trained model.
        """
        if oversample:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(self.features, self.target)
        else:
            # Under-sampling: Down-sample the majority class
            class_counts = self.target.value_counts()
            min_class = class_counts.idxmin()
            max_class = class_counts.idxmax()
            min_size = class_counts[min_class]

            majority_class_indices = self.target[self.target == max_class].index
            minority_class_indices = self.target[self.target == min_class].index

            undersampled_majority_indices = np.random.choice(majority_class_indices, min_size, replace=False)
            balanced_indices = np.concatenate([undersampled_majority_indices, minority_class_indices])
            X_resampled = self.features.loc[balanced_indices]
            y_resampled = self.target.loc[balanced_indices]

        model = RandomForestClassifier(random_state=42)
        model.fit(X_resampled, y_resampled)
        return model

    def train_cost_sensitive_model(self):
        """
        Train a cost-sensitive model using weighted loss functions.

        Returns:
        --------
        model : RandomForestClassifier
            Trained cost-sensitive model.
        """
        # Compute sample weights
        sample_weights = compute_sample_weight(class_weight='balanced', y=self.target)

        # Train a cost-sensitive model
        model = RandomForestClassifier(random_state=42)
        model.fit(self.features, self.target, sample_weight=sample_weights)
        return model