import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.utils import resample
import joblib
from base import Base
from feature_engineering import FeatureEngineering
import logging
from sklearn.ensemble import RandomForestClassifier

class DataFactory(Base):
    def __init__(self):
        super().__init__()
        self.raw_data_dir = self.config["paths"]["raw_data_dir"]
        self.processed_data_dir = self.config["paths"]["processed_data_dir"]
        self.preprocessor_dir = self.config["paths"]["preprocessor_dir"]
        self.train_data_path = self.config["paths"]["train_data_path"]
        self.test_data_path = self.config["paths"]["test_data_path"]
        self.prod_data_path = self.config["paths"]["prod_data_path"]
        
        self.split_size = self.config["data"]["split_size"]
        self.seed = self.config["data"]["seed"]
        self.target_col = self.config["data"]["target_col"]
        self.categorical_cols = self.config["data"]["categorical_cols"]
        self.numerical_cols = self.config["data"]["numerical_cols"]
        
        # Create directories
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.preprocessor_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.test_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.prod_data_path), exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def extract_features(self):
        """Extract features from raw bearing data"""
        self.logger.info("Starting feature extraction")
        feature_engineering = FeatureEngineering()
        features_df = feature_engineering.process_all_bearings()
        self.logger.info(f"Feature extraction complete. Shape: {features_df.shape if features_df is not None else 'None'}")
        return features_df
    
    def validate_data(self, df):
        """Validate data quality and report issues"""
        if df is None or df.empty:
            self.logger.error("Empty or None DataFrame provided for validation")
            return None
            
        validation_results = {
            "missing_values": df.isnull().sum().sum(),
            "duplicate_rows": df.duplicated().sum(),
            "feature_count": len(df.columns),
            "row_count": len(df)
        }
        
        # Check for infinite values
        num_inf = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        validation_results["infinite_values"] = num_inf
        
        # Log validation results
        self.logger.info(f"Data validation results: {validation_results}")
        
        # Handle missing values
        if validation_results["missing_values"] > 0:
            self.logger.warning(f"Found {validation_results['missing_values']} missing values. Filling with appropriate values.")
            # Fill numeric columns with median
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].fillna(df[col].median())
            
            # Fill categorical columns with mode
            for col in df.select_dtypes(include=['object', 'category']).columns:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Handle infinite values
        if num_inf > 0:
            self.logger.warning(f"Found {num_inf} infinite values. Replacing with large values.")
            df = df.replace([np.inf, -np.inf], [1e9, -1e9])
        
        return df
    
    def balance_classes(self, X, y):
        """Balance classes for classification problems"""
        if self.target_col != "failure" or len(np.unique(y)) <= 1:
            return X, y
            
        # Check class distribution
        class_counts = pd.Series(y).value_counts()
        self.logger.info(f"Class distribution before balancing: {class_counts.to_dict()}")
        
        # If imbalanced, oversample minority class
        if class_counts.min() / class_counts.max() < 0.5:  # Arbitrary threshold
            self.logger.info("Balancing classes by oversampling minority class")
            
            # Combine features and target for resampling
            data = pd.concat([X, pd.Series(y, name=self.target_col)], axis=1)
            
            # Separate by class
            majority_class = class_counts.idxmax()
            minority_class = class_counts.idxmin()
            
            majority_data = data[data[self.target_col] == majority_class]
            minority_data = data[data[self.target_col] == minority_class]
            
            # Oversample minority class
            minority_upsampled = resample(
                minority_data,
                replace=True,
                n_samples=len(majority_data),
                random_state=self.seed
            )
            
            # Combine majority and upsampled minority
            balanced_data = pd.concat([majority_data, minority_upsampled])
            
            # Separate features and target
            X_balanced = balanced_data.drop(columns=[self.target_col])
            y_balanced = balanced_data[self.target_col]
            
            # Log new class distribution
            new_class_counts = pd.Series(y_balanced).value_counts()
            self.logger.info(f"Class distribution after balancing: {new_class_counts.to_dict()}")
            
            return X_balanced, y_balanced
        
        return X, y
    
    def select_features(self, X_train, y_train, X_test, k=None):
        """Select most important features"""
        if k is None:
            # Default to 50% of features or at least 10
            k = max(10, X_train.shape[1] // 2)
        
        self.logger.info(f"Selecting top {k} features")
        
        # Determine if classification or regression
        is_classification = len(np.unique(y_train)) <= 10
        
        # Select appropriate scoring function
        if is_classification:
            selector = SelectKBest(score_func=f_classif, k=k)
        else:
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        
        # Fit and transform
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature names
        selected_features = X_train.columns[selector.get_support()].tolist()
        self.logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        
        return X_train_selected, X_test_selected, selected_features
    
    def load_processed_features(self):
        """Load processed features from the processed_features.csv file"""
        self.logger.info("Loading processed features from file")
        processed_features_path = os.path.join(self.processed_data_dir, "processed_features.csv")
        
        if not os.path.exists(processed_features_path):
            self.logger.error(f"Processed features file not found at {processed_features_path}")
            return None
            
        try:
            features_df = pd.read_csv(processed_features_path)
            self.logger.info(f"Loaded processed features with shape: {features_df.shape}")
            self.logger.info(f"Available columns: {features_df.columns.tolist()}")
            return features_df
        except Exception as e:
            self.logger.error(f"Error loading processed features: {str(e)}")
            return None

    def prepare_data(self, features_df=None):
        """Prepare data for training and testing"""
        self.logger.info("Starting data preparation")
        
        # Load processed features if not provided
        if features_df is None:
            features_df = self.load_processed_features()
            if features_df is None:
                self.logger.error("Failed to load processed features")
                return None, None
        
        # Validate data
        features_df = self.validate_data(features_df)
        if features_df is None:
            self.logger.error("Data validation failed")
            return None, None
        
        # Determine if this is a classification or regression task
        is_classification = len(np.unique(features_df[self.target_col])) <= 10
        
        # Identify feature columns
        id_cols = ['bearing_id', 'window_id']
        self.id_cols = [col for col in id_cols if col in features_df.columns]
        
        # Identify feature columns (including RUL columns)
        feature_cols = [col for col in features_df.columns 
                        if col != self.target_col and col not in self.id_cols]
        
        # Update numerical columns if not specified
        if not self.numerical_cols:
            self.numerical_cols = [col for col in feature_cols 
                                  if col not in self.categorical_cols]
            
        # Ensure RUL columns are treated as numerical features
        rul_cols = [col for col in feature_cols if 'rul' in col.lower()]
        for rul_col in rul_cols:
            if rul_col not in self.numerical_cols:
                self.numerical_cols.append(rul_col)
        
        # Split data into train and test
        if is_classification:
            # Use stratified split for classification
            train_df, test_df = train_test_split(
                features_df, 
                test_size=self.split_size, 
                random_state=self.seed,
                stratify=features_df[self.target_col]
            )
        else:
            # Regular split for regression
            train_df, test_df = train_test_split(
                features_df, 
                test_size=self.split_size, 
                random_state=self.seed
            )
        
        self.logger.info(f"Train set shape: {train_df.shape}, Test set shape: {test_df.shape}")
        
        # Prepare features and target
        X_train = train_df[feature_cols]
        y_train = train_df[self.target_col]
        
        X_test = test_df[feature_cols]
        y_test = test_df[self.target_col]
        
        # Balance classes if needed
        if is_classification:
            X_train, y_train = self.balance_classes(X_train, y_train)
        
        # Select important features
        X_train, X_test, selected_features = self.select_features(X_train, y_train, X_test)
        
        # Create preprocessor
        preprocessor = self._create_preprocessor(selected_features)
        
        # Save preprocessor
        preprocessor_path = os.path.join(self.preprocessor_dir, "preprocessor.joblib")
        joblib.dump(preprocessor, preprocessor_path)
        self.logger.info(f"Preprocessor saved to {preprocessor_path}")
        
        # Save train and test data
        train_df.to_csv(self.train_data_path, index=False)
        test_df.to_csv(self.test_data_path, index=False)
        self.logger.info(f"Train data saved to {self.train_data_path}")
        self.logger.info(f"Test data saved to {self.test_data_path}")
        
        return train_df, test_df
    
    def _create_preprocessor(self, feature_cols):
        """Create preprocessing pipeline"""
        # Identify categorical and numerical columns from selected features
        cat_cols = [col for col in self.categorical_cols if col in feature_cols]
        num_cols = [col for col in self.numerical_cols if col in feature_cols]
        
        # Create transformers
        transformers = []
        
        if num_cols:
            num_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', num_transformer, num_cols))
        
        if cat_cols:
            cat_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            transformers.append(('cat', cat_transformer, cat_cols))
        
        # Create preprocessor
        preprocessor = ColumnTransformer(transformers=transformers)
        
        return preprocessor
    
    def load_train_test_data(self):
        """Load train and test data"""
        self.logger.info("Loading train and test data")
        
        # Check if files exist
        if not os.path.exists(self.train_data_path) or not os.path.exists(self.test_data_path):
            self.logger.warning("Train or test data files not found. Running data preparation.")
            return self.prepare_data()
        
        # Load data
        try:
            train_df = pd.read_csv(self.train_data_path)
            test_df = pd.read_csv(self.test_data_path)
            self.logger.info(f"Loaded train data: {train_df.shape}, test data: {test_df.shape}")
            return train_df, test_df
        except Exception as e:
            self.logger.error(f"Error loading train/test data: {str(e)}")
            return None, None
    
    def load_production_data(self):
        """Load production data"""
        self.logger.info("Loading production data")
        
        # Check if file exists
        if not os.path.exists(self.prod_data_path):
            self.logger.warning("Production data file not found")
            return None
        
        # Load data
        try:
            prod_df = pd.read_csv(self.prod_data_path)
            self.logger.info(f"Loaded production data: {prod_df.shape}")
            return prod_df
        except Exception as e:
            self.logger.error(f"Error loading production data: {str(e)}")
            return None

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select important features using feature importance."""
        try:
            # Separate features and target
            X = df.drop(columns=[self.target_col] + self.id_cols)
            y = df[self.target_col]
            
            # Create and fit RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Get feature importance
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            })
            
            # Sort by importance
            importance = importance.sort_values('importance', ascending=False)
            
            # Always keep RUL columns
            rul_cols = [col for col in X.columns if 'rul' in col.lower()]
            
            # Select top features plus RUL columns
            top_features = importance.head(self.n_features_to_keep)['feature'].tolist()
            selected_features = list(set(top_features + rul_cols))
            
            # Keep ID columns and selected features
            keep_cols = self.id_cols + [self.target_col] + selected_features
            return df[keep_cols]
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {str(e)}")
            raise