import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_squared_error, r2_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from base import Base
from data_factory import DataFactory

class Predictor(Base):
    def __init__(self, model_name="DecisionTreeClassifier"):
        super().__init__()
        self.model_dir = self.config["paths"]["model_dir"]
        self.preprocessor_dir = self.config["paths"]["preprocessor_dir"]
        self.reports_dir = self.config["paths"]["reports_dir"]
        self.figures_dir = self.config["paths"]["figures_dir"]
        self.model_name = model_name
        self.target_col = self.config["data"]["target_col"]
        
        # Create directories if they don't exist
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Load model
        self.model = self.load_model()
        
        # Load preprocessor
        self.preprocessor = self.load_preprocessor()
        
        # Determine if this is a classification task
        self.is_classification = hasattr(self.model, 'predict_proba')
    
    def load_model(self):
        """Load the trained model"""
        # Convert model name to filename format
        if self.model_name.lower() == "xgbclassifier":
            model_filename = "xgboost"
        else:
            model_filename = self.model_name.lower().replace("classifier", "").strip()
        
        model_path = os.path.join(self.model_dir, f"{model_filename}.pkl")
        
        if not os.path.exists(model_path):
            self.logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        self.logger.info(f"Loaded model from {model_path}")
        return model
    
    def load_preprocessor(self):
        """Load the feature preprocessor"""
        # Try to load the full preprocessor first
        preprocessor_path = os.path.join(self.preprocessor_dir, "preprocessor.pkl")
        
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            self.logger.info(f"Loaded preprocessor from {preprocessor_path}")
            return preprocessor
        
        # Fall back to just the scaler if full preprocessor not found
        scaler_path = os.path.join(self.preprocessor_dir, "scaler.pkl")
        
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            self.logger.info(f"Loaded scaler from {scaler_path}")
            return scaler
        
        self.logger.warning("No preprocessor or scaler found")
        return None
    
    def preprocess_data(self, data):
        """Apply preprocessing to input data"""
        # Extract features (remove ID columns and target if present)
        id_cols = ['bearing_id', 'window_id', 'condition']
        exclude_cols = [col for col in id_cols if col in data.columns]
        if self.target_col in data.columns:
            exclude_cols.append(self.target_col)
        
        # Store ID columns for later
        id_data = data[exclude_cols].copy() if exclude_cols else None
        
        # Extract features
        X = data.drop(columns=exclude_cols, errors='ignore')
        
        # Apply preprocessing if available
        if self.preprocessor is not None:
            try:
                # If it's a full preprocessor (ColumnTransformer)
                if hasattr(self.preprocessor, 'transformers_'):
                    X_processed = self.preprocessor.transform(X)
                    
                    # Convert to DataFrame with proper column names if possible
                    try:
                        # Get feature names from preprocessor if available
                        if hasattr(self.preprocessor, 'get_feature_names_out'):
                            feature_names = self.preprocessor.get_feature_names_out()
                        else:
                            # Fallback for older scikit-learn versions
                            feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
                        
                        X = pd.DataFrame(X_processed, columns=feature_names)
                    except:
                        # If conversion fails, use as numpy array
                        X = X_processed
                
                # If it's just a scaler
                elif hasattr(self.preprocessor, 'transform'):
                    numerical_cols = self.config["data"]["numerical_cols"]
                    if numerical_cols and numerical_cols[0] != "name-of-numerical-col":
                        X[numerical_cols] = self.preprocessor.transform(X[numerical_cols])
            except Exception as e:
                self.logger.error(f"Error applying preprocessing: {str(e)}")
                # Continue with unprocessed data
        
        return X, id_data
    
    def predict(self, data=None):
        """Make predictions on input data"""
        # If no data provided, load production data
        if data is None:
            data_factory = DataFactory()
            data = data_factory.load_production_data()
            
            if data is None:
                self.logger.error("No data available for prediction")
                return None
        
        # Preprocess data
        X, id_data = self.preprocess_data(data)
        
        # Make predictions
        try:
            y_pred = self.model.predict(X)
            
            # Get prediction probabilities if applicable
            if self.is_classification:
                y_prob = self.model.predict_proba(X)
                # Get probability of positive class for binary classification
                if y_prob.shape[1] == 2:
                    prob_col = y_prob[:, 1]
                else:
                    prob_col = np.max(y_prob, axis=1)
            else:
                prob_col = np.nan
                
            # Create results dataframe
            results = pd.DataFrame({
                'prediction': y_pred,
                'probability': prob_col
            })
            
            # Add original IDs if available
            if id_data is not None:
                for col in id_data.columns:
                    results[col] = id_data[col].reset_index(drop=True)
            
            self.logger.info(f"Generated predictions for {len(results)} samples")
            return results
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def evaluate(self, data=None):
        """Evaluate model on test data with known targets"""
        # Load test data if not provided
        if data is None:
            data_factory = DataFactory()
            _, data = data_factory.load_train_test_data()
            
            if data is None:
                self.logger.error("No test data available for evaluation")
                return None
        
        # Check if target column exists
        if self.target_col not in data.columns:
            self.logger.error(f"Target column '{self.target_col}' not found in data")
            return None
        
        # Get predictions
        predictions = self.predict(data)
        if predictions is None:
            return None
        
        # Add actual values
        predictions['actual'] = data[self.target_col].values
        
        # Calculate metrics
        if self.is_classification:
            metrics = {
                'accuracy': accuracy_score(predictions['actual'], predictions['prediction']),
                'precision': precision_score(predictions['actual'], predictions['prediction'], 
                                           zero_division=0, average='weighted'),
                'recall': recall_score(predictions['actual'], predictions['prediction'], 
                                     zero_division=0, average='weighted'),
                'f1': f1_score(predictions['actual'], predictions['prediction'], 
                              zero_division=0, average='weighted')
            }
            
            # Generate confusion matrix
            cm = confusion_matrix(predictions['actual'], predictions['prediction'])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {self.model_name}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.savefig(os.path.join(self.figures_dir, f"{self.model_name.lower()}_confusion_matrix.png"))
            plt.close()
            
            # Generate classification report
            report = classification_report(predictions['actual'], predictions['prediction'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(os.path.join(self.reports_dir, f"{self.model_name.lower()}_classification_report.csv"))
            
        else:  # Regression
            metrics = {
                'mse': mean_squared_error(predictions['actual'], predictions['prediction']),
                'rmse': np.sqrt(mean_squared_error(predictions['actual'], predictions['prediction'])),
                'r2': r2_score(predictions['actual'], predictions['prediction'])
            }
            
            # Generate scatter plot of actual vs predicted
            plt.figure(figsize=(8, 8))
            plt.scatter(predictions['actual'], predictions['prediction'], alpha=0.5)
            plt.plot([predictions['actual'].min(), predictions['actual'].max()], 
                    [predictions['actual'].min(), predictions['actual'].max()], 'r--')
            plt.title(f'Actual vs Predicted - {self.model_name}')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.savefig(os.path.join(self.figures_dir, f"{self.model_name.lower()}_actual_vs_predicted.png"))
            plt.close()
        
        # Save predictions
        predictions.to_csv(os.path.join(self.reports_dir, f"{self.model_name.lower()}_predictions.csv"), index=False)
        
        # Evaluate by condition if available
        if 'condition' in predictions.columns:
            self.evaluate_by_condition(predictions)
        
        self.logger.info(f"Model evaluation metrics: {metrics}")
        return predictions, metrics
    
    def evaluate_by_condition(self, predictions):
        """Evaluate model performance by operating condition"""
        # Group by condition
        condition_metrics = {}
        
        for condition, group in predictions.groupby('condition'):
            if self.is_classification:
                condition_metrics[f"Condition {condition}"] = {
                    'accuracy': accuracy_score(group['actual'], group['prediction']),
                    'precision': precision_score(group['actual'], group['prediction'], 
                                               zero_division=0, average='weighted'),
                    'recall': recall_score(group['actual'], group['prediction'], 
                                         zero_division=0, average='weighted'),
                    'f1': f1_score(group['actual'], group['prediction'], 
                                  zero_division=0, average='weighted')
                }
            else:
                condition_metrics[f"Condition {condition}"] = {
                    'mse': mean_squared_error(group['actual'], group['prediction']),
                    'rmse': np.sqrt(mean_squared_error(group['actual'], group['prediction'])),
                    'r2': r2_score(group['actual'], group['prediction'])
                }
        
        # Create DataFrame from metrics
        condition_report = pd.DataFrame.from_dict(condition_metrics, orient='index')
        
        # Save as CSV
        report_path = os.path.join(self.reports_dir, f"{self.model_name.lower()}_condition_metrics.csv")
        condition_report.to_csv(report_path)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot metrics by condition
        for i, metric in enumerate(condition_report.columns):
            plt.subplot(2, 2, i+1)
            sns.barplot(x=condition_report.index, y=metric, data=condition_report)
            plt.title(f'{metric} by Condition')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, f"{self.model_name.lower()}_condition_metrics.png"))
        plt.close()
        
        self.logger.info(f"Condition-specific evaluation saved for {self.model_name}")
        
        return condition_metrics
    
    def predict_and_save(self, data=None, output_path=None):
        """Make predictions and save to file"""
        predictions = self.predict(data)
        
        if predictions is None:
            return None
        
        if output_path is None:
            output_path = os.path.join(self.reports_dir, f"{self.model_name.lower()}_predictions.csv")
        
        predictions.to_csv(output_path, index=False)
        self.logger.info(f"Predictions saved to {output_path}")
        
        return predictions