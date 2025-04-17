import pandas as pd
import numpy as np
import os
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_squared_error, r2_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from base import Base
from data_factory import DataFactory

class ModelTrainer(Base):
    def __init__(self):
        super().__init__()
        self.model_dir = self.config["paths"]["model_dir"]
        self.figures_dir = self.config["paths"]["figures_dir"]
        self.reports_dir = self.config["paths"]["reports_dir"]
        self.target_col = self.config["data"]["target_col"]
        
        # Create directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Load data
        data_factory = DataFactory()
        self.train_df, self.test_df = data_factory.load_train_test_data()
        
        # Prepare features and target
        if self.train_df is not None and self.test_df is not None:
            id_cols = ['bearing_id', 'window_id']
            self.id_cols = [col for col in id_cols if col in self.train_df.columns]
            
            feature_cols = [col for col in self.train_df.columns 
                            if col != self.target_col and col not in self.id_cols]
            
            self.X_train = self.train_df[feature_cols]
            self.y_train = self.train_df[self.target_col]
            
            self.X_test = self.test_df[feature_cols]
            self.y_test = self.test_df[self.target_col]
            
            # Determine if this is a classification or regression task
            self.is_classification = len(np.unique(self.y_train)) <= 10  # Arbitrary threshold
            self.logger.info(f"Task type: {'Classification' if self.is_classification else 'Regression'}")
            
            # Disable MLflow autologging
            mlflow.autolog(disable=True)
    
    def perform_cross_validation(self, model, X, y, cv=5):
        """Perform cross-validation and return scores"""
        if self.is_classification:
            # Use stratified k-fold for classification
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.config["data"]["seed"])
            scoring = 'f1'
        else:
            # Use regular k-fold for regression
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=self.config["data"]["seed"])
            scoring = 'neg_mean_squared_error'
        
        cv_scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring)
        
        if self.is_classification:
            self.logger.info(f"Cross-validation F1 scores: {cv_scores}")
            self.logger.info(f"Mean F1 score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            return cv_scores.mean(), cv_scores.std()
        else:
            # Convert negative MSE to positive RMSE
            rmse_scores = np.sqrt(-cv_scores)
            self.logger.info(f"Cross-validation RMSE scores: {rmse_scores}")
            self.logger.info(f"Mean RMSE: {rmse_scores.mean():.4f} (±{rmse_scores.std():.4f})")
            return rmse_scores.mean(), rmse_scores.std()
    
    def calculate_metrics(self, y_true, y_pred, y_prob=None):
        """Calculate performance metrics based on task type"""
        if self.is_classification:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0, average='weighted'),
                'recall': recall_score(y_true, y_pred, zero_division=0, average='weighted'),
                'f1': f1_score(y_true, y_pred, zero_division=0, average='weighted')
            }
            
            # Add ROC AUC if probabilities are provided and binary classification
            if y_prob is not None and len(np.unique(y_true)) == 2:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                metrics['roc_auc'] = auc(fpr, tpr)
                
                # Save ROC curve
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {metrics["roc_auc"]:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                plt.savefig(os.path.join(self.figures_dir, f"roc_curve.png"))
                plt.close()
        else:
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred)
            }
        
        return metrics
    
    def analyze_by_condition(self, model, model_name):
        """Analyze model performance by operating condition"""
        if 'condition' not in self.test_df.columns:
            self.logger.warning("Condition column not found, skipping condition-specific analysis")
            return
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        
        # Create DataFrame with predictions and actual values
        results_df = pd.DataFrame({
            'actual': self.y_test,
            'predicted': y_pred,
            'condition': self.test_df['condition'] if 'condition' in self.test_df.columns else None,
            'bearing_id': self.test_df['bearing_id'] if 'bearing_id' in self.test_df.columns else None
        })
        
        # Group by condition and calculate metrics
        condition_metrics = {}
        for condition, group in results_df.groupby('condition'):
            condition_metrics[f"Condition {condition}"] = self.calculate_metrics(
                group['actual'], group['predicted']
            )
        
        # Create DataFrame from metrics
        condition_report = pd.DataFrame.from_dict(condition_metrics, orient='index')
        
        # Save as CSV
        report_path = os.path.join(self.reports_dir, f"{model_name.lower()}_condition_metrics.csv")
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
        plt.savefig(os.path.join(self.figures_dir, f"{model_name.lower()}_condition_metrics.png"))
        plt.close()
        
        self.logger.info(f"Condition-specific analysis saved for {model_name}")
        
        return condition_metrics
    
    def train_decision_tree(self):
        """Train Decision Tree model"""
        # Find Decision Tree config
        dt_config = next((m for m in self.config["models"] if m["name"] == "DecisionTreeClassifier"), None)
        
        if dt_config is None:
            self.logger.warning("Decision Tree configuration not found in config.yaml")
            return None, None
        
        # Train model
        dt_model = DecisionTreeClassifier(
            max_depth=dt_config["params"]["max_depth"],
            criterion=dt_config["params"]["criterion"],
            random_state=self.config["data"]["seed"]
        )
        
        # Perform cross-validation
        cv_score, cv_std = self.perform_cross_validation(dt_model, self.X_train, self.y_train)
        
        # Train on full training set
        dt_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = dt_model.predict(self.X_test)
        
        # Get probabilities if classification
        y_prob = None
        if self.is_classification and hasattr(dt_model, 'predict_proba'):
            y_proba = dt_model.predict_proba(self.X_test)
            if y_proba.shape[1] == 2:  # Binary classification
                y_prob = y_proba[:, 1]
        
        # Calculate metrics
        metrics = self.calculate_metrics(self.y_test, y_pred, y_prob)
        
        # Add cross-validation score
        if self.is_classification:
            metrics['cv_f1'] = cv_score
            metrics['cv_f1_std'] = cv_std
        else:
            metrics['cv_rmse'] = cv_score
            metrics['cv_rmse_std'] = cv_std
        
        # Save model
        model_path = os.path.join(self.model_dir, "decision_tree.pkl")
        joblib.dump(dt_model, model_path)
        self.logger.info(f"Decision Tree model saved to {model_path}")
        
        # Analyze by condition
        self.analyze_by_condition(dt_model, "DecisionTree")
        
        return dt_model, metrics
    
    def train_random_forest(self):
        """Train Random Forest model"""
        # Find Random Forest config
        rf_config = next((m for m in self.config["models"] if m["name"] == "RandomForestClassifier"), None)
        
        if rf_config is None:
            self.logger.warning("Random Forest configuration not found in config.yaml")
            return None, None
        
        # Train model
        rf_model = RandomForestClassifier(
            n_estimators=rf_config["params"]["n_estimators"],
            random_state=self.config["data"]["seed"],
            n_jobs=-1  # Use all available cores
        )
        
        # Perform cross-validation
        cv_score, cv_std = self.perform_cross_validation(rf_model, self.X_train, self.y_train)
        
        # Train on full training set
        rf_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = rf_model.predict(self.X_test)
        
        # Get probabilities if classification
        y_prob = None
        if self.is_classification and hasattr(rf_model, 'predict_proba'):
            y_proba = rf_model.predict_proba(self.X_test)
            if y_proba.shape[1] == 2:  # Binary classification
                y_prob = y_proba[:, 1]
        
        # Calculate metrics
        metrics = self.calculate_metrics(self.y_test, y_pred, y_prob)
        
        # Add cross-validation score
        if self.is_classification:
            metrics['cv_f1'] = cv_score
            metrics['cv_f1_std'] = cv_std
        else:
            metrics['cv_rmse'] = cv_score
            metrics['cv_rmse_std'] = cv_std
        
        # Save model
        model_path = os.path.join(self.model_dir, "random_forest.pkl")
        joblib.dump(rf_model, model_path)
        self.logger.info(f"Random Forest model saved to {model_path}")
        
        # Analyze by condition
        self.analyze_by_condition(rf_model, "RandomForest")
        
        return rf_model, metrics
    
    def train_xgboost(self):
        """Train XGBoost model"""
        # Find XGBoost config
        xgb_config = next((m for m in self.config["models"] if m["name"] == "XGBClassifier"), None)
        
        if xgb_config is None:
            self.logger.warning("XGBoost configuration not found in config.yaml")
            return None, None
        
        # Create validation set for early stopping
        from sklearn.model_selection import train_test_split
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            self.X_train, self.y_train, 
            test_size=0.2, 
            random_state=self.config["data"]["seed"]
        )
        
        # Select appropriate XGBoost model based on task
        if self.is_classification:
            xgb_model = XGBClassifier(
                n_estimators=xgb_config["params"]["n_estimators"],
                max_depth=xgb_config["params"]["max_depth"],
                learning_rate=xgb_config["params"]["learning_rate"],
                subsample=xgb_config["params"]["subsample"],
                colsample_bytree=xgb_config["params"]["colsample_bytree"],
                objective=xgb_config["params"]["objective"],
                eval_metric=xgb_config["params"]["eval_metric"],
                random_state=self.config["data"]["seed"],
                n_jobs=-1  # Use all available cores
            )
        else:
            xgb_model = XGBRegressor(
                n_estimators=xgb_config["params"]["n_estimators"],
                max_depth=xgb_config["params"]["max_depth"],
                learning_rate=xgb_config["params"]["learning_rate"],
                subsample=xgb_config["params"]["subsample"],
                colsample_bytree=xgb_config["params"]["colsample_bytree"],
                objective='reg:squarederror',
                eval_metric='rmse',
                random_state=self.config["data"]["seed"],
                n_jobs=-1  # Use all available cores
            )
        
        # Perform cross-validation
        cv_score, cv_std = self.perform_cross_validation(xgb_model, self.X_train, self.y_train)
        
        # Train with early stopping
        eval_set = [(X_val, y_val)]
        xgb_model.fit(
            X_train_sub, y_train_sub,
            eval_set=eval_set,
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Get best iteration
        best_iteration = getattr(xgb_model, 'best_iteration', xgb_model.n_estimators)
        self.logger.info(f"XGBoost best iteration: {best_iteration}")
        
        # Retrain on full training set with best iteration
        if self.is_classification:
            final_xgb_model = XGBClassifier(
                n_estimators=best_iteration,
                max_depth=xgb_config["params"]["max_depth"],
                learning_rate=xgb_config["params"]["learning_rate"],
                subsample=xgb_config["params"]["subsample"],
                colsample_bytree=xgb_config["params"]["colsample_bytree"],
                objective=xgb_config["params"]["objective"],
                eval_metric=xgb_config["params"]["eval_metric"],
                random_state=self.config["data"]["seed"],
                n_jobs=-1
            )
        else:
            final_xgb_model = XGBRegressor(
                n_estimators=best_iteration,
                max_depth=xgb_config["params"]["max_depth"],
                learning_rate=xgb_config["params"]["learning_rate"],
                subsample=xgb_config["params"]["subsample"],
                colsample_bytree=xgb_config["params"]["colsample_bytree"],
                objective='reg:squarederror',
                eval_metric='rmse',
                random_state=self.config["data"]["seed"],
                n_jobs=-1
            )
        
        # Train final model on full training set
        final_xgb_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = final_xgb_model.predict(self.X_test)
        
        # Get probabilities if classification
        y_prob = None
        if self.is_classification and hasattr(final_xgb_model, 'predict_proba'):
            y_proba = final_xgb_model.predict_proba(self.X_test)
            if y_proba.shape[1] == 2:  # Binary classification
                y_prob = y_proba[:, 1]
        
        # Calculate metrics
        metrics = self.calculate_metrics(self.y_test, y_pred, y_prob)
        
        # Add cross-validation score
        if self.is_classification:
            metrics['cv_f1'] = cv_score
            metrics['cv_f1_std'] = cv_std
        else:
            metrics['cv_rmse'] = cv_score
            metrics['cv_rmse_std'] = cv_std
        
        # Save model
        model_path = os.path.join(self.model_dir, "xgboost.pkl")
        joblib.dump(final_xgb_model, model_path)
        self.logger.info(f"XGBoost model saved to {model_path}")
        
        # Plot feature importance
        self.plot_feature_importance(final_xgb_model, "XGBoost")
        
        # Analyze by condition
        self.analyze_by_condition(final_xgb_model, "XGBoost")
        
        return final_xgb_model, metrics

    def plot_feature_importance(self, model, model_name):
        """Plot feature importance for a trained model"""
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': model.feature_importances_
            })
            
            # Sort by importance
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            # Plot top 20 features (or fewer if less than 20 features)
            top_n = min(20, len(feature_importance))
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance.head(top_n))
            plt.title(f'Top {top_n} Features - {model_name}')
            plt.tight_layout()
            
            # Save figure
            fig_path = os.path.join(self.figures_dir, f"{model_name.lower()}_feature_importance.png")
            plt.savefig(fig_path)
            plt.close()
            
            self.logger.info(f"Feature importance plot saved to {fig_path}")
            
            # Save feature importance to CSV
            csv_path = os.path.join(self.reports_dir, f"{model_name.lower()}_feature_importance.csv")
            feature_importance.to_csv(csv_path, index=False)

    def train_all_models(self):
        """Train all models defined in config"""
        # Check if data is loaded
        if self.train_df is None or self.test_df is None:
            self.logger.error("Training/test data not available")
            return None
        
        # Set up MLflow tracking
        exp_id = self._setup_mlflow(self.experiment_name)
        
        # Dictionary to store model results
        models_metrics = {}
        
        # Train Decision Tree
        with mlflow.start_run(experiment_id=exp_id, run_name="decision_tree"):
            dt_model, dt_metrics = self.train_decision_tree()
            
            if dt_model is not None:
                # Log parameters and metrics
                mlflow.log_params({
                    'max_depth': dt_model.max_depth,
                    'criterion': dt_model.criterion
                })
                mlflow.log_metrics(dt_metrics)
                
                # Log model
                mlflow.sklearn.log_model(dt_model, "model")
                
                # Store metrics
                models_metrics['DecisionTree'] = dt_metrics
                self.logger.info(f"Decision Tree metrics: {dt_metrics}")
        
        # Train Random Forest
        with mlflow.start_run(experiment_id=exp_id, run_name="random_forest"):
            rf_model, rf_metrics = self.train_random_forest()
            
            if rf_model is not None:
                # Log parameters and metrics
                mlflow.log_params({
                    'n_estimators': rf_model.n_estimators
                })
                mlflow.log_metrics(rf_metrics)
                
                # Log model
                mlflow.sklearn.log_model(rf_model, "model")
                
                # Store metrics
                models_metrics['RandomForest'] = rf_metrics
                self.logger.info(f"Random Forest metrics: {rf_metrics}")
        
        # Train XGBoost
        with mlflow.start_run(experiment_id=exp_id, run_name="xgboost"):
            xgb_model, xgb_metrics = self.train_xgboost()
            
            if xgb_model is not None:
                # Log parameters
                params = {
                    'n_estimators': xgb_model.n_estimators,
                    'max_depth': xgb_model.max_depth,
                    'learning_rate': xgb_model.learning_rate,
                    'subsample': xgb_model.subsample,
                    'colsample_bytree': xgb_model.colsample_bytree
                }
                mlflow.log_params(params)
                
                # Log metrics
                mlflow.log_metrics(xgb_metrics)
                
                # Log model
                mlflow.sklearn.log_model(xgb_model, "model")
                
                # Store metrics
                models_metrics['XGBoost'] = xgb_metrics
                self.logger.info(f"XGBoost metrics: {xgb_metrics}")
        
        # Generate comparative report
        if models_metrics:
            self.generate_comparison_report(models_metrics)
        
        return models_metrics

    def generate_comparison_report(self, models_metrics):
        """Generate a comparison report of model performances"""
        # Create a DataFrame from metrics
        report_df = pd.DataFrame.from_dict(models_metrics, orient='index')
        
        # Save as CSV
        report_path = os.path.join(self.reports_dir, "model_comparison.csv")
        report_df.to_csv(report_path)
        self.logger.info(f"Model comparison report saved to {report_path}")
        
        # Create bar plot for each metric
        for metric in report_df.columns:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=report_df.index, y=metric, data=report_df)
            plt.title(f'Model Comparison - {metric}')
            plt.ylabel(metric)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save figure
            fig_path = os.path.join(self.figures_dir, f"model_comparison_{metric}.png")
            plt.savefig(fig_path)
            plt.close()
        
        # Create a heatmap for all metrics
        plt.figure(figsize=(12, 8))
        sns.heatmap(report_df, annot=True, cmap='viridis', fmt=".3f")
        plt.title('Model Performance Comparison')
        plt.tight_layout()
        
        # Save heatmap
        heatmap_path = os.path.join(self.figures_dir, "model_comparison_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        
        self.logger.info(f"Model comparison visualizations saved to {self.figures_dir}")
        
        # Find best model
        if 'f1' in report_df.columns:  # Classification
            best_metric = 'f1'
        elif 'r2' in report_df.columns:  # Regression
            best_metric = 'r2'
        else:
            best_metric = report_df.columns[0]
        
        best_model = report_df[best_metric].idxmax()
        self.logger.info(f"Best model based on {best_metric}: {best_model} with {best_metric}={report_df.loc[best_model, best_metric]:.4f}")
        
        return best_model