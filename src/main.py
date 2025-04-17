import argparse
import os
import logging
from data_factory import DataFactory
from feature_engineering import FeatureEngineering
from train_pipeline import ModelTrainer
from prediction import Predictor
from base import Base

class BearingFailurePipeline(Base):
    def __init__(self):
        super().__init__()
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def run_feature_engineering(self):
        """Run feature engineering process"""
        self.logger.info("Starting feature engineering")
        feature_eng = FeatureEngineering()
        features_df = feature_eng.process_all_bearings()
        self.logger.info("Feature engineering completed")
        return features_df
    
    def run_data_preparation(self):
        """Run data preparation process"""
        self.logger.info("Starting data preparation")
        data_factory = DataFactory()
        train_df, test_df = data_factory.prepare_data()
        self.logger.info("Data preparation completed")
        return train_df, test_df
    
    def run_model_training(self):
        """Run model training process"""
        self.logger.info("Starting model training")
        trainer = ModelTrainer()
        metrics = trainer.train_all_models()
        self.logger.info("Model training completed")
        return metrics
    
    def run_prediction(self, model_name=None):
        """Run prediction on production data"""
        if model_name is None:
            # Use first model in config or default to Decision Tree
            model_name = next((m["name"] for m in self.config["models"]), "DecisionTreeClassifier")
            
        self.logger.info(f"Running predictions with {model_name}")
        predictor = Predictor(model_name=model_name)
        results = predictor.predict()
        
        # Save predictions
        output_dir = self.config["paths"]["processed_data_dir"]
        output_path = os.path.join(output_dir, f"{model_name.lower()}_predictions.csv")
        results.to_csv(output_path, index=False)
        
        self.logger.info(f"Predictions saved to {output_path}")
        return results
    
    def run_evaluation(self, model_name=None):
        """Run model evaluation on test data"""
        if model_name is None:
            # Use first model in config or default to Decision Tree
            model_name = next((m["name"] for m in self.config["models"]), "DecisionTreeClassifier")
            
        self.logger.info(f"Evaluating model {model_name}")
        predictor = Predictor(model_name=model_name)
        results, metrics = predictor.evaluate()
        
        # Save evaluation results
        output_dir = self.config["paths"]["reports_dir"]
        output_path = os.path.join(output_dir, f"{model_name.lower()}_evaluation.csv")
        results.to_csv(output_path, index=False)
        
        self.logger.info(f"Evaluation results saved to {output_path}")
        return results, metrics
    
    def run_pipeline(self, steps=None):
        """Run the complete pipeline or specified steps"""
        if steps is None:
            steps = ['feature_engineering', 'data_preparation', 'model_training', 'prediction', 'evaluation']
        
        self.logger.info(f"Starting pipeline with steps: {steps}")
        
        results = {}
        
        if 'feature_engineering' in steps:
            results['features'] = self.run_feature_engineering()
        
        if 'data_preparation' in steps:
            results['train_data'], results['test_data'] = self.run_data_preparation()
        
        if 'model_training' in steps:
            results['model_metrics'] = self.run_model_training()
        
        if 'prediction' in steps:
            results['predictions'] = self.run_prediction()
        
        if 'evaluation' in steps:
            results['evaluation_results'], results['evaluation_metrics'] = self.run_evaluation()
        
        self.logger.info("Pipeline completed successfully")
        return results

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Bearing Failure Prediction Pipeline')
    parser.add_argument('--steps', nargs='+', choices=['feature_engineering', 'data_preparation', 'model_training', 'prediction', 'evaluation'],
                        help='Pipeline steps to run (default: all)')
    parser.add_argument('--model', type=str, help='Model name for prediction/evaluation')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='Set the logging level')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Run pipeline
    try:
        pipeline = BearingFailurePipeline()
        results = pipeline.run_pipeline(steps=args.steps)
        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())