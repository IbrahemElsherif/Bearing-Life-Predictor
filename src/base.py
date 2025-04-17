import os
from pathlib import Path
from datetime import datetime
import yaml
import logging
import mlflow


class Base():
    def __init__(self):
        self.config = self._read_config()
        self.log_dir = self.config['paths']["log_dir"]
        self.experiment_name = self.config['mlflow']['experiment_name']
        
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = self._setup_logger()
    
    def _read_config(self):
        try:
            with open(Path("config.yaml"), "r") as file:
                return yaml.safe_load(file)
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Error loading configuration: {str(e)}")
            # Return default config or raise exception
            return {"paths": {"log_dir": "logs"}, "mlflow": {"experiment_name": "default"}}

    def _setup_logger(self):
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(
            level=logging.INFO,
            format= '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, f"{current_date}.log")),
            ]
        )
        return logger
    
    def _setup_mlflow(self, experiment_name=None):
        if experiment_name is None:
            experiment_name = self.experiment_name
        try:
            exp_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        mlflow.autolog(
            log_input_examples=True,
            log_model_signatures=True,
            log_models=True  
        )
        return exp_id