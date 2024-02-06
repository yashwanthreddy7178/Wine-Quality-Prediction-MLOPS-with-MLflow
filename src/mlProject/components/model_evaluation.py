import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json
from pathlib import Path
#mlflow.set_tracking_uri('https://dagshub.com/yashwanthreddy7178/Wine-Quality-Prediction-MLOPS-with-MLflow.mlflow')
os.environ['MLFLOW_TRACKING_USERNAME']
os.environ['MLFLOW_TRACKING_PASSWORD']

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        try:
            # Load test data and model
            test_data = pd.read_csv(self.config.test_data_path)
            model = joblib.load(self.config.model_path)

            # Prepare test data
            test_x = test_data.drop([self.config.target_column], axis=1)
            test_y = test_data[[self.config.target_column]]

            # Set MLflow registry URI
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            #print(self.config.all_params)
            mlflow.set_experiment('Wine-Quality-Prediction-Experiment1')

            with mlflow.start_run():
                # Make predictions
                predicted_qualities = model.predict(test_x)

                # Evaluate metrics
                (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)

                # Save metrics as local JSON
                scores = {"rmse": rmse, "mae": mae, "r2": r2}
                save_json(path=Path(self.config.metric_file_name), data=scores)
                try:
                    mlflow.log_params(self.config.all_params)
                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("r2", r2)
                    mlflow.log_metric("mae", mae)

                    if tracking_url_type_store != "file":
                        mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
                    else:
                        mlflow.sklearn.log_model(model, "model")

                    print('Model evaluation and logging into MLflow completed successfully.')

                except mlflow.exceptions.MlflowException as e:
                    print(f'MLflow exception: {e}')
                except Exception as e:
                    print(f'Error: {e}')
                    raise e

        except Exception as e:
            print(f'Error: {e}')
            raise e

    
