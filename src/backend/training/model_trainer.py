import numpy as np
import pandas as pd
import pickle
from typing import Any, Optional
from dataclasses import dataclass
import shap

# Preprocessing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit, train_test_split

# scikit-learn models
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LinearRegression

# Import custom logger
from utils.custom_logger import CustomLogger

# Import custom modules
from loaders.training import ModelVersioningManager
from training.optuna_tuner import OptunaStudy
from utils.metrics import RegressionMetrics


@dataclass
class DataConfig:
    df: pd.DataFrame
    target_variable: str
    latitude: float
    longitude: float
    test_size: float
    batch_size: int


@dataclass
class TrainingConfig:
    cv_splits: int
    n_iter: int
    early_stop_patience: int
    early_stop_mode: str
    early_stop_threshold: float
    n_study_runs: int


@dataclass
class ModelConfig:
    models: dict[str, Any]
    params: dict[str, dict[str, Any]]
    model_type: str
    model_metric_name: str
    forecast_accuracy_threshold: float

@dataclass
class DataPreprocessor:
    data_config: DataConfig
    
    def preprocess(self):
        X = self.data_config.df.drop(columns=self.data_config.target_variable).copy()
        y = self.data_config.df[self.data_config.target_variable].copy()

        X = X.astype(float)
        y = y.astype(float)
        
        train_size = int(len(X) * (1 - self.data_config.test_size))
        train_indices = X.index[:train_size]
        test_indices = X.index[train_size:]

        X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.data_config.test_size,
                shuffle=False
        )

        X_train.index = pd.to_datetime(train_indices)
        X_test.index = pd.to_datetime(test_indices)
        y_train.index = pd.to_datetime(train_indices)
        y_test.index = pd.to_datetime(test_indices)
        
        return X_train, X_test, y_train, y_test

@dataclass
class ModelTrainer:
    """
    Contains different methods to train machine learning models.
    """
    model_manager: ModelVersioningManager
    logger: CustomLogger
    data_preprocessor: DataPreprocessor
    training_config: TrainingConfig
    model_config: ModelConfig

    def train(self) -> None:
        """
        Performs cross-validation for the models in the models_dict dictionary. For
        each model uses the specific parameters selected from the params_dict.
        """
        if not self.model_config.models:
            raise ValueError("The models dictionary is empty.")
        
        X_train, X_test, y_train, y_test = self.data_preprocessor.preprocess()

        models_df = pd.DataFrame()
        for model_name, model in self.model_config.models.items():
            # ----------------- DEBUGGING -----------------
            self.logger.debug(f"Lat: {self.data_preprocessor.data_config.latitude} Lon: {self.data_preprocessor.data_config.longitude}")
            self.logger.debug(f"Training model: {model_name} for {self.data_preprocessor.data_config.target_variable}")
            # ---------------------------------------------
            # Train the models using cross-validation
            estimator, scaler, polynomial, val_score = \
                        self._model_trainer(X_train, y_train, model_name, model)
            # ----------------- DEBUGGING -----------------
            self.logger.debug(f"The model {model_name} was trained successfully.")
            self.logger.debug(f"Model object: {estimator}")
            self.logger.debug(f"Validation score is: {val_score}")
            # ---------------------------------------------

            X_train_tmp, X_test_tmp = self._apply_polynomial(X_train, X_test, polynomial)
            X_train_tmp, X_test_tmp, _, y_test_tmp = self._apply_scaler(
                X_train,
                X_test,
                y_train,
                y_test,
                scaler
            )

            test_scores = self._eval_model(
                X_test_tmp,
                y_test_tmp,
                model_name,
                estimator,
                scaler
            )
            
            self.logger.debug(f"The test score is: {test_scores[model_name][self.model_config.model_metric_name]}")

            feature_importance_variables, feature_importance_values = \
                self._get_feature_importance(estimator, X_train_tmp, X_test_tmp)

            self.logger.debug(f"Feature importance variables: {feature_importance_variables}")
            self.logger.debug(f"Feature importance values: {feature_importance_values}")

            model_date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            model_version = self.model_manager.get_model_version(
                model_type=self.model_config.model_type,
                latitude=self.data_preprocessor.data_config.latitude,
                longitude=self.data_preprocessor.data_config.longitude,
                target_variable=self.data_preprocessor.data_config.target_variable
            )

            data = {
                'model_date': model_date,
                'model_name': model_name,
                'model_type': self.model_config.model_type,
                'latitude': self.data_preprocessor.data_config.latitude,
                'longitude': self.data_preprocessor.data_config.longitude,
                'target_variable': self.data_preprocessor.data_config.target_variable,
                'model_feature_names': str(estimator.feature_names_in_.tolist()),
                'model_features_count': estimator.n_features_in_,
                'model_version': model_version,
                'model_parameters': str(estimator.get_params()),
                'model_metric_name': self.model_config.model_metric_name,
                'model_metric_validation_value': val_score,
                'model_metric_test_value': test_scores[model_name][self.model_config.model_metric_name],
                'standard_scaler_binary': pickle.dumps(scaler),
                'model_binary': pickle.dumps(estimator),
                'model_forecast_accuracy': test_scores[model_name]['forecast_acc'],
                'polynomial_transformer_binary': pickle.dumps(polynomial) if polynomial is not None else None,
                'feature_importance_variables': str(feature_importance_variables),
                'feature_importance_values': str(feature_importance_values)
            }

            model_col = pd.DataFrame.from_dict(data, orient='index')
            models_df = pd.concat([models_df, model_col.T], axis=0)
            models_df.reset_index(drop=True, inplace=True)

            # ----------------- DEBUGGING -----------------
            self.logger.debug(f"Model date: {model_date}")
            self.logger.debug(f"Model version: {model_version}")
            self.logger.debug(f"Last data received: {self.data_preprocessor.data_config.df.index[-1]}")
            # ---------------------------------------------
        self.model_manager.save(models_df, self.data_preprocessor.data_config.batch_size)

    def _get_feature_importance(self,
            model: Any, 
            X_train: pd.DataFrame,
            X_test: pd.DataFrame
        ) -> tuple[list[str], list[str]]:
        
        explainer = shap.Explainer(model.predict, X_train, algorithm='auto')
        shap_values = explainer(X_test, max_evals=10000)
        importance_values = np.abs(shap_values.values).mean(0).tolist()
        importance_variables = X_train.columns.tolist()

        return importance_variables, importance_values

    def _apply_polynomial(self,
            X_train: pd.DataFrame, 
            X_test: pd.DataFrame,
            polynomial: Any, 
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
        def transform_and_create_df(X, polynomial):
            X_tmp = polynomial.transform(X)
            return pd.DataFrame(
                X_tmp,
                index=X.index,
                columns=polynomial.get_feature_names_out(X.columns)
            )

        if polynomial is not None:
            X_train = transform_and_create_df(X_train, polynomial)
            X_test = transform_and_create_df(X_test, polynomial)

        return X_train, X_test

    def _apply_scaler(self,
            X_train: pd.DataFrame, 
            X_test: pd.DataFrame,
            y_train: pd.Series, 
            y_test: pd.Series,
            scaler: StandardScaler
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        def scale_and_split(df, scaler, target_name):
            scaled_df = pd.DataFrame(
                scaler.transform(df),
                index=df.index,
                columns=df.columns
            )
            return scaled_df.drop(columns=target_name), scaled_df[target_name]

        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        X_train_scaled, y_train_scaled = scale_and_split(train_df, scaler, y_train.name)
        X_test_scaled, y_test_scaled = scale_and_split(test_df, scaler, y_test.name)

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

    def _eval_model(self,
                X_test: pd.DataFrame,
                y_test: pd.Series,
                model_name: str,
                model: Any,
                scaler: Any,
                percentage: Optional[int] = None
                ) -> dict[str, dict[str, float]]:
        test_scores = {}
        if percentage is not None:
            model_name = f"{model_name}_{percentage}"

        y_preds = model.predict(X_test)

        unscaled_y_test, unscaled_y_pred = self._get_unscaled_data(X_test, y_test, y_preds, scaler)

        model_score = RegressionMetrics()(
            metric_name=self.model_config.model_metric_name,
            y_true=unscaled_y_test,
            y_pred=unscaled_y_pred,
        )

        forecast_accuracy = self._get_train_forecast_accuracy(unscaled_y_test, unscaled_y_pred)

        if model_name not in test_scores:
            test_scores[model_name] = {'forecast_acc': [], f'{self.model_config.model_metric_name}': []}
        test_scores[model_name]['forecast_acc'] = forecast_accuracy
        test_scores[model_name][self.model_config.model_metric_name] = model_score
        return test_scores

    def _get_unscaled_data(self, X_test: pd.DataFrame, y_test: pd.Series, y_preds: np.ndarray, scaler: Any) -> tuple[pd.Series, pd.Series]:
        original_data = pd.concat([X_test, y_test], axis=1)
        unscaled_original_data = pd.DataFrame(
            scaler.inverse_transform(original_data),
            columns=original_data.columns,
            index=original_data.index
        )

        preds_series = pd.Series(y_preds, index=original_data.index, name=y_test.name)
        pred_data = pd.concat([X_test, preds_series], axis=1)

        unscaled_pred_data = pd.DataFrame(
            scaler.inverse_transform(pred_data),
            columns=pred_data.columns,
            index=pred_data.index
        )

        unscaled_y_test = unscaled_original_data[y_test.name]
        unscaled_y_pred = unscaled_pred_data[y_test.name]

        return unscaled_y_test, unscaled_y_pred

    def _get_train_forecast_accuracy(self, y_test: pd.Series, y_preds: pd.Series) -> float:
        try:
            forecast_accuracy = np.mean((1 - (abs(y_preds - y_test) / y_test)) * 100)
            forecast_accuracy = max(forecast_accuracy, 0)
            return np.clip(forecast_accuracy, 0, 100)
        except Exception:
            return 0

    def _create_pipeline(self, model_name: str, model: Any) -> Pipeline:
        """
        Creates the pipeline for the models based on their name.
        """
        pipeline = make_pipeline(StandardScaler(), model, memory=None)

        if model_name in self.model_config.params.keys():
            param_grid = self.model_config.params[model_name]
        else:
            # ----------------- DEBUGGING -----------------
            self.logger.debug(f"The model name {model_name} was not found in the parameters dictionary.")
            # ---------------------------------------------
            raise KeyError('The model name was not found in the parameters dictionary.')
        if 'polynomial' in model_name:
            pipeline = make_pipeline(PolynomialFeatures(), StandardScaler(), model, memory=None)

        # Set the parameters for the pipeline and train the model
        if 'linear_regression' in model_name or 'polynomial' in model_name:
            pipeline.set_params(**param_grid)
        return pipeline

    def _model_trainer(self,
            X: pd.DataFrame, y: pd.Series, model_name: str, model: Any
        ) -> tuple[Any, StandardScaler, Optional[PolynomialFeatures], float]:
        """
        Trains models with Optuna if they are not LinearRegression. Otherwise,
        trains the LinearRegression model.
        """
        for _ in range(self.training_config.n_study_runs):
            estimator, scaler, polynomial, val_score = None, None, None, None
            pipeline = self._create_pipeline(model_name, model)
            if model_name in self.model_config.params.keys():
                param_grid = self.model_config.params[model_name]
            else:
                # ----------------- DEBUGGING -----------------
                self.logger.debug(f"The model name {model_name} was not found in the parameters dictionary.")
                # ---------------------------------------------
                raise KeyError('The model name was not found in the parameters dictionary.')
            if 'linear_regression' in model_name or 'polynomial' in model_name:
                estimator, scaler, polynomial, val_score = \
                                        self.linear_regression_model(pipeline, X, y)
            else:
                estimator, scaler, val_score = self._optuna_hyperparam_search(
                    pipeline,
                    param_grid,
                    model_name,
                    X, y
                )
            if val_score >= self.model_config.forecast_accuracy_threshold:
                break
        return estimator, scaler, polynomial, val_score

    def linear_regression_model(self,
            pipeline: Pipeline, X: pd.DataFrame, y: pd.Series
        ) -> tuple[LinearRegression, StandardScaler, Optional[PolynomialFeatures], float]:
        """
        Performs multilinear regression on the input dataframe and target variable.
        """
        poly = self._apply_polynomial_features(pipeline, X)
        scaler, X_scaled, y_scaled = self._scale_data(pipeline, X, y)
        lin_reg = pipeline.named_steps['linearregression']
        lr, val_score = self._linear_reg_time_series_cv(X_scaled, y_scaled, lin_reg)
        return lr, scaler, poly, val_score

    def _apply_polynomial_features(self, pipeline: Pipeline, X: pd.DataFrame) -> Optional[PolynomialFeatures]:
        """
        Applies polynomial features transformation if present in the pipeline.
        """
        if 'polynomialfeatures' in pipeline.named_steps:
            poly = pipeline.named_steps['polynomialfeatures']
            X_transformed = poly.fit_transform(X)
            X = pd.DataFrame(X_transformed, columns=poly.get_feature_names_out(), index=X.index)
            return poly
        return None

    def _scale_data(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> tuple[StandardScaler, pd.DataFrame, pd.Series]:
        """
        Scales the data using the scaler in the pipeline.
        """
        scaler = pipeline.named_steps['standardscaler']
        data = pd.concat([X, y], axis=1)
        scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=X.index)
        X_scaled = scaled_data.drop(columns=y.name)
        y_scaled = scaled_data[y.name]
        return scaler, X_scaled, y_scaled

    def _linear_reg_time_series_cv(self,
            X: pd.DataFrame, y: pd.Series, lr: LinearRegression
        ) -> tuple[LinearRegression, float]:
        """
        Implements cross-validation to time series forecasting.
        Calculates the coefficients for each fold and the average values and uses
        the average coefficients to fit the whole data.
        """
        coefs, scores = [], []

        tscv = TimeSeriesSplit(n_splits=self.training_config.cv_splits)
        for train_index, val_index in tscv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            lr.fit(X_train, y_train)
            y_preds = pd.Series(lr.predict(X_val), index=y_val.index, name=y_val.name)

            model_score = RegressionMetrics()(
                metric_name=self.model_config.model_metric_name,
                y_true=y_val,
                y_pred=y_preds,
            )

            coefs.append(lr.coef_)
            scores.append(model_score)

        mean_coefs = np.mean(coefs, axis=0)
        mean_score = float(np.mean(scores))

        lr.coef_ = mean_coefs
        return lr.fit(X, y), mean_score

    def _optuna_hyperparam_search(self,
            pipeline: Pipeline,
            param_grid: dict,
            model_name: str,
            X: pd.DataFrame,
            y: pd.Series
        ) -> tuple[Any, StandardScaler, float]:
        """
        Implements Optuna hyperparameter optimization for models that are not
        based on the LinearRegression class.
        """
        scaler, X_scaled, y_scaled = self._scale_data(pipeline, X, y)
        model = pipeline.steps[1][1]
        optuna_opt = self._create_optuna_study(param_grid, model_name)

        best_study, storage = self._run_optuna_study(optuna_opt, model, X_scaled, y_scaled)
        model.set_params(**best_study.best_params)
        best_score = best_study.best_value
        optuna_opt.delete_storage(storage)
        model.fit(X_scaled, y_scaled)
        
        return model, scaler, best_score

    def _create_optuna_study(self, param_grid: dict, model_name: str) -> OptunaStudy:
        """
        Creates an OptunaStudy object with the given parameters.
        """
        return OptunaStudy(
            params=param_grid,
            model_name=model_name,
            task_type=self.model_config.model_type,
            scoring=self.model_config.model_metric_name,
            test_size=self.data_preprocessor.data_config.test_size,
            early_stop_patience=self.training_config.early_stop_patience,
            early_stop_mode=self.training_config.early_stop_mode,
            early_stop_threshold=self.training_config.early_stop_threshold,
        )

    def _run_optuna_study(self, optuna_opt: OptunaStudy, model: Any, X: pd.DataFrame, y: pd.Series) -> tuple[Any, str]:
        """
        Runs the Optuna study and returns the best study and storage.
        """
        best_study = None
        storage = ''
        for _ in range(self.training_config.n_study_runs):
            study, _, storage = optuna_opt.run_parallel_optuna(
                X, y,
                model=model,
                cv=self.training_config.cv_splits, 
                n_trials=self.training_config.n_iter,
                n_jobs=1
            )

            if (
                best_study is None or (
                study.best_value is not None and
                study.best_value > best_study.best_value
            )):
                best_study = study
            if (
                best_study is not None and (
                best_study.best_value is not None and
                best_study.best_value > self.training_config.early_stop_threshold
            )):
                break
        return best_study, storage