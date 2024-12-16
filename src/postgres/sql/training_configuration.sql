CREATE TABLE IF NOT EXISTS training_configuration (
    "timestamp" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "stage" TEXT, -- training or prediction
    "parameter_code" TEXT,
    "parameter_value" TEXT,
    "created_date" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "update_date" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT pk_train_pred_params PRIMARY KEY ("timestamp", "stage", "parameter_code")
);

INSERT INTO training_configuration ("timestamp", stage, parameter_code, parameter_value, created_date, update_date)
VALUES
    (CURRENT_TIMESTAMP, 'training', 'start_time_window', '35040', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    (CURRENT_TIMESTAMP, 'prediction', 'start_time_window', '100', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    (CURRENT_TIMESTAMP, 'training', 'date_frequency', '15min', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    (CURRENT_TIMESTAMP, 'prediction', 'date_frequency', '15min', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    (CURRENT_TIMESTAMP, 'prediction', 'day_frequency', '30', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    (CURRENT_TIMESTAMP, 'training', 'model_metric', 'root_mean_squared_error', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    (CURRENT_TIMESTAMP, 'training', 'model_names', '["linear_regression", "xgboost_regression", "mlpregressor"]', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    (CURRENT_TIMESTAMP, 'training', 'auto_training', 'True', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    (CURRENT_TIMESTAMP, 'training', 'train_models_older_than', '7', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    (CURRENT_TIMESTAMP, 'training', 'cv_splits', '10', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    (CURRENT_TIMESTAMP, 'training', 'test_size', '0.1', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    (CURRENT_TIMESTAMP, 'training', 'n_iter', '100', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    (CURRENT_TIMESTAMP, 'training', 'early_stop_patience', '10', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    (CURRENT_TIMESTAMP, 'training', 'early_stop_mode', 'min', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    (CURRENT_TIMESTAMP, 'training', 'early_stop_threshold', '0.05', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    (CURRENT_TIMESTAMP, 'training', 'n_study_runs', '1', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    (CURRENT_TIMESTAMP, 'training', 'linear_regression_params', '{}', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    (CURRENT_TIMESTAMP, 'training', 'xgboost_regression_params', '{"xgbregressor__n_estimators":[100, 1000], "xgbregressor__max_depth":[2, 8], "xgbregressor__learning_rate":[0.0, 0.5], "xgbregressor__min_split_loss":[0, 100], "xgbregressor__min_child_weight":[100], "xgbregressor__max_delta_step":[0, 100], "xgbregressor__subsample":[0.2, 1.0], "xgbregressor__reg_alpha":[0, 100], "xgbregressor__reg_lambda":[0, 100]}', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    (CURRENT_TIMESTAMP, 'training', 'mlpregressor_params', '{"mlpregressor__hidden_layer_sizes": [(10,), (512,)], "mlpregressor__alpha": [0.0001], "mlpregressor__learning_rate_init": [0.0001], "mlpregressor__max_iter": [100, 10000]}', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    (CURRENT_TIMESTAMP, 'training', 'direction', 'backward', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    (CURRENT_TIMESTAMP, 'prediction', 'direction', 'backward', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    (CURRENT_TIMESTAMP, 'prediction', 'batch_size', '1000', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    (CURRENT_TIMESTAMP, 'training', 'forecast_acc_threshold', '85', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    (CURRENT_TIMESTAMP, 'prediction', 'forecast_acc_threshold', '85', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);