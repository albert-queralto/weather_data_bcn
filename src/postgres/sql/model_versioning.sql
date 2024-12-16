CREATE TABLE IF NOT EXISTS model_versioning (
	"model_date" TIMESTAMP WITHOUT TIME ZONE,
	"model_name" TEXT,
	"model_type" TEXT, -- "clf", "reg"
	"latitude" TEXT,
	"longitude" TEXT,
	"target_variable" TEXT,
	"model_feature_names" TEXT,
	"model_features_count" INTEGER,
	"model_version" INTEGER,
	"model_parameters" TEXT,
	"model_metric_name" TEXT,
	"model_metric_validation_value" REAL,
	"model_metric_test_value" REAL,
	"model_forecast_accuracy" REAL,
	"standard_scaler_binary" BYTEA,
	"polynomial_transformer_binary" BYTEA,
	"model_binary" BYTEA,
	"feature_importance_variables" TEXT,
	"feature_importance_values" TEXT,
	CONSTRAINT pk_model_versioning PRIMARY KEY (
		"model_date",
		"model_name",
		"model_type",
		"latitude",
		"longitude",
		"target_variable"
	)
);