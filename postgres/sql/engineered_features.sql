CREATE TABLE IF NOT EXISTS engineered_features (
	"timestamp" TIMESTAMP WITHOUT TIME ZONE,
	"latitude" TEXT,
	"longitude" TEXT,
	"variable_code" TEXT,
	"value" REAL,
	"update_date" TIMESTAMP WITHOUT TIME ZONE,
	CONSTRAINT pk_engineered_features PRIMARY KEY ("timestamp", latitude, longitude, variable_code)
);