CREATE TABLE IF NOT EXISTS last_processed_data (
                            "latitude" REAL,
                            "longitude" REAL,
							"data_type" TEXT,
							"timestamp" TIMESTAMP WITHOUT TIME ZONE,
							"update_date" TIMESTAMP WITHOUT TIME ZONE,
							CONSTRAINT pk_last_processed_data PRIMARY KEY (
								"latitude",
                                "longitude",
								"data_type"
							)
);