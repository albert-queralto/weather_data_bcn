CREATE TABLE IF NOT EXISTS Users (
    "email" TEXT PRIMARY KEY,
    "password" TEXT NOT NULL,
    "role" TEXT NOT NULL,
    "created_date" TIMESTAMP WITHOUT TIME ZONE,
    "update_date" TIMESTAMP WITHOUT TIME ZONE
);


INSERT INTO Users (email, password, role, created_date, update_date)
-- Hashed password is 'admin' for testing purposes
VALUES ('admin@admin.com', '$2b$12$dtmXI5WxbAmtFtfBsmeN0.DzYGUc/PT2Dc.XMay0YH/alnnOk1bqW', 'admin', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);